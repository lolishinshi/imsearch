pub mod invlists;
pub mod quantizer;
mod utils;

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use crossbeam_channel::bounded;
pub use invlists::*;
use itertools::izip;
use log::debug;
pub use quantizer::*;
use rayon::prelude::*;
use tokio::time::Instant;

use crate::hamming::knn_hamming;
use crate::ivf::utils::TopKNeighbors;

pub type IvfHnswDisk = IvfHnsw<32, HnswQuantizer<32>, OnDiskInvlists<32>>;
pub type IvfHnswArray = IvfHnsw<32, HnswQuantizer<32>, ArrayInvertedLists<32>>;

#[derive(Debug)]
pub struct SearchResult {
    /// 量化耗时
    pub quantizer_time: Duration,
    /// 每个线程的总 IO 耗时
    pub io_time: Duration,
    /// 每个线程的总耗时
    pub compute_time: Duration,
    /// 总搜索耗时
    pub search_time: Duration,
    /// 搜索结果
    pub neighbors: Vec<Neighbor>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Neighbor {
    // 注意此处 distance 排在前面，保证自动 derive 的 Ord 正确
    pub distance: u32,
    pub id: u64,
}

impl Default for Neighbor {
    fn default() -> Self {
        Self { id: 0, distance: u32::MAX }
    }
}

/// 基于 HNSW 量化器的倒排索引
pub struct IvfHnsw<const N: usize, Q: Quantizer<N>, I: InvertedLists<N>> {
    pub quantizer: Q,
    pub invlists: I,
}

impl<const N: usize, Q: Quantizer<N>, I: InvertedLists<N>> IvfHnsw<N, Q, I>
where
    I: Sync,
    Q: Sync,
{
    /// 往倒排列表中增加一组向量，并使用自定义 id
    pub fn add(&mut self, data: &[[u8; N]], ids: &[u64]) -> Result<()> {
        debug!("quantizing {} vectors", data.len());
        let vlists = self.quantizer.search(data, 1)?;
        let centroids = self.quantizer.centroids()?;
        debug!("adding {} vectors", data.len());
        for (list_no, id, bvec) in izip!(vlists, ids, data) {
            assert!(list_no != -1);
            // 此处将向量和中心点异或，这样可以在后续压缩过程中节省空间
            let bcent = &centroids[list_no as usize];
            let bvec = xor(bvec, bcent);
            self.invlists.add_entry(list_no as usize, *id, &bvec)?;
        }
        Ok(())
    }
}

impl<const N: usize> IvfHnsw<N, HnswQuantizer<N>, ArrayInvertedLists<N>> {
    pub fn open_array<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        let quantizer = HnswQuantizer::open(path.join("quantizer.bin"))?;

        let nlist = quantizer.nlist();
        let invlists = ArrayInvertedLists::<N>::new(nlist);
        Ok(Self { quantizer, invlists })
    }
}

impl<const N: usize> IvfHnsw<N, HnswQuantizer<N>, OnDiskInvlists<N>> {
    pub fn open_disk<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        let quantizer = HnswQuantizer::open(path.join("quantizer.bin"))?;

        let nlist = quantizer.nlist();
        let invlists = OnDiskInvlists::<N>::load(path.join("invlists.bin"))?;
        assert_eq!(nlist, invlists.nlist(), "nlist mismatch");
        Ok(Self { quantizer, invlists })
    }

    /// 在索引中搜索一组向量，并返回搜索结果
    /// 注意：搜索结果的大小并不等于 len(data) * k，也不保证顺序，因为对于 imsearch 应用场景来说来说这是可以接受的
    pub fn search(&self, data: &[[u8; N]], k: usize, nprobe: usize) -> Result<SearchResult> {
        let start = Instant::now();

        // 量化得到每个向量对应的倒排列表序号
        let vlists = self.quantizer.search(data, nprobe)?;
        let quantizer_time = start.elapsed();

        let io_time = AtomicU64::new(0);
        let compute_time = AtomicU64::new(0);
        let neighbors = Arc::new(Mutex::new(
            (0..data.len()).map(|_| TopKNeighbors::new(k)).collect::<Vec<_>>(),
        ));
        let centroids = self.quantizer.centroids()?;

        // 对 vlists 按 offset 排序，变成顺序读取
        let vlists = self.invlists.reorder_lists(&vlists);

        std::thread::scope(|s| {
            // 倒排列表读取线程
            let (tx1, rx1) = bounded(32);
            let time = &io_time;
            s.spawn(move || {
                vlists.par_iter().for_each(|(i, list_no)| {
                    let t = Instant::now();
                    let (ids, codes) = self.invlists.get_list(*list_no).unwrap();
                    let idx = i / nprobe;
                    time.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    tx1.send((idx, *list_no, ids, codes)).unwrap();
                });
            });

            // 距离计算线程
            for _ in 0..num_cpus::get() {
                let time = &compute_time;
                let rx = rx1.clone();
                let neighbors = neighbors.clone();
                s.spawn(move || {
                    while let Ok((idx, list_no, ids, codes)) = rx.recv() {
                        let t = Instant::now();
                        // codes 中的值和 centroid 异或过，这里需要给 xq 也异或一次才能确保汉明距离正确
                        let xq = xor(&data[idx], &centroids[list_no]);
                        // 查询最近的 k 个邻居
                        let n = knn_hamming::<N>(&xq, &codes, k);
                        let n = n.iter().map(|&(i, d)| Neighbor { id: ids[i], distance: d });
                        neighbors.lock().unwrap()[idx].extend(n);
                        time.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                });
            }
        });

        // NOTE: 此处没有进行任何排序，因为 imsearch 不关心顺序，只关心频率
        let neighbors = Arc::into_inner(neighbors)
            .unwrap()
            .into_inner()
            .unwrap()
            .into_par_iter()
            .map(|l| l.into_vec())
            .flatten()
            .collect::<Vec<_>>();

        let search_time = start.elapsed() - quantizer_time;
        Ok(SearchResult {
            quantizer_time,
            io_time: Duration::from_nanos(io_time.load(Ordering::Relaxed)),
            compute_time: Duration::from_nanos(compute_time.load(Ordering::Relaxed)),
            search_time,
            neighbors,
        })
    }
}

// 对两个向量进行异或
#[inline(always)]
fn xor<const N: usize>(va: &[u8; N], vb: &[u8; N]) -> [u8; N] {
    let mut vc = [0u8; N];
    for i in 0..N {
        vc[i] = va[i] ^ vb[i];
    }
    vc
}
