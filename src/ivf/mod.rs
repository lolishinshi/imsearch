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
use rayon::ThreadPool;
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
    // 倒排列表读取线程池
    pub pool: ThreadPool,
    // 倒排列表读取线程数
    pub threads: usize,
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

        let pool = rayon::ThreadPoolBuilder::new().num_threads(num_cpus::get()).build()?;

        Ok(Self { quantizer, invlists, pool, threads: num_cpus::get() })
    }
}

impl<const N: usize> IvfHnsw<N, HnswQuantizer<N>, OnDiskInvlists<N>> {
    pub fn open_disk<P: AsRef<Path>>(path: P, threads: usize) -> Result<Self> {
        let path = path.as_ref();

        let quantizer = HnswQuantizer::open(path.join("quantizer.bin"))?;

        let nlist = quantizer.nlist();
        let invlists = OnDiskInvlists::<N>::load(path.join("invlists.bin"))?;
        assert_eq!(nlist, invlists.nlist(), "nlist mismatch");

        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;

        Ok(Self { quantizer, invlists, pool, threads })
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

        rayon::scope(|s| {
            // 倒排列表读取线程
            let (tx1, rx1) = bounded(self.threads * 2);
            let time = &io_time;
            s.spawn(move |_| {
                // 此处如果按照 nprobe 分组，并批量读取组内的每个列表，反而会导致性能下降
                self.pool.install(move || {
                    vlists.par_iter().enumerate().for_each(|(i, &list_no)| {
                        if list_no == -1 {
                            return;
                        }
                        let t = Instant::now();
                        let list_no = list_no as usize;
                        let (ids, codes) = self.invlists.get_list(list_no).unwrap();
                        let idx = i / nprobe;
                        time.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        tx1.send((idx, list_no, ids, codes)).unwrap();
                    });
                });
            });

            rx1.iter().par_bridge().for_each(|(idx, list_no, ids, codes)| {
                let t = Instant::now();
                let xq = xor(&data[idx], &centroids[list_no]);
                let n = knn_hamming::<N>(&xq, &codes, k);
                let n = n.iter().map(|&(i, d)| Neighbor { id: ids[i], distance: d });
                neighbors.lock().unwrap()[idx].extend(n);
                compute_time.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            });
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
