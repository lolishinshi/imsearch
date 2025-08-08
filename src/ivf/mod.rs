pub mod invlists;
pub mod quantizer;

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use crossbeam_channel::bounded;
pub use invlists::*;
pub use quantizer::*;
use tokio::time::Instant;

use crate::hamming::knn_hamming;

pub type IvfHnswDisk = IvfHnsw<32, HnswQuantizer<32>, OnDiskInvlists<32>>;

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

#[derive(Debug, Clone)]
pub struct Neighbor {
    pub id: u64,
    pub distance: u32,
}

impl Default for Neighbor {
    fn default() -> Self {
        Self { id: 0, distance: u32::MAX }
    }
}

/// 基于 HNSW 量化器的倒排索引
pub struct IvfHnsw<const N: usize, Q: Quantizer<N>, I: InvertedLists<N>> {
    quantizer: Q,
    invlists: I,
}

impl<const N: usize, Q: Quantizer<N>, I: InvertedLists<N>> IvfHnsw<N, Q, I>
where
    I: Sync,
    Q: Sync,
{
    /// 往倒排列表中增加一组向量，并使用自定义 id
    pub fn add(&mut self, data: &[[u8; N]], ids: &[u64]) -> Result<()> {
        let vlists = self.quantizer.search(data, 1)?;
        let centroids = self.quantizer.centroids()?;
        for (&list_no, (&id, bvec)) in vlists.iter().zip(ids.iter().zip(data)) {
            assert!(list_no != -1);
            // 此处将向量和中心点异或，这样可以在后续压缩过程中节省空间
            let bcent = &centroids[list_no as usize];
            let bvec = xor(bvec, bcent);
            self.invlists.add_entry(list_no as usize, id, &bvec)?;
        }
        Ok(())
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
        let neighbors = Arc::new(Mutex::new(vec![]));
        let centroids = self.quantizer.centroids()?;

        std::thread::scope(|s| {
            let (tx, rx) = bounded(32);

            // 此处将数据分块，并发送到子线程进行预取
            // 其中由于 data 中的每个向量对应 nprobe 个 list，所以 vlists 分块大小需要乘以 nprobe
            let chunk_size = (data.len() / 32).max(4); // 避免为 0
            for chunk in data.chunks(chunk_size).zip(vlists.chunks(chunk_size * nprobe)) {
                let tx = tx.clone();
                let io_time = &io_time;
                s.spawn(move || {
                    // 遍历每条向量和对应的 list
                    for (xq, lists) in chunk.0.iter().zip(chunk.1.chunks_exact(nprobe)) {
                        let t = Instant::now();
                        // 收集对应倒排列表中的所有 id 和 code
                        // TODO: 是否可以考虑改成 Vec<Vec<u64>> 和 Vec<Vec<[u8; N]>> 来避免拷贝？
                        let mut all_ids = Vec::with_capacity(lists.len());
                        let mut all_codes = Vec::with_capacity(lists.len());
                        let mut all_centroids = Vec::with_capacity(lists.len());
                        for &list_no in lists {
                            if list_no == -1 {
                                continue;
                            }
                            let (ids, codes) = self.invlists.get_list(list_no as usize).unwrap();
                            all_ids.push(ids);
                            all_codes.push(codes);
                            all_centroids.push(centroids[list_no as usize]);
                        }
                        io_time.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        tx.send((xq, all_ids, all_codes, all_centroids)).unwrap();
                    }
                });
            }

            for _ in 0..num_cpus::get() {
                let neighbors = neighbors.clone();
                let calc_time = &compute_time;
                let rx = rx.clone();
                s.spawn(move || {
                    while let Ok((xq, ids, codes, centroids)) = rx.recv() {
                        let t = Instant::now();
                        let mut v = Vec::with_capacity(k * centroids.len());
                        for ((ids, codes), centroids) in
                            ids.iter().zip(codes.iter()).zip(centroids.iter())
                        {
                            let xq = xor(xq, centroids);
                            let r = knn_hamming::<N>(&xq, &codes, k);
                            v.extend(r.iter().map(|&(i, d)| Neighbor { id: ids[i], distance: d }))
                        }
                        // 我们只需要排名前 k 的结果
                        v.select_nth_unstable_by_key(k, |n| n.distance);
                        v.truncate(k);
                        neighbors.lock().unwrap().extend(v);
                        calc_time.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                });
            }
        });

        // NOTE: 此处没有进行任何排序，因为 imsearch 不关心顺序，只关心频率
        let neighbors = Arc::into_inner(neighbors).unwrap().into_inner().unwrap();

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

impl<const N: usize> IvfHnsw<N, HnswQuantizer<N>, ArrayInvertedLists<N>> {
    pub fn open_array<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        let quantizer = HnswQuantizer::open(path.join("quantizer.bin"))?;

        let nlist = quantizer.nlist();
        let invlists = ArrayInvertedLists::<N>::new(nlist);
        Ok(Self { quantizer, invlists })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        save_invlists(&self.invlists, path)?;
        Ok(())
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
