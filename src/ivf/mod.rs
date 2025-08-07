pub mod invlists;
pub mod quantizer;

use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use bytemuck::checked::cast_slice;
use crossbeam_channel::bounded;
pub use invlists::*;
pub use quantizer::*;
use rayon::prelude::*;
use tokio::time::Instant;

use crate::hamming::knn_hamming;

pub type IvfHnswDisk = IvfHnsw<32, HnswQuantizer<32>, OnDiskInvlists<32>>;

#[derive(Debug)]
pub struct SeachResult {
    /// 量化耗时
    pub quantizer_time: Duration,
    /// 每个线程的总 IO 耗时
    pub io_time: Duration,
    /// 每个线程的总耗时
    pub thread_time: Duration,
    /// 总搜索耗时
    pub search_time: Duration,
    /// 搜索结果
    pub neighbors: Vec<Neighbor>,
}

#[derive(Debug, Clone)]
pub struct Neighbor {
    pub id: i64,
    pub distance: u32,
}

impl Default for Neighbor {
    fn default() -> Self {
        Self { id: -1, distance: u32::MAX }
    }
}

pub struct IvfHnsw<const N: usize, Q: Quantizer<N>, I: InvertedLists<N>> {
    quantizer: Q,
    invlists: I,
    nlist: usize,
}

impl<'a, const N: usize, Q: Quantizer<N>, I: InvertedLists<N>> IvfHnsw<N, Q, I>
where
    I: 'a + Sync,
    Q: Sync,
{
    // TODO: 这里的 API 能不能改成接受 IntoIterator ？
    pub fn add(&mut self, data: &[[u8; N]], ids: &[u64]) -> Result<()> {
        let vlists = self.quantizer.search(data, 1)?;
        for ((xq, id), list_no) in data.iter().zip(ids).zip(vlists) {
            let (xq, _) = xq.as_chunks();
            self.invlists.add_entries(list_no as usize, &[*id], xq)?;
        }
        Ok(())
    }

    pub fn search(&'a self, data: &[[u8; N]], k: usize, nprobe: usize) -> Result<SeachResult> {
        let start = Instant::now();
        let vlists = self.quantizer.search(data, nprobe)?;
        let quantizer_time = start.elapsed();

        let io_time = AtomicU64::new(0);
        let calc_time = AtomicU64::new(0);

        let neighbors = Arc::new(Mutex::new(vec![]));

        std::thread::scope(|s| {
            let (tx, rx) = bounded(32);

            let chunk_size = data.len() / 32;

            // 此处将数据分块，并发送到子线程进行预取
            // 其中由于 data 中的每个向量对应 nprobe 个 list，所以 vlists 分块大小需要乘以 nprobe
            for chunk in data.chunks(chunk_size).zip(vlists.chunks(chunk_size * nprobe)) {
                let tx = tx.clone();
                let io_time = &io_time;
                s.spawn(move || {
                    // 遍历每条向量和对应的 list
                    for (xq, lists) in chunk.0.iter().zip(chunk.1.chunks_exact(nprobe)) {
                        let t = Instant::now();
                        // 收集对应倒排列表中的所有 id 和 code
                        let mut all_ids: Vec<u64> = vec![];
                        let mut all_codes: Vec<[u8; N]> = vec![];
                        for &list_no in lists {
                            if list_no == -1 {
                                continue;
                            }
                            let (ids, codes) = self.invlists.get_list(list_no as usize).unwrap();
                            all_ids.extend(ids);
                            all_codes.extend(codes);
                        }
                        io_time.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        tx.send((xq, all_ids, all_codes)).unwrap();
                    }
                });
            }

            for _ in 0..num_cpus::get() {
                let neighbors = neighbors.clone();
                let calc_time = &calc_time;
                let rx = rx.clone();
                s.spawn(move || {
                    while let Ok((xq, ids, codes)) = rx.recv() {
                        let t = Instant::now();
                        let r = knn_hamming::<N>(xq, &codes, k);
                        let v = r
                            .into_iter()
                            .map(|(i, d)| Neighbor { id: ids[i] as i64, distance: d })
                            .collect::<Vec<_>>();
                        neighbors.lock().unwrap().extend(v);
                        calc_time.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                });
            }
        });

        // NOTE: 此处没有进行任何排序，因为 imsearch 不关心顺序，只关心频率
        let neighbors = Arc::into_inner(neighbors).unwrap().into_inner().unwrap();

        let search_time = start.elapsed() - quantizer_time;
        Ok(SeachResult {
            quantizer_time,
            io_time: Duration::from_nanos(io_time.load(Ordering::Relaxed)),
            thread_time: Duration::from_nanos(calc_time.load(Ordering::Relaxed)),
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
        Ok(Self { quantizer, invlists, nlist })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.invlists.save(path)?;
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
        Ok(Self { quantizer, invlists, nlist })
    }
}
