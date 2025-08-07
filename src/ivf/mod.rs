pub mod invlists;
pub mod quantizer;

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::Result;
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
    pub neighbors: Vec<Vec<Neighbor>>,
}

#[derive(Debug)]
pub struct Neighbor {
    pub id: usize,
    pub distance: u32,
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

        let neighbors = data
            .into_iter()
            .zip(vlists.chunks_exact(nprobe))
            .par_bridge()
            .map(|(xq, lists)| {
                let mut v = Vec::with_capacity(k * lists.len());
                for &list_no in lists {
                    if list_no == -1 {
                        continue;
                    }
                    let t = Instant::now();
                    // NOTE: 此处统计的 IO 时间并不准确，因为 mmap 的实际 IO 发生在访问时
                    let (ids, codes) = self.invlists.get_list(list_no as usize).unwrap();
                    io_time.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    let r = knn_hamming::<N>(xq, &codes, k);
                    let n =
                        r.into_iter().map(|(i, d)| Neighbor { id: ids[i] as usize, distance: d });
                    v.extend(n);
                    calc_time.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
                v.select_nth_unstable_by_key(k, |n| n.distance);
                v.truncate(k);
                v
            })
            .collect::<Vec<_>>();
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
