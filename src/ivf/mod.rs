pub mod invlists;
pub mod quantizer;

use std::time::Duration;

use anyhow::{Result, anyhow};
pub use invlists::*;
pub use quantizer::*;
use rayon::prelude::*;
use tokio::time::Instant;

use crate::hamming::knn_hamming;
use crate::kmeans::binary_kmeans_2level;

pub trait FromFile {
    /// 从文件中加载
    fn from_file(path: &str) -> Result<Self>
    where
        Self: Sized;
}

pub struct SeachResult {
    pub quantizer_time: Duration,
    pub search_time: Duration,
    pub neighbors: Vec<Vec<Neighbor>>,
}

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
    pub fn new(quantizer: Q, invlists: I, nlist: usize) -> Self {
        Self { quantizer, invlists, nlist }
    }

    pub fn from_file() -> Result<()> {
        Ok(())
    }

    pub fn train(&mut self, data: &[[u8; N]], max_iter: usize) -> Result<()> {
        if self.quantizer.is_trained() {
            return Err(anyhow!("quantizer has been trained"));
        }
        let centroids = binary_kmeans_2level::<N>(data, self.nlist, max_iter);
        self.quantizer.add(&centroids)?;
        self.quantizer.save()?;
        Ok(())
    }

    pub fn add(&mut self, data: &[[u8; N]], ids: &[u64]) -> Result<()> {
        let vlists = self.quantizer.search(data, 1)?;
        let mut writer = self.invlists.writer()?;
        for ((xq, id), lists) in data.iter().zip(ids).zip(vlists) {
            let list_no = lists[0] as u32;
            let (xq, _) = xq.as_chunks();
            writer.add_entries(list_no, &[*id], &xq)?;
        }
        Ok(())
    }

    pub fn search(&'a self, data: &[[u8; N]], k: usize, nprobe: usize) -> Result<SeachResult> {
        let start = Instant::now();
        let vlists = self.quantizer.search(data, nprobe)?;
        let quantizer_time = start.elapsed();

        let neighbors = data
            .iter()
            .zip(vlists)
            .par_bridge()
            .map_init(
                || self.invlists.reader().unwrap(),
                |reader, (xq, lists)| {
                    let mut v = vec![];
                    for list_no in lists {
                        let (ids, codes) = reader.get_list(list_no as u32)?;
                        let r = knn_hamming::<N>(xq, &codes, k);
                        let n = r
                            .into_iter()
                            .map(|(i, d)| Neighbor { id: ids[i] as usize, distance: d })
                            .collect::<Vec<_>>();
                        v.extend(n);
                    }
                    v.sort_unstable_by_key(|n| n.distance);
                    v.truncate(k);
                    Ok(v)
                },
            )
            .collect::<Result<Vec<_>>>()?;
        let search_time = start.elapsed();
        Ok(SeachResult { quantizer_time, search_time, neighbors })
    }
}
