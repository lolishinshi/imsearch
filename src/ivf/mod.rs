mod quantizer;

use std::time::Duration;

use anyhow::{Result, anyhow};
pub use quantizer::*;
use rayon::prelude::*;
use tokio::time::Instant;

use crate::hamming::knn_hamming;
use crate::invlists::{InvertedLists, InvertedListsReader, InvertedListsWriter};
use crate::kmeans::binary_kmeans_2level;

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

impl<const N: usize, Q: Quantizer<N>, I: InvertedLists<N>> IvfHnsw<N, Q, I> {
    pub fn new(quantizer: Q, invlists: I, nlist: usize) -> Self {
        Self { quantizer, invlists, nlist }
    }

    pub fn train(&mut self, data: &[u8], max_iter: usize) -> Result<()> {
        if self.quantizer.is_trained() {
            return Err(anyhow!("quantizer has been trained"));
        }
        let centroids = binary_kmeans_2level::<N>(data, self.nlist, max_iter);
        self.quantizer.add(&centroids)?;
        self.quantizer.save()?;
        Ok(())
    }

    pub fn add(&mut self, data: &[u8], ids: &[u64]) -> Result<()> {
        let vlists = self.quantizer.search(data, 1)?;
        let mut writer = self.invlists.writer()?;
        for ((xq, id), lists) in data.chunks_exact(N).zip(ids).zip(vlists) {
            let list_no = lists[0] as u32;
            writer.add_entries(list_no, &[*id], xq)?;
        }
        Ok(())
    }

    pub fn search(&self, data: &[u8], k: usize, nprobe: usize) -> Result<SeachResult> {
        let start = Instant::now();
        let mut neighbors = vec![];

        let vlists = self.quantizer.search(data, nprobe)?;
        let quantizer_time = start.elapsed();

        let reader = self.invlists.reader()?;
        for (xq, lists) in data.chunks_exact(N).zip(vlists) {
            let mut v = vec![];
            for list_no in lists {
                let (ids, codes) = reader.get_list(list_no as u32)?;
                let (idx, dis) = knn_hamming::<N>(xq, &codes, k);
                let n = idx
                    .into_iter()
                    .zip(dis)
                    .map(|(i, d)| Neighbor { id: ids[i] as usize, distance: d })
                    .collect::<Vec<_>>();
                v.extend(n);
            }
            v.sort_unstable_by_key(|n| n.distance);
            v.truncate(k);
            neighbors.push(v);
        }
        let search_time = start.elapsed();
        Ok(SeachResult { quantizer_time, search_time, neighbors })
    }
}
