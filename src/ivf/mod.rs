mod quantizer;

use anyhow::{Result, anyhow};
pub use quantizer::*;
use rayon::prelude::*;

use crate::hamming::knn_hamming;
use crate::invlists::{InvertedLists, InvertedListsReader, InvertedListsWriter};
use crate::kmeans::binary_kmeans_2level;

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

    pub fn search(
        &self,
        data: &[u8],
        k: usize,
        nprobe: usize,
    ) -> Result<(Vec<Vec<usize>>, Vec<Vec<u32>>)> {
        let mut vids = vec![];
        let mut vdis = vec![];
        let vlists = self.quantizer.search(data, nprobe)?;
        let reader = self.invlists.reader()?;
        for (xq, lists) in data.chunks_exact(N).zip(vlists) {
            for list_no in lists {
                let (ids, codes) = reader.get_list(list_no as u32)?;
                let (idx, dis) = knn_hamming::<N>(xq, &codes, k);
                let ids = idx.into_iter().map(|i| ids[i] as usize).collect::<Vec<_>>();
                vids.push(ids);
                vdis.push(dis);
            }
        }
        Ok((vids, vdis))
    }
}
