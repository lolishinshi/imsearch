mod quantizer;

use std::path::Path;

use anyhow::Result;
pub use quantizer::*;

use crate::invlists::InvertedLists;
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
        let centroids = binary_kmeans_2level::<N>(data, self.nlist, max_iter);
        self.quantizer.add(&centroids)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.quantizer.save(path)?;
        // self.invlists.save(path)?;
        Ok(())
    }
}
