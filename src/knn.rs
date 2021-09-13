use std::slice;
use hnsw::{Hnsw, Searcher};
use space::{Metric, Neighbor};
use rand_pcg::Pcg64;
use rayon::prelude::*;

pub struct HammingDistance;

impl Metric<[u8; 32]> for HammingDistance {
    type Unit = u8;

    fn distance(&self, a: &[u8; 32], b: &[u8; 32]) -> Self::Unit {
        let mut dist = 0;
        for i in 0..32 {
            let d = a[i] ^ b[i];
            dist += d.count_ones() as u8;
        }
        dist
    }
}

pub struct KnnSearcher {
    hnsw: Hnsw<HammingDistance, [u8; 32], Pcg64, 12, 24>,
    searcher: Searcher<u8>,
}

impl KnnSearcher {
    pub fn new() -> Self {
        let searcher = Searcher::default();
        let hnsw = Hnsw::new(HammingDistance);
        Self { hnsw, searcher }
    }

    pub fn add_iter<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = [u8; 32]>
    {
        for q in iter {
            self.hnsw.insert(q, &mut self.searcher);
        }
    }

    pub fn search_batch(&mut self, query_points: &[[u8; 32]]) -> Vec<[Neighbor<u8>; 8]>
    {
        query_points.into_par_iter()
            .map(|p| {
                let mut searcher = self.searcher.clone();
                let mut neighbors = [Neighbor {
                    index: !0,
                    distance: !0,
                }; 8];
                self.hnsw.nearest(p, 24, &mut searcher, &mut neighbors);
                neighbors
            }).collect::<Vec<_>>()
    }

    #[inline]
    pub fn feature(&self, idx: usize) -> &[u8; 32] {
        self.hnsw.feature(idx)
    }
}
