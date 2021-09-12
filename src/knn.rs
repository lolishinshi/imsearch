use hnsw::{Hnsw, Searcher};
use space::{Metric, Neighbor};
use std::mem;
use rand_pcg::Pcg64;

struct DescriptorDistance;

impl Metric<[u8; 32]> for DescriptorDistance {
    type Unit = u8;

    fn distance(&self, a: &[u8; 32], b: &[u8; 32]) -> Self::Unit {
        let (a, b) = unsafe {
            (
                mem::transmute::<&[u8; 32], &[u32; 8]>(a),
                mem::transmute::<&[u8; 32], &[u32; 8]>(b),
            )
        };
        let mut dist = 0;
        for i in 0..8 {
            let d = a[i] ^ b[i];
            dist += d.count_ones() as u8;
        }
        dist
    }
}

pub struct KnnSearcher {
    hnsw: Hnsw<DescriptorDistance, [u8; 32], Pcg64, 12, 24>,
    searcher: Searcher<[u8; 32]>,
}

impl KnnSearcher {
    pub fn new() -> Self {
        let searcher = Searcher::default();
        let hnsw = Hnsw::new(DescriptorDistance);
        Self { hnsw, searcher }
    }

    #[inline]
    pub fn add_iter<I>(&mut self, q: I)
    where
        I: Iterator<Item = [u8; 32]>
    {
        self.hnsw.insert(q, &mut self.searcher);
    }

    #[inline]
    pub fn search(&mut self, q: &[u8; 32]) -> [Neighbor<i32, i32>; 8] {
        let mut neighbors = [Neighbor {
            index: !0,
            distance: !0,
        }; 8];
        self.hnsw.nearest(q, 24, &mut self.searcher, &mut neighbors);
        neighbors
    }
}