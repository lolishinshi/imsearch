use std::path::Path;

use anyhow::Result;
use bytemuck::cast_slice;
use hnsw_rs::prelude::*;

struct DistHamming<const N: usize>;

impl<const N: usize> Distance<u8> for DistHamming<N> {
    fn eval(&self, va: &[u8], vb: &[u8]) -> f32 {
        let va: &[u64] = cast_slice(va);
        let vb: &[u64] = cast_slice(vb);
        let mut sum = 0;
        for i in 0..N / 8 {
            sum += (va[i] ^ vb[i]).count_ones();
        }
        sum as f32
    }
}

pub struct HNSW {
    index: Hnsw<'static, u8, DistHamming<8>>,
}

impl Default for HNSW {
    fn default() -> Self {
        Self::new()
    }
}

impl HNSW {
    pub fn new() -> Self {
        Self { index: Hnsw::<u8, _>::new(32, 1_000_000, 16, 128, DistHamming::<8>) }
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let reloader = HnswIo::new(path.as_ref(), "phash");
        let reloader = Box::leak(Box::new(reloader));
        let index = reloader.load_hnsw_with_dist(DistHamming::<8>)?;
        Ok(Self { index })
    }

    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        self.index.file_dump(path.as_ref(), "phash")?;
        Ok(())
    }

    pub fn ntotal(&self) -> usize {
        self.index.get_nb_point()
    }

    pub fn add(&self, data: &[u8], id: usize) {
        self.index.insert((data, id));
    }

    pub fn search(&self, data: &[u8], k: usize) -> Vec<Neighbour> {
        self.index.search(data, k, 16)
    }
}
