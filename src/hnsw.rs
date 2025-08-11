use std::path::{Path, PathBuf};

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
    path: PathBuf,
}

impl HNSW {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let index = Hnsw::<u8, _>::new(32, 1_000_000, 16, 128, DistHamming::<8>);
        Ok(Self { index, path: path.to_path_buf() })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let index = if path.join("phash.hnsw.graph").exists() {
            let reloader = HnswIo::new(path, "phash");
            let reloader = Box::leak(Box::new(reloader));
            reloader.load_hnsw_with_dist(DistHamming::<8>)?
        } else {
            Hnsw::<u8, _>::new(32, 1_000_000, 16, 128, DistHamming::<8>)
        };
        Ok(Self { index, path: path.to_path_buf() })
    }

    pub fn write(&self) -> Result<()> {
        self.index.file_dump(&self.path, "phash")?;
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
