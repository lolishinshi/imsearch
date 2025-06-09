use std::path::Path;

use anyhow::Result;
use hnsw_rs::prelude::*;
use rayon::prelude::*;

use crate::hamming::hamming;
use crate::ivf::Quantizer;

struct DistHamming<const N: usize>;

impl<const N: usize> Distance<u8> for DistHamming<N> {
    fn eval(&self, va: &[u8], vb: &[u8]) -> f32 {
        hamming::<N>(va, vb) as f32
    }
}

pub struct HnswQuantizer<const N: usize> {
    hnsw: Hnsw<'static, u8, DistHamming<N>>,
}

impl<const N: usize> Quantizer<N> for HnswQuantizer<N> {
    fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let reloader = HnswIo::new(path.as_ref(), "quantizer");
        let reloader = Box::leak(Box::new(reloader));
        // NOTE: reloader 加载的 HNSW 生命周期依赖于 reloader 的引用，所以需要使用 Box::leak 来延长生命周期
        let hnsw = reloader.load_hnsw_with_dist(DistHamming::<N>)?;
        Ok(Self { hnsw })
    }

    fn init(x: &[[u8; N]]) -> Result<Self> {
        let nlist = x.len();
        //let nb_layer = 16.min((nlist as f32).ln().trunc() as usize);
        let hnsw = Hnsw::<u8, _>::new(32, nlist, 16, 128, DistHamming::<N>);
        x.par_iter().enumerate().for_each(|(i, chunk)| {
            hnsw.insert((chunk, i));
        });
        Ok(Self { hnsw })
    }

    fn search(&self, x: &[[u8; N]], k: usize) -> Result<Vec<Vec<usize>>> {
        let v = x
            .par_iter()
            .map(|chunk| self.hnsw.search(chunk, k, 16).iter().map(|n| n.d_id).collect::<Vec<_>>())
            .collect();
        Ok(v)
    }

    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.hnsw.file_dump(path.as_ref(), "quantizer")?;
        Ok(())
    }

    fn nlist(&self) -> usize {
        self.hnsw.get_nb_point()
    }
}
