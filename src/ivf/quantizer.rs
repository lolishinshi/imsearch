use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind, b1x8};

/// 适用于 N 位二进制向量的量化器
pub trait Quantizer<const N: usize> {
    /// 批量添加向量
    ///
    /// x 为展平的 n * N 的二维数组
    ///
    fn add(&mut self, x: &[u8]) -> Result<()>;
    /// 在数据集中搜索最接近的的 k 个向量，返回最匹配的 k 个 ID 列表
    ///
    /// x 为展平的 n * N 的一维数组
    fn search(&self, x: &[u8], k: usize) -> Result<Vec<Vec<usize>>>;

    /// 保存量化器
    fn save(&self) -> Result<()>;

    /// 是否已经训练
    fn is_trained(&self) -> bool;
}

pub struct USearchQuantizer<const N: usize> {
    index: Index,
    path: String,
}

impl<const N: usize> USearchQuantizer<N> {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let options = IndexOptions {
            dimensions: N * 8,
            metric: MetricKind::Hamming,
            quantization: ScalarKind::B1,
            connectivity: 32,
            expansion_add: 128,
            expansion_search: 64,
            ..Default::default()
        };
        let index = Index::new(&options)?;
        let path = path.as_ref();
        let path_str = path.to_str().unwrap().to_string();
        if path.exists() {
            index.load(&path_str)?;
        }
        Ok(Self { index, path: path_str })
    }
}

impl<const N: usize> Quantizer<N> for USearchQuantizer<N> {
    fn add(&mut self, x: &[u8]) -> Result<()> {
        assert_eq!(self.index.size(), 0, "quantizer has been trained");
        self.index.reserve(x.len() / N)?;
        for (i, chunk) in x.chunks_exact(N).enumerate() {
            let v = b1x8::from_u8s(chunk);
            self.index.add(i as u64, v)?;
        }
        Ok(())
    }

    fn search(&self, x: &[u8], k: usize) -> Result<Vec<Vec<usize>>> {
        x.chunks_exact(N)
            .par_bridge()
            .map(|chunk| {
                let q = b1x8::from_u8s(chunk);
                let m = self.index.search(q, k)?;
                Ok(m.keys.iter().map(|&key| key as usize).collect())
            })
            .collect::<Result<Vec<_>>>()
    }

    fn save(&self) -> Result<()> {
        self.index.save(&self.path)?;
        Ok(())
    }

    fn is_trained(&self) -> bool {
        self.index.size() > 0
    }
}
