use std::path::Path;

use anyhow::Result;
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

    /// 将量化器保存到文件
    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()>;
}

pub struct USearchQuantizer<const N: usize> {
    index: Index,
}

impl<const N: usize> USearchQuantizer<N> {
    pub fn new() -> Result<Self> {
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
        Ok(Self { index })
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
            .map(|chunk| {
                let q = b1x8::from_u8s(chunk);
                let m = self.index.search(q, k)?;
                Ok(m.keys.iter().map(|&key| key as usize).collect())
            })
            .collect::<Result<Vec<_>>>()
    }

    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref().to_str().unwrap();
        self.index.save(path)?;
        Ok(())
    }
}
