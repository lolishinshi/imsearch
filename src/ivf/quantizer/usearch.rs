use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind, b1x8};

use super::Quantizer;

pub struct USearchQuantizer<const N: usize> {
    /// 索引
    index: Index,
}

impl<const N: usize> USearchQuantizer<N> {
    pub fn new() -> Result<Self> {
        let options = IndexOptions {
            // 向量的二进制位数
            dimensions: N * 8,
            metric: MetricKind::Hamming,
            quantization: ScalarKind::B1,
            // 此处为 usearch 默认参数
            // faiss 默认为 32 - 40 - 16
            connectivity: 32,
            expansion_add: 40,
            expansion_search: 16,
            ..Default::default()
        };
        let index = Index::new(&options)?;
        Ok(Self { index })
    }
}

impl<const N: usize> Quantizer<N> for USearchQuantizer<N> {
    fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let s = Self::new()?;
        s.index.load(path.as_ref().to_str().unwrap())?;
        Ok(s)
    }

    /// 为量化器填充训练好的聚类中心
    fn init(x: &[[u8; N]]) -> Result<Self> {
        let s = Self::new()?;
        s.index.reserve(x.len())?;
        x.par_iter().enumerate().for_each(|(i, chunk)| {
            let v = b1x8::from_u8s(chunk);
            s.index.add(i as u64, v).unwrap();
        });
        Ok(s)
    }

    /// 搜索一组向量，返回最接近的 k 个聚类中心
    fn search(&self, x: &[[u8; N]], k: usize) -> Result<Vec<Vec<usize>>> {
        x.par_iter()
            .map(|chunk| {
                let q = b1x8::from_u8s(chunk);
                let m = self.index.search(q, k)?;
                Ok(m.keys.into_iter().map(|key| key as usize).collect())
            })
            .collect::<Result<Vec<_>>>()
    }

    /// 保存量化器
    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.index.save(path.as_ref().to_str().unwrap())?;
        Ok(())
    }

    /// 获取量化器聚类中心数量
    fn nlist(&self) -> usize {
        self.index.size()
    }
}
