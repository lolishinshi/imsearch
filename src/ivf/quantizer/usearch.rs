use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind, b1x8};

use super::Quantizer;

pub struct USearchQuantizer<const N: usize> {
    /// 索引
    index: Index,
    /// 索引保存路径
    path: String,
}

impl<const N: usize> USearchQuantizer<N> {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let options = IndexOptions {
            // 向量的二进制位数
            dimensions: N * 8,
            metric: MetricKind::Hamming,
            quantization: ScalarKind::B1,
            // 此处为 usearch 默认参数
            // faiss 默认为 32 - 40 - 16
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
    /// 为量化器填充训练好的聚类中心
    fn add(&mut self, x: &[[u8; N]]) -> Result<()> {
        assert_eq!(self.index.size(), 0, "quantizer has been trained");
        // NOTE: 注意这里因为假设了初始大小为 0，所以只需要预留新增空间
        self.index.reserve(x.len())?;
        x.par_iter().enumerate().for_each(|(i, chunk)| {
            let v = b1x8::from_u8s(chunk);
            self.index.add(i as u64, v).unwrap();
        });
        Ok(())
    }

    /// 搜索一组向量，返回最接近的 k 个聚类中心
    fn search(&self, x: &[[u8; N]], k: usize) -> Result<Vec<Vec<usize>>> {
        x.par_iter()
            .map(|chunk| {
                let q = b1x8::from_u8s(chunk);
                let m = self.index.search(q, k)?;
                Ok(m.keys.iter().map(|&key| key as usize).collect())
            })
            .collect::<Result<Vec<_>>>()
    }

    /// 保存量化器
    fn save(&self) -> Result<()> {
        self.index.save(&self.path)?;
        Ok(())
    }

    /// 判断量化器是否已经训练
    fn is_trained(&self) -> bool {
        self.index.size() > 0
    }

    /// 获取量化器聚类中心数量
    fn nlist(&self) -> usize {
        self.index.size()
    }
}
