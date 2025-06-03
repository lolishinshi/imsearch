mod usearch;

use anyhow::Result;
pub use usearch::USearchQuantizer;

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

    /// 聚类中心数量
    fn nlist(&self) -> usize;
}
