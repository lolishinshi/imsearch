mod faiss;

use std::path::Path;

use anyhow::Result;
pub use faiss::FaissHNSWQuantizer as HnswQuantizer;

/// 适用于 N 位二进制向量的量化器
pub trait Quantizer<const N: usize> {
    /// 从指定路径加载量化器
    fn open<P: AsRef<Path>>(path: P) -> Result<Self>
    where
        Self: Sized;

    /// 使用指定向量初始化量化器
    fn init(x: &[[u8; N]]) -> Result<Self>
    where
        Self: Sized;

    /// 在数据集中为多组向量搜索最接近的的 k 个向量，返回最匹配的 k 个 ID 列表
    /// 返回结果大小固定为 x.len() * k，没有找到位置的填充 -1
    fn search(&self, x: &[[u8; N]], k: usize) -> Result<Vec<i64>>;

    /// 保存量化器
    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()>;

    /// 聚类中心数量
    fn nlist(&self) -> usize;

    /// 获取聚类中心
    fn centroids(&self) -> Result<&[[u8; N]]>;
}
