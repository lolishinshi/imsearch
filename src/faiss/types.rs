/// Faiss 搜索参数

#[derive(Debug, Clone)]
pub struct FaissSearchParams {
    /// 需要搜索的倒排列表数量，默认为 1
    pub nprobe: usize,
    /// 搜索时最多检查多少个向量，默认为 0，表示不限制
    pub max_codes: usize,
}

impl Default for FaissSearchParams {
    fn default() -> Self {
        Self { nprobe: 1, max_codes: 0 }
    }
}

/// Faiss 搜索结果
#[derive(Debug, Clone)]
pub struct Neighbor {
    /// 向量在索引中的 ID
    pub index: i64,
    /// 向量与查询向量的距离
    pub distance: i32,
}
