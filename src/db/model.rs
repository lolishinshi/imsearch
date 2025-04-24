/// 图片记录
pub struct ImageRecord {
    /// 图片 ID
    pub id: i64,
    /// 图片 blake3 哈希
    pub hash: Vec<u8>,
    /// 图片路径
    pub path: String,
}

/// 图片特征向量统计
pub struct VectorStatsRecord {
    /// 图片 ID
    pub id: i64,
    /// 特征向量数量
    pub vector_count: i64,
    /// 截至到当前位置的特征向量总数，用于加速计算
    pub total_vector_count: i64,
    /// 是否索引
    pub indexed: bool,
}

/// 图片特征向量记录
pub struct VectorRecord {
    /// 图片 ID
    pub id: i64,
    /// 多维向量，维数为 count * 512bit
    pub vector: Vec<u8>,
}

pub struct VectorIdxRecord {
    pub id: i64,
    pub vector: Vec<u8>,
    pub total_vector_count: i64,
}
