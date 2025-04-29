use axum::body::Bytes;
use axum_typed_multipart::{FieldData, TryFromMultipart};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// 搜索请求参数
#[derive(TryFromMultipart)]
pub struct SearchRequest {
    pub file: Vec<Bytes>,
    pub orb_nfeatures: Option<u32>,
    pub orb_scale_factor: Option<f32>,
    pub nprobe: Option<usize>,
    pub max_codes: Option<usize>,
}

/// 搜索表单（用于API文档）
#[derive(Debug, ToSchema)]
#[allow(unused)]
pub struct SearchForm {
    /// 上传的图片文件，可以是多张图片
    #[schema(format = Binary, content_media_type = "application/octet-stream")]
    pub file: String,
    /// ORB特征提取数量
    pub orb_nfeatures: Option<u32>,
    /// ORB特征提取缩放因子
    pub orb_scale_factor: Option<f32>,
    /// 搜索扫描的倒排列表数量
    pub nprobe: Option<usize>,
    /// 搜索扫描的最大向量数量
    pub max_codes: Option<usize>,
}

/// 搜索响应
#[derive(Debug, ToSchema)]
pub struct SearchResponse {
    /// 搜索耗时，单位为毫秒
    pub time: u32,
    /// 每张图片的搜索结果，格式为 `(相似度, 图片路径)`
    pub result: Vec<Vec<(f32, String)>>,
}

/// 重新加载索引的参数
#[derive(Debug, Deserialize, ToSchema)]
pub struct ReloadRequest {
    /// 是否不使用 mmap 模式
    #[schema(default = false)]
    #[serde(default)]
    pub no_mmap: bool,
    /// 是否转换为 hnsw 索引加载
    #[schema(default = false)]
    #[serde(default)]
    pub hnsw: bool,
}

#[derive(Debug, TryFromMultipart)]
pub struct AddImageRequest {
    pub file: Vec<FieldData<Bytes>>,
    pub min_keypoints: Option<u32>,
}

#[derive(Debug, ToSchema)]
#[allow(unused)]
pub struct AddImageForm {
    /// 上传的图片文件，可以是多张图片。主要需要包含文件名，否则无法插入数据库。
    #[schema(format = Binary, content_media_type = "application/octet-stream")]
    pub file: String,
    /// 最少特征点，低于该值的图片会被过滤
    #[schema(default = 250)]
    pub min_keypoints: Option<u32>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct BuildRequest {
    /// 是否使用 OnDiskInvertedLists 格式
    #[schema(default = false)]
    #[serde(default)]
    pub on_disk: bool,
    /// 构建索引时，多少张图片为一个批次
    #[schema(default = 100000)]
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// 构建索引时，不分割文件，而是直接在内存中构建，每次直接保存全量索引
    #[schema(default = false)]
    #[serde(default)]
    pub no_split: bool,
}

/// 索引统计信息
#[derive(Debug, Serialize, ToSchema)]
pub struct StatsResponse {
    /// 扫描的距离总数
    pub ndis: usize,
    /// 扫描的倒排列表数量
    pub nprobe: usize,
    /// 堆更新次数
    pub nheap_updates: usize,
    /// 量化时间，单位为毫秒
    pub quantization_time: f64,
    /// 搜索时间，单位为毫秒
    pub search_time: f64,
}

fn default_batch_size() -> usize {
    100000
}
