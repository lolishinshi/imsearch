use axum::body::Bytes;
use axum_typed_multipart::TryFromMultipart;
use serde::Deserialize;
use utoipa::ToSchema;

/// 搜索请求参数
#[derive(TryFromMultipart)]
pub struct SearchRequest {
    pub file: Vec<Bytes>,
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
}
