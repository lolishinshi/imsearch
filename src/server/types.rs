use axum::body::Bytes;
use axum_typed_multipart::TryFromMultipart;
use utoipa::ToSchema;

/// 搜索请求参数
#[derive(TryFromMultipart)]
pub struct SearchRequest {
    pub file: Bytes,
    pub orb_scale_factor: Option<f32>,
    pub nprobe: Option<usize>,
    pub max_codes: Option<usize>,
}

/// 搜索表单（用于API文档）
#[derive(Debug, ToSchema)]
#[allow(unused)]
pub struct SearchForm {
    /// 上传的图片文件
    #[schema(format = Binary, content_media_type = "application/octet-stream")]
    pub file: String,
    /// ORB特征提取缩放因子
    pub orb_scale_factor: Option<f32>,
    /// 搜索扫描的倒排列表数量
    pub nprobe: Option<usize>,
    /// 搜索扫描的最大向量数量
    pub max_codes: Option<usize>,
}
