use axum::body::Bytes;
use axum_typed_multipart::TryFromMultipart;
use serde::{Deserialize, Serialize};
use utoipa::openapi::security::{HttpAuthScheme, HttpBuilder, SecurityScheme};
use utoipa::openapi::{Components, OpenApi};
use utoipa::{Modify, ToSchema};

pub struct SecurityAddon;

impl Modify for SecurityAddon {
    fn modify(&self, openapi: &mut OpenApi) {
        if openapi.components.is_none() {
            openapi.components = Some(Components::new());
        }
        openapi.components.as_mut().unwrap().add_security_scheme(
            "bearerAuth",
            SecurityScheme::Http(HttpBuilder::new().scheme(HttpAuthScheme::Bearer).build()),
        );
    }
}

/// 搜索请求参数
#[derive(TryFromMultipart)]
pub struct SearchRequest {
    pub file: Bytes,
    pub orb_nfeatures: Option<u32>,
    pub orb_scale_factor: Option<f32>,
    pub nprobe: Option<usize>,
}

/// 搜索表单（用于API文档）
#[derive(Debug, ToSchema)]
#[allow(unused)]
pub struct SearchForm {
    /// 上传的图片文件
    #[schema(format = Binary, content_media_type = "application/octet-stream")]
    pub file: String,
    /// ORB特征提取数量
    pub orb_nfeatures: Option<u32>,
    /// ORB特征提取缩放因子
    pub orb_scale_factor: Option<f32>,
    /// 搜索扫描的倒排列表数量
    pub nprobe: Option<usize>,
    /// HNSW 搜索时每次访问的节点数量
    pub ef_search: Option<usize>,
}

/// 搜索响应
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SearchResponse {
    /// 搜索耗时，单位为毫秒
    pub time: u32,
    /// 图片的搜索结果，格式为 `(相似度, 图片路径)`
    pub result: Vec<(f32, String)>,
}
