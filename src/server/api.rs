use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum_typed_multipart::TypedMultipart;
use log::info;
use opencv::imgcodecs;
use opencv::prelude::*;
use prometheus::TextEncoder;
use tokio::task::spawn_blocking;

use super::error::Result;
use super::state::AppState;
use super::types::*;
use crate::config::{OrbOptions, SearchOptions};
use crate::metrics::*;
use crate::orb::ORBDetector;

/// 搜索一张图片
#[utoipa::path(
    post,
    path = "/search",
    request_body(content = SearchForm, content_type = "multipart/form-data"),
    responses(
        (status = 200, body = SearchResponse),
    )
)]
pub async fn search_handler(
    State(state): State<Arc<AppState>>,
    data: TypedMultipart<SearchRequest>,
) -> Result<Json<SearchResponse>> {
    // 处理上传的文件和参数
    let orb = OrbOptions {
        orb_nfeatures: data.orb_nfeatures.unwrap_or(state.orb.orb_nfeatures),
        orb_scale_factor: data.orb_scale_factor.unwrap_or(state.orb.orb_scale_factor),
        ..state.orb
    };
    let SearchOptions { k, distance, count, .. } = state.search;
    let nprobe = data.nprobe.unwrap_or(state.search.nprobe);

    let start = Instant::now();

    info!("正在搜索上传图片");

    let orbc = orb.clone();
    let (size, des) = spawn_blocking(move || {
        let mat = Mat::from_slice(&data.file)?;
        let img = imgcodecs::imdecode(&mat, imgcodecs::IMREAD_GRAYSCALE)?;
        let size = (img.cols() as u32, img.rows() as u32);
        let mut orb = ORBDetector::create(orbc);
        let (_, des) = orb.detect_image(img)?;
        Result::Ok((size, des))
    })
    .await??;

    let result = { state.db.search(state.index.clone(), des, k, distance, count, nprobe).await? };

    inc_image_count(size, nprobe, orb.orb_scale_factor);
    inc_search_duration(size, nprobe, orb.orb_scale_factor, start.elapsed().as_secs_f32());
    inc_search_max_score(size, nprobe, orb.orb_scale_factor, result[0].0);

    Ok(Json(SearchResponse { time: start.elapsed().as_millis() as u32, result }))
}

/// 获取 Prometheus 指标
#[utoipa::path(get, path = "/metrics")]
pub async fn metrics_handler(State(_state): State<Arc<AppState>>) -> Result<String> {
    let encoder = TextEncoder::new();
    let metrics_families = prometheus::gather();
    let body = encoder.encode_to_string(&metrics_families)?;
    Ok(body)
}
