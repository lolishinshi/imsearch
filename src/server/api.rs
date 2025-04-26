use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum_typed_multipart::TypedMultipart;
use log::info;
use opencv::imgcodecs;
use opencv::prelude::*;
use serde_json::{Value, json};
use tokio::task::block_in_place;
use utoipa;

use super::error::AppError;
use super::state::AppState;
use super::types::SearchRequest;
use crate::index::FaissSearchParams;
use crate::{Slam3ORB, utils};

/// 搜索一张图片
#[utoipa::path(
    post,
    path = "/search",
    request_body(content = super::types::SearchForm, content_type = "multipart/form-data")
)]
pub async fn search_handler(
    State(state): State<Arc<AppState>>,
    data: TypedMultipart<SearchRequest>,
) -> Result<Json<Value>, AppError> {
    // 处理上传的文件和参数
    let mut orb = state.orb.clone();
    let mut params =
        FaissSearchParams { nprobe: state.search.nprobe, max_codes: state.search.max_codes };
    if let Some(orb_scale_factor) = data.orb_scale_factor {
        orb.orb_scale_factor = orb_scale_factor;
    }
    params.nprobe = data.nprobe.unwrap_or(1);
    params.max_codes = data.max_codes.unwrap_or_default();

    let start = Instant::now();

    info!("正在搜索上传图片");

    let mut orb = Slam3ORB::from(&orb);

    let mat = Mat::from_slice(&data.file)?;
    let (_, des) = block_in_place(|| {
        let img = imgcodecs::imdecode(&mat, imgcodecs::IMREAD_GRAYSCALE)?;
        utils::detect_and_compute(&mut orb, &img)
    })?;

    let index = state.index.read().await;
    let mut result = state
        .db
        .search(&index, des, state.search.k, state.search.distance, state.search.count, params)
        .await?;
    result.truncate(state.search.count);

    Ok(Json(json!({
        "time": start.elapsed().as_millis(),
        "result": result,
    })))
}
