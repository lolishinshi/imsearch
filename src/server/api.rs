use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum_typed_multipart::TypedMultipart;
use log::info;
use opencv::imgcodecs;
use opencv::prelude::*;
use rayon::prelude::*;
use serde_json::{Value, json};
use tokio::task::block_in_place;

use super::error::Result;
use super::state::AppState;
use super::types::*;
use crate::faiss::FaissSearchParams;
use crate::{Slam3ORB, utils};

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
) -> Result<Json<Value>> {
    // 处理上传的文件和参数
    let mut orb = state.orb.clone();
    let mut params =
        FaissSearchParams { nprobe: state.search.nprobe, max_codes: state.search.max_codes };
    if let Some(orb_scale_factor) = data.orb_scale_factor {
        orb.orb_scale_factor = orb_scale_factor;
    }
    params.nprobe = data.nprobe.unwrap_or(state.search.nprobe);
    params.max_codes = data.max_codes.unwrap_or(state.search.max_codes);

    let start = Instant::now();

    info!("正在搜索上传图片");

    let des = block_in_place(|| {
        data.file
            .par_iter()
            .map(|file| {
                let mut orb = Slam3ORB::from(&orb);
                let mat = Mat::from_slice(&file)?;
                let img = imgcodecs::imdecode(&mat, imgcodecs::IMREAD_GRAYSCALE)?;
                let (_, des) = utils::detect_and_compute(&mut orb, &img)?;
                Ok(des)
            })
            .collect::<Result<Vec<_>>>()
    })?;

    let index = state.index.read().await;
    let result = state
        .db
        .search(&index, &des, state.search.k, state.search.distance, state.search.count, params)
        .await?;

    Ok(Json(json!({
        "time": start.elapsed().as_millis(),
        "result": result,
    })))
}

/// 使用指定参数重新加载索引
#[utoipa::path(
    post,
    path = "/reload",
    request_body = ReloadRequest
)]
pub async fn reload_handler(
    State(state): State<Arc<AppState>>,
    data: Json<ReloadRequest>,
) -> Result<Json<Value>> {
    let mut lock = state.index.write().await;
    // NOTE: 此处先释放旧索引，再重新加载新索引
    *lock = state.db.get_index_template();
    state.db.set_mmap(!data.no_mmap);
    let index = state.db.get_index();
    *lock = index;
    Ok(Json(json!({})))
}
