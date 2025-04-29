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
use crate::config::OrbOptions;
use crate::faiss::{FaissSearchParams, get_faiss_stats, reset_faiss_stats};
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
    let orb = OrbOptions {
        orb_nfeatures: data.orb_nfeatures.unwrap_or(state.orb.orb_nfeatures),
        orb_scale_factor: data.orb_scale_factor.unwrap_or(state.orb.orb_scale_factor),
        ..state.orb
    };
    let params = FaissSearchParams {
        nprobe: data.nprobe.unwrap_or(state.search.nprobe),
        max_codes: data.max_codes.unwrap_or(state.search.max_codes),
    };

    let start = Instant::now();

    info!("正在搜索上传图片");

    let des = block_in_place(|| {
        data.file
            .par_iter()
            .map(|file| {
                let mut orb = Slam3ORB::from(&orb);
                let mat = Mat::from_slice(file)?;
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
) -> Result<()> {
    let mut lock = state.index.write().await;
    // NOTE: 此处先释放旧索引，再重新加载新索引
    *lock = state.db.get_index_template();
    state.db.set_mmap(!data.no_mmap);
    let mut index = state.db.get_index();
    if data.hnsw {
        index.to_hnsw();
    }
    *lock = index;
    // 更新缓存 ID
    state.db.load_total_vector_count().await?;
    Ok(())
}

/// 添加图片到数据库
#[utoipa::path(
    post,
    path = "/add",
    request_body(content = AddImageForm, content_type = "multipart/form-data")
)]
pub async fn add_image_handler(
    State(state): State<Arc<AppState>>,
    data: TypedMultipart<AddImageRequest>,
) -> Result<Json<Value>> {
    for file in &data.file {
        let file_name = match &file.metadata.file_name {
            Some(file_name) => file_name,
            None => {
                return Err(anyhow::anyhow!("文件名不能为空").into());
            }
        };

        let hash = blake3::hash(&file.contents);
        if state.db.check_hash(hash.as_bytes()).await? {
            continue;
        }
        let des = block_in_place(|| -> Result<_> {
            let mut orb = Slam3ORB::from(&state.orb);
            let img = utils::imdecode(&file.contents, state.orb.img_max_width)?;
            let (_, des) = utils::detect_and_compute(&mut orb, &img)?;
            Ok(des)
        })?;
        if des.rows() <= 10 {
            continue;
        }
        state.db.add_image(file_name, hash.as_bytes(), des).await?;
    }
    Ok(Json(json!({})))
}

/// 构建索引
#[utoipa::path(
    post,
    path = "/build",
    request_body = BuildRequest
)]
pub async fn build_handler(
    State(state): State<Arc<AppState>>,
    data: Json<BuildRequest>,
) -> Result<()> {
    state.db.set_ondisk(data.on_disk);
    state.db.build_index(data.batch_size, data.no_split).await?;
    Ok(())
}

/// 获取搜索统计信息
#[utoipa::path(
    post,
    path = "/stats",
    responses(
        (status = 200, body = StatsResponse),
    )
)]
pub async fn stats_handler() -> Result<Json<StatsResponse>> {
    let stats = get_faiss_stats();
    Ok(Json(StatsResponse {
        ndis: stats.ndis,
        nprobe: stats.nlist,
        nheap_updates: stats.nheap_updates,
        quantization_time: stats.quantization_time,
        search_time: stats.search_time,
    }))
}

/// 重置搜索统计信息
#[utoipa::path(post, path = "/reset_stats")]
pub async fn reset_stats_handler() -> Result<()> {
    reset_faiss_stats();
    Ok(())
}
