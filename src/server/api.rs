use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum_auth::AuthBearer;
use axum_typed_multipart::TypedMultipart;
use log::info;
use opencv::imgcodecs;
use opencv::prelude::*;
use prometheus::TextEncoder;
use rayon::prelude::*;
use serde_json::{Value, json};
use tokio::task::spawn_blocking;

use super::error::Result;
use super::state::AppState;
use super::types::*;
use crate::config::OrbOptions;
use crate::faiss::{FaissSearchParams, get_faiss_stats, reset_faiss_stats};
use crate::imdb::BuildOptions;
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

    let mut sizes = vec![];
    let mut deses = vec![];
    let orbc = orb.clone();
    let r = spawn_blocking(move || {
        data.file
            .par_iter()
            .map(|file| {
                let mat = Mat::from_slice(file)?;
                let img = imgcodecs::imdecode(&mat, imgcodecs::IMREAD_GRAYSCALE)?;
                let size = (img.cols() as u32, img.rows() as u32);
                let mut orb = ORBDetector::create(orbc.clone());
                let (_, des) = orb.detect_image(img)?;
                Ok((size, des))
            })
            .collect::<Result<Vec<_>>>()
    })
    .await??;
    for (size, des) in r {
        sizes.push(size);
        deses.push(des);
    }

    let lock = state.index.write().await;
    let index = lock.clone().unwrap();
    let result = state
        .db
        .search(
            index,
            &deses,
            state.search.k,
            state.search.distance,
            state.search.count,
            params.clone(),
        )
        .await?;

    for (v, size) in result.iter().zip(sizes.iter()) {
        if !v.is_empty() {
            inc_image_count(*size, params.nprobe, orb.orb_scale_factor);
            inc_search_duration(
                *size,
                params.nprobe,
                orb.orb_scale_factor,
                start.elapsed().as_secs_f32() / deses.len() as f32,
            );
            inc_search_max_score(*size, params.nprobe, orb.orb_scale_factor, v[0].0);
        }
    }

    Ok(Json(json!({
        "time": start.elapsed().as_millis(),
        "result": result,
    })))
}

/// 使用指定参数重新加载索引
#[utoipa::path(
    post,
    path = "/reload",
    request_body = ReloadRequest,
    security(("bearerAuth" = []))
)]
pub async fn reload_handler(
    State(state): State<Arc<AppState>>,
    AuthBearer(token): AuthBearer,
    data: Json<ReloadRequest>,
) -> Result<()> {
    if token != state.token {
        return Err(anyhow::anyhow!("鉴权失败").into());
    }
    let mut lock = state.index.write().await;
    // NOTE: 此处先释放旧索引，再重新加载新索引
    drop(lock.take().unwrap());
    let mut index = state.db.get_index(!data.no_mmap);
    if data.hnsw {
        index.to_hnsw();
    }
    lock.replace(Arc::new(index));
    // 更新缓存 ID
    state.db.load_total_vector_count().await?;
    Ok(())
}

/// 添加图片到数据库
#[utoipa::path(
    post,
    path = "/add",
    request_body(content = AddImageForm, content_type = "multipart/form-data"),
    security(("bearerAuth" = []))
)]
pub async fn add_image_handler(
    State(state): State<Arc<AppState>>,
    AuthBearer(token): AuthBearer,
    data: TypedMultipart<AddImageRequest>,
) -> Result<Json<Value>> {
    if token != state.token {
        return Err(anyhow::anyhow!("鉴权失败").into());
    }
    let hash = data.hash.unwrap_or_default();

    for file in &data.file {
        let file_name = match &file.metadata.file_name {
            Some(file_name) => file_name,
            None => {
                return Err(anyhow::anyhow!("文件名不能为空").into());
            }
        };

        let (img, hash) = hash.hash_bytes(&file.contents)?;
        if state.db.check_hash(&hash).await? {
            continue;
        }

        let orb = state.orb.clone();
        let file = file.contents.clone();
        let des = spawn_blocking(move || -> Result<_> {
            let mut orb = ORBDetector::create(orb);
            let (_, des) = match img {
                Some(img) => orb.detect_image(img)?,
                None => orb.detect_bytes(&file)?,
            };
            Ok(des)
        })
        .await??;

        if des.rows() <= 10 {
            continue;
        }
        state.db.add_image(file_name, &hash, des).await?;
    }
    Ok(Json(json!({})))
}

/// 构建索引
#[utoipa::path(
    post,
    path = "/build",
    request_body = BuildRequest,
    security(("bearerAuth" = []))
)]
pub async fn build_handler(
    State(state): State<Arc<AppState>>,
    AuthBearer(token): AuthBearer,
    data: Json<BuildRequest>,
) -> Result<()> {
    if token != state.token {
        return Err(anyhow::anyhow!("鉴权失败").into());
    }
    let options = BuildOptions {
        on_disk: data.on_disk,
        batch_size: data.batch_size,
        no_merge: data.no_merge,
    };
    state.db.build_index(options).await?;
    Ok(())
}

/// 获取搜索统计信息
#[utoipa::path(
    get,
    path = "/stats",
    responses(
        (status = 200, body = StatsResponse),
    ),
    security(("bearerAuth" = []))
)]
pub async fn stats_handler(
    State(state): State<Arc<AppState>>,
    AuthBearer(token): AuthBearer,
) -> Result<Json<StatsResponse>> {
    if token != state.token {
        return Err(anyhow::anyhow!("鉴权失败").into());
    }
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
#[utoipa::path(post, path = "/reset_stats", security(("bearerAuth" = [])))]
pub async fn reset_stats_handler(
    State(state): State<Arc<AppState>>,
    AuthBearer(token): AuthBearer,
) -> Result<()> {
    if token != state.token {
        return Err(anyhow::anyhow!("鉴权失败").into());
    }
    reset_faiss_stats();
    Ok(())
}

/// 获取 Prometheus 指标
#[utoipa::path(get, path = "/metrics")]
pub async fn metrics_handler(State(_state): State<Arc<AppState>>) -> Result<String> {
    let encoder = TextEncoder::new();
    let metrics_families = prometheus::gather();
    let body = encoder.encode_to_string(&metrics_families)?;
    Ok(body)
}
