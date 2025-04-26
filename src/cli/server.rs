use std::sync::Arc;
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use axum_typed_multipart::{TryFromMultipart, TypedMultipart};
use clap::Parser;
use log::info;
use opencv::imgcodecs;
use opencv::prelude::*;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tokio::task::block_in_place;
use tower_http::limit::RequestBodyLimitLayer;
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

use crate::cli::SubCommandExtend;
use crate::config::{OrbOptions, SearchOptions};
use crate::index::{FaissIndex, FaissSearchParams};
use crate::{IMDB, IMDBBuilder, Opts, Slam3ORB, utils};

#[derive(Parser, Debug, Clone)]
pub struct ServerCommand {
    #[command(flatten)]
    pub orb: OrbOptions,
    #[command(flatten)]
    pub search: SearchOptions,
    /// 监听地址
    #[arg(long, default_value = "127.0.0.1:8000")]
    pub addr: String,
}

#[derive(OpenApi)]
#[openapi(paths(search_handler,), components(schemas(SearchForm,),))]
pub struct ApiDoc;

// 定义共享状态
struct AppState {
    index: RwLock<FaissIndex>,
    db: IMDB,
    opts: ServerCommand,
}

impl SubCommandExtend for ServerCommand {
    async fn run(&self, opts: &Opts) -> anyhow::Result<()> {
        let db = IMDBBuilder::new(opts.conf_dir.clone())
            .mmap(!self.search.no_mmap)
            .cache(true)
            .open()
            .await?;

        let index = db.get_index();

        // 创建共享状态
        let state = Arc::new(AppState { index: RwLock::new(index), db, opts: self.clone() });

        // 创建路由
        let app = Router::new()
            .route("/search", post(search_handler))
            .merge(SwaggerUi::new("/docs").url("/api-docs/openapi.json", ApiDoc::openapi()))
            .layer(DefaultBodyLimit::disable())
            // 上传限制：10M
            .layer(RequestBodyLimitLayer::new(1024 * 1024 * 10))
            .with_state(state);

        // 启动服务器
        info!("starting server at http://{}", &self.addr);
        let listener = TcpListener::bind(&self.addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}

#[derive(TryFromMultipart)]
struct SearchRequest {
    file: Bytes,
    orb_scale_factor: Option<f32>,
    nprobe: Option<usize>,
    max_codes: Option<usize>,
}

#[derive(Debug, ToSchema)]
#[allow(unused)]
struct SearchForm {
    #[schema(format = Binary, content_media_type = "application/octet-stream")]
    file: String,
    orb_scale_factor: Option<f32>,
    nprobe: Option<usize>,
    max_codes: Option<usize>,
}

/// 搜索一张图片
#[utoipa::path(
    post,
    path = "/search",
    request_body(content = SearchForm, content_type = "multipart/form-data")
)]
async fn search_handler(
    State(state): State<Arc<AppState>>,
    data: TypedMultipart<SearchRequest>,
) -> Result<Json<Value>, AppError> {
    // 处理上传的文件和参数
    let mut opts = state.opts.clone();
    let mut params =
        FaissSearchParams { nprobe: opts.search.nprobe, max_codes: opts.search.max_codes };
    if let Some(orb_scale_factor) = data.orb_scale_factor {
        opts.orb.orb_scale_factor = orb_scale_factor;
    }
    params.nprobe = data.nprobe.unwrap_or(1);
    params.max_codes = data.max_codes.unwrap_or_default();

    let start = Instant::now();

    info!("正在搜索上传图片");

    let mut orb = Slam3ORB::from(&opts.orb);

    let mat = Mat::from_slice(&data.file)?;
    let (_, des) = block_in_place(|| {
        let img = imgcodecs::imdecode(&mat, imgcodecs::IMREAD_GRAYSCALE)?;
        utils::detect_and_compute(&mut orb, &img)
    })?;

    let index = state.index.read().await;
    let mut result = state
        .db
        .search(&index, des, opts.search.k, opts.search.distance, opts.search.count, params)
        .await?;
    result.truncate(opts.search.count);

    Ok(Json(json!({
        "time": start.elapsed().as_millis(),
        "result": result,
    })))
}

struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Something went wrong: {}", self.0))
            .into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}
