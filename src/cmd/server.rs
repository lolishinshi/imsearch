use crate::cmd::SubCommandExtend;
use crate::index::FaissSearchParams;
use crate::utils;
use crate::{IMDB, Opts, Slam3ORB, index::FaissIndex};
use anyhow::Context;
use axum::extract::DefaultBodyLimit;
use axum::{
    Json, Router,
    body::Body,
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use clap::Parser;
use log::info;
use opencv::imgcodecs;
use opencv::prelude::*;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tokio::task::block_in_place;
use tower_http::limit::RequestBodyLimitLayer;

#[derive(Parser, Debug, Clone)]
pub struct StartServer {
    /// Listen address
    #[arg(long, default_value = "127.0.0.1:8000")]
    pub addr: String,
}

// 定义共享状态
struct AppState {
    index: RwLock<FaissIndex>,
    db: IMDB,
    opts: Opts,
}

impl SubCommandExtend for StartServer {
    async fn run(&self, opts: &Opts) -> anyhow::Result<()> {
        let db = IMDB::new(opts.conf_dir.clone()).await?;

        let index = db.get_index(opts.mmap);

        // 创建共享状态
        let state = Arc::new(AppState { index: RwLock::new(index), db, opts: opts.clone() });

        // 创建路由
        let app = Router::new()
            .route("/", get(index_handler))
            .route("/search", post(search_handler))
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

// 首页处理程序，显示API使用说明
async fn index_handler() -> Response<Body> {
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/html")
        .body(Body::from(
            r#"
        <h1>Image Search API</h1>
        <p>
        示例用法：
        <br>
        <code>http --form http://127.0.0.1:8000/search file@test.jpg orb_scale_factor=1.2 nprobe=3 max_codes=0</code>
        <br>
        </p>
        "#,
        ))
        .unwrap()
}

// 处理搜索请求
#[axum::debug_handler]
async fn search_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<Value>, AppError> {
    // 处理上传的文件和参数
    let mut upload_file = None;
    let mut opts = state.opts.clone();
    let mut params = FaissSearchParams::default();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();

        if name == "file" {
            if let Ok(data) = field.bytes().await {
                upload_file = Some(data);
            }
        } else if name == "orb_scale_factor" {
            if let Ok(value) = field.text().await {
                if let Ok(value) = value.parse::<f32>() {
                    opts.orb_scale_factor = value;
                }
            }
        } else if name == "nprobe" {
            if let Ok(value) = field.text().await {
                if let Ok(value) = value.parse::<usize>() {
                    params.nprobe = value;
                }
            }
        } else if name == "max_codes" {
            if let Ok(value) = field.text().await {
                if let Ok(value) = value.parse::<usize>() {
                    params.max_codes = value;
                }
            }
        }
    }

    let start = Instant::now();

    info!("正在搜索上传图片");

    let mut orb = Slam3ORB::from(&opts);

    let upload_file = upload_file.context("没有找到上传文件")?;
    let mat = Mat::from_slice(&upload_file)?;
    let (_, des) = block_in_place(|| {
        let img = imgcodecs::imdecode(&mat, imgcodecs::IMREAD_GRAYSCALE)?;
        utils::detect_and_compute(&mut orb, &img)
    })?;

    let index = state.index.read().await;
    let mut result = state.db.search(&index, des, opts.knn_k, opts.distance, params).await?;
    result.truncate(opts.output_count);

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
