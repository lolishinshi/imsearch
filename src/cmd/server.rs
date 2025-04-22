use crate::cmd::SubCommandExtend;
use crate::utils;
use crate::{index::FaissIndex, Opts, Slam3ORB, IMDB};
use axum::{
    body::Body,
    extract::{Multipart, State},
    http::StatusCode,
    response::Response,
    routing::{get, post},
    Json, Router,
};
use log::info;
use opencv::imgcodecs;
use opencv::prelude::*;
use serde::Deserialize;
use serde_json::json;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use structopt::StructOpt;
use tokio::net::TcpListener;
use tokio::task::block_in_place;
use tower_http::limit::RequestBodyLimitLayer;

#[derive(StructOpt, Debug, Clone)]
pub struct StartServer {
    /// Listen address
    #[structopt(long, default_value = "127.0.0.1:8000")]
    pub addr: String,
}

// 定义共享状态
struct AppState {
    index: RwLock<FaissIndex>,
    db: IMDB,
    opts: Opts,
}

// 定义请求和响应类型
#[derive(Deserialize)]
struct SetNprobeRequest {
    n: usize,
}

impl SubCommandExtend for StartServer {
    #[tokio::main]
    async fn run(&self, opts: &Opts) -> anyhow::Result<()> {
        let db = IMDB::new(opts.conf_dir.clone(), true)?;

        let mut index = db.get_index(opts.mmap);
        index.set_nprobe(opts.nprobe);

        // 创建共享状态
        let state = Arc::new(AppState {
            index: RwLock::new(index),
            db,
            opts: opts.clone(),
        });

        // 创建路由
        let app = Router::new()
            .route("/", get(index_handler))
            .route("/search", post(search_handler))
            .route("/set_nprobe", post(set_nprobe_handler))
            // 限制上传文件大小为50MB
            .layer(RequestBodyLimitLayer::new(1024 * 1024 * 50))
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
        <code>http --form http://127.0.0.1:8000/search file@test.jpg orb_scale_factor=1.2</code>
        <br>
        <code>http --json http://127.0.0.1:8000/set_nprobe n=128</code>
        </p>
        "#,
        ))
        .unwrap()
}

// 处理搜索请求
async fn search_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response<Body> {
    // 处理上传的文件和参数
    let mut file_data = Vec::new();
    let mut orb_scale_factor = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();

        if name == "file" {
            if let Ok(data) = field.bytes().await {
                file_data = data.to_vec();
            }
        } else if name == "orb_scale_factor" {
            if let Ok(value) = field.text().await {
                orb_scale_factor = value.parse::<f32>().ok();
            }
        }
    }

    if file_data.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "No file uploaded");
    }

    // 处理图像
    let mat = match Mat::from_slice(&file_data) {
        Ok(mat) => mat,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid image data: {}", e),
            )
        }
    };

    let img = match block_in_place(|| imgcodecs::imdecode(&mat, imgcodecs::IMREAD_GRAYSCALE)) {
        Ok(img) => img,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("Failed to decode image: {}", e),
            )
        }
    };

    info!("searching uploaded image");

    // 处理搜索
    let mut opts = state.opts.clone();
    if let Some(factor) = orb_scale_factor {
        opts.orb_scale_factor = factor;
    }
    let mut orb = Slam3ORB::from(&opts);

    let start = Instant::now();
    let result = block_in_place(|| {
        utils::detect_and_compute(&mut orb, &img).and_then(|(_, descriptors)| {
            let index = state.index.read().expect("failed to acquire rw lock");
            state
                .db
                .search_des(&*index, descriptors, opts.knn_k, opts.distance)
        })
    });
    let elapsed = start.elapsed().as_secs_f32();

    match result {
        Ok(mut result) => {
            result.truncate(opts.output_count);
            json_response(json!({
                "time": elapsed,
                "result": result,
            }))
        }
        Err(err) => error_response(StatusCode::BAD_REQUEST, err.to_string()),
    }
}

// 处理设置nprobe请求
async fn set_nprobe_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SetNprobeRequest>,
) -> Response<Body> {
    match state.index.write() {
        Ok(mut index) => {
            index.set_nprobe(payload.n);
            Response::builder()
                .status(StatusCode::OK)
                .body(Body::from(""))
                .unwrap_or_else(|_| {
                    error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Failed to build response",
                    )
                })
        }
        Err(_) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to acquire write lock on index",
        ),
    }
}

// 创建一个辅助函数来生成错误响应
fn error_response(status: StatusCode, message: impl Into<String>) -> Response<Body> {
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(json!({"error": message.into()}).to_string()))
        .unwrap_or_else(|_| {
            // 如果构建响应失败，返回500错误
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from("Internal server error"))
                .unwrap()
        })
}

// 创建一个辅助函数来生成成功响应
fn json_response<T: serde::Serialize>(data: T) -> Response<Body> {
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json!(data).to_string()))
        .unwrap_or_else(|_| {
            // 如果构建响应失败，返回500错误
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from("Internal server error"))
                .unwrap()
        })
}
