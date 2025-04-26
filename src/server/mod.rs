mod api;
mod error;
mod state;
mod types;

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use tower_http::limit::RequestBodyLimitLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

pub use self::state::*;

#[derive(OpenApi)]
#[openapi(paths(api::search_handler,), components(schemas(types::SearchForm,),))]
pub struct ApiDoc;

/// 构建API服务器
pub fn create_app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/search", axum::routing::post(api::search_handler))
        .merge(SwaggerUi::new("/docs").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .layer(DefaultBodyLimit::disable())
        // 上传限制：10M
        .layer(RequestBodyLimitLayer::new(1024 * 1024 * 10))
        .with_state(state)
}
