use std::sync::Arc;

use tokio::sync::RwLock;

use crate::IMDB;
use crate::cli::server::ServerCommand;
use crate::config::{OrbOptions, SearchOptions};
use crate::faiss::FaissIndex;

/// 应用状态
pub struct AppState {
    /// Faiss索引
    pub index: RwLock<Option<Arc<FaissIndex>>>,
    /// 数据库连接
    pub db: IMDB,
    /// 服务器配置选项
    pub orb: OrbOptions,
    /// 搜索配置选项
    pub search: SearchOptions,
    /// 鉴权 token
    pub token: String,
}

impl AppState {
    /// 创建新的应用状态
    pub fn new(index: FaissIndex, db: IMDB, opts: ServerCommand) -> Arc<Self> {
        Arc::new(AppState {
            index: RwLock::new(Some(Arc::new(index))),
            db,
            orb: opts.orb,
            search: opts.search,
            token: opts.token,
        })
    }
}
