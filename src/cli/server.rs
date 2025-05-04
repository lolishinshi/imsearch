use clap::Parser;
use log::info;
use rand::distr::{Alphanumeric, SampleString};
use tokio::net::TcpListener;

use crate::cli::SubCommandExtend;
use crate::config::{OrbOptions, SearchOptions};
use crate::{IMDBBuilder, Opts, server};

#[derive(Parser, Debug, Clone)]
pub struct ServerCommand {
    #[command(flatten)]
    pub orb: OrbOptions,
    #[command(flatten)]
    pub search: SearchOptions,
    /// 监听地址
    #[arg(long, default_value = "127.0.0.1:8000")]
    pub addr: String,
    /// 请求验证 token，不填则随机生成
    #[arg(long, default_value_t = String::new())]
    pub token: String,
    /// 转换为 hnsw 索引加载
    #[arg(long)]
    pub hnsw: bool,
}

impl SubCommandExtend for ServerCommand {
    async fn run(&self, opts: &Opts) -> anyhow::Result<()> {
        let db = IMDBBuilder::new(opts.conf_dir.clone())
            .mmap(!self.search.no_mmap)
            .cache(true)
            .open()
            .await?;

        let mut index = db.get_index();
        if self.hnsw {
            index.to_hnsw();
        }

        let mut self_clone = self.clone();
        if self_clone.token.is_empty() {
            self_clone.token = Alphanumeric.sample_string(&mut rand::rng(), 32);
            info!("鉴权 token: {}", self_clone.token);
        }

        // 创建应用状态
        let state = server::AppState::new(index, db, self_clone);

        // 创建应用
        let app = server::create_app(state);

        // 启动服务器
        info!("服务器启动：http://{}", &self.addr);
        let listener = TcpListener::bind(&self.addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}
