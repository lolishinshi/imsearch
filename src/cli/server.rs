use std::env::set_current_dir;

use clap::Parser;
use log::info;
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
}

impl SubCommandExtend for ServerCommand {
    async fn run(&self, opts: &Opts) -> anyhow::Result<()> {
        let db = IMDBBuilder::new(opts.conf_dir.clone())
            .mmap(!self.search.no_mmap)
            .cache(true)
            .open()
            .await?;

        if opts.conf_dir.ondisk_ivf().exists() {
            if !self.search.no_mmap {
                return Err(anyhow::anyhow!("磁盘索引必须使用 --no-mmap 选项"));
            }
            set_current_dir(opts.conf_dir.path())?;
        }

        let index = db.get_index();

        // 创建应用状态
        let state = server::AppState::new(index, db, self.clone());

        // 创建应用
        let app = server::create_app(state);

        // 启动服务器
        info!("starting server at http://{}", &self.addr);
        let listener = TcpListener::bind(&self.addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}
