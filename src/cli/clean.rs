use anyhow::Result;
use clap::Parser;
use log::info;

use crate::cli::SubCommandExtend;
use crate::{IMDBBuilder, Opts};

#[derive(Parser, Debug, Clone)]
pub struct CleanCommand {
    /// 清理所有缓存，由于不需要筛选，速度更快
    #[arg(long)]
    pub all: bool,
}

impl SubCommandExtend for CleanCommand {
    async fn run(&self, opts: &Opts) -> Result<()> {
        let db = IMDBBuilder::new(opts.conf_dir.clone()).wal(false).open().await?;
        info!("清理缓存中……");
        db.clear_cache(self.all).await?;
        info!("清理完成");
        Ok(())
    }
}
