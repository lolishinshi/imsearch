use std::io::{self, Write};

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
    /// 不需要确认，强制清理
    #[arg(long)]
    pub force: bool,
}

impl SubCommandExtend for CleanCommand {
    async fn run(&self, opts: &Opts) -> Result<()> {
        let db = IMDBBuilder::new(opts.conf_dir.clone()).wal(false).open().await?;

        if !self.force {
            print!("确定要清理{}缓存吗？[y/N] ", if self.all { "所有" } else { "" });
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;

            if input.trim().to_lowercase() != "Y" {
                info!("操作已取消");
                return Ok(());
            }
        }

        info!("清理缓存中……");
        db.clear_cache(self.all).await?;
        info!("清理完成");
        Ok(())
    }
}
