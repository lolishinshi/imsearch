use anyhow::Result;
use clap::Parser;
use log::info;

use crate::cli::SubCommandExtend;
use crate::{IMDBBuilder, Opts};

#[derive(Parser, Debug, Clone)]
pub struct BuildCOmmand {
    /// 使用 mmap 模式合并索引
    #[arg(long)]
    pub mmap: bool,
    /// 构建索引时，多少张图片为一个批次
    #[arg(long, value_name = "SIZE", default_value_t = 10000)]
    pub batch_size: usize,
}

impl SubCommandExtend for BuildCOmmand {
    async fn run(&self, opts: &Opts) -> Result<()> {
        let db = IMDBBuilder::new(opts.conf_dir.clone()).mmap(self.mmap).open().await?;
        db.build_index(self.batch_size).await?;
        info!("构建索引成功");
        Ok(())
    }
}
