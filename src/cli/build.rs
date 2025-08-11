use anyhow::Result;
use clap::Parser;
use log::info;

use crate::cli::SubCommandExtend;
use crate::imdb::BuildOptions;
use crate::{IMDBBuilder, Opts};

#[derive(Parser, Debug, Clone)]
pub struct BuildCommand {
    /// 构建索引时，多少张图片为一个批次
    #[arg(short, long, value_name = "SIZE", default_value_t = 100000)]
    pub batch_size: usize,
}

impl SubCommandExtend for BuildCommand {
    async fn run(&self, opts: &Opts) -> Result<()> {
        let db = IMDBBuilder::new(opts.conf_dir.clone()).open().await?;
        let options = BuildOptions { batch_size: self.batch_size };
        db.build_index(options).await?;
        info!("构建索引成功");
        Ok(())
    }
}
