use anyhow::Result;
use clap::Parser;
use log::info;

use crate::cli::SubCommandExtend;
use crate::imdb::BuildOptions;
use crate::{IMDBBuilder, Opts};

#[derive(Parser, Debug, Clone)]
pub struct BuildCommand {
    /// 使用 OnDiskInvertedLists 格式合并，注意这种格式无法转回普通格式
    #[arg(long)]
    pub on_disk: bool,
    /// 构建索引时，多少张图片为一个批次
    #[arg(short, long, value_name = "SIZE", default_value_t = 100000)]
    pub batch_size: usize,
    /// 不合并文件，而是直接保存为多索引
    #[arg(long, conflicts_with_all = ["on_disk"])]
    pub no_merge: bool,
    /// 构建索引时使用的 efSearch 参数
    #[arg(long, value_name = "EF_SEARCH", default_value_t = 16)]
    pub ef_search: usize,
}

impl SubCommandExtend for BuildCommand {
    async fn run(&self, opts: &Opts) -> Result<()> {
        let db = IMDBBuilder::new(opts.conf_dir.clone()).open().await?;
        let options = BuildOptions {
            on_disk: self.on_disk,
            batch_size: self.batch_size,
            no_merge: self.no_merge,
            ef_search: self.ef_search,
        };
        db.build_index(options).await?;
        info!("构建索引成功");
        Ok(())
    }
}
