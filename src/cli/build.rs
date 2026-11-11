use anyhow::Result;
use clap::Parser;
use log::info;

use crate::cli::SubCommandExtend;
use crate::imdb::BuildOptions;
use crate::{IMDBBuilder, Opts};

/// 构建索引
#[derive(Parser, Debug, Clone)]
pub struct BuildCommand {
    /// 构建索引时，多少张图片为一个批次
    #[arg(short, long, value_name = "SIZE", default_value_t = 100000)]
    pub batch_size: usize,
    /// 生成临时索引后退出，再次运行 build 继续合并
    ///
    /// 同时指定 --clean-before-merge 时不执行清理
    #[arg(long)]
    pub stop_before_merge: bool,
    /// 合并前清理已索引的特征缓存，保留图片元数据
    ///
    /// 使用 VACUUM 回收空间，需要额外临时空间
    #[arg(long)]
    pub clean_before_merge: bool,
}

impl SubCommandExtend for BuildCommand {
    async fn run(&self, opts: &Opts) -> Result<()> {
        // 清理时关闭 WAL，避免 VACUUM 写出大体积的 WAL 文件。
        let db = IMDBBuilder::new(opts.conf_dir.clone())
            .wal(!self.clean_before_merge || self.stop_before_merge)
            .open()
            .await?;
        let options = BuildOptions {
            batch_size: self.batch_size,
            stop_before_merge: self.stop_before_merge,
            clean_before_merge: self.clean_before_merge,
        };
        db.build_index(options).await?;
        if !self.stop_before_merge {
            info!("构建索引成功");
        }
        Ok(())
    }
}
