use anyhow::Result;
use clap::Parser;
use log::info;

use crate::cli::SubCommandExtend;
use crate::{IMDBBuilder, Opts};

#[derive(Parser, Debug, Clone)]
pub struct BuildCommand {
    /// 使用 OnDiskInvertedLists 格式合并，注意这种格式无法再转回普通格式并且难以更新，建议仅在输出最终产物时使用
    #[arg(long)]
    pub on_disk: bool,
    /// 构建索引时，多少张图片为一个批次
    #[arg(short, long, value_name = "SIZE", default_value_t = 100000)]
    pub batch_size: usize,
}

impl SubCommandExtend for BuildCommand {
    async fn run(&self, opts: &Opts) -> Result<()> {
        let db =
            IMDBBuilder::new(opts.conf_dir.clone()).mmap(false).ondisk(self.on_disk).open().await?;
        db.build_index(self.batch_size).await?;
        info!("构建索引成功");
        Ok(())
    }
}
