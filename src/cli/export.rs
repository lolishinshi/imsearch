use anyhow::Result;
use clap::Parser;
use log::info;
use ndarray_npy::write_npy;

use crate::cli::SubCommandExtend;
use crate::{IMDBBuilder, Opts};

#[derive(Parser, Debug, Clone)]
pub struct ExportCommand {
    /// 导出图片的数量，默认导出全部
    /// 注意总特征点数量不要超过 2^31-1，否则训练时会溢出
    #[clap(short, long)]
    pub count: Option<usize>,
    /// 保存文件的位置
    #[clap(short, long, default_value = "train.npy")]
    pub output: String,
}

impl SubCommandExtend for ExportCommand {
    async fn run(&self, opts: &Opts) -> Result<()> {
        let db = IMDBBuilder::new(opts.conf_dir.clone()).open().await?;
        let data = db.export(self.count).await?;
        write_npy(&self.output, &data)?;
        info!("导出成功");
        Ok(())
    }
}
