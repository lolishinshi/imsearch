use anyhow::Result;
use clap::Parser;
use log::info;
use ndarray_npy::write_npy;

use crate::cli::SubCommandExtend;
use crate::{IMDBBuilder, Opts};

#[derive(Parser, Debug, Clone)]
pub struct ExportCommand {}

impl SubCommandExtend for ExportCommand {
    async fn run(&self, opts: &Opts) -> Result<()> {
        let db = IMDBBuilder::new(opts.conf_dir.clone()).open().await?;
        let data = db.export().await?;
        write_npy("train.npy", &data)?;
        info!("导出成功");
        Ok(())
    }
}
