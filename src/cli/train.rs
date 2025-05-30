use anyhow::Result;
use clap::Parser;

use crate::cli::SubCommandExtend;
use crate::invlists::ArrayInvertedLists;
use crate::ivf::{IvfHnsw, USearchQuantizer};
use crate::{IMDBBuilder, Opts};

#[derive(Parser, Debug, Clone)]
pub struct TrainCommand {
    /// 聚类中心点数量
    #[arg(short, long)]
    pub centers: usize,
    /// 用于训练的图片数量，推荐使用中心点数量的 6% ~ 50%
    #[arg(short, long)]
    pub images: usize,
    /// 最大迭代次数
    #[arg(short, long, default_value_t = 20)]
    pub max_iter: usize,
}

impl SubCommandExtend for TrainCommand {
    async fn run(&self, opts: &Opts) -> Result<()> {
        let db = IMDBBuilder::new(opts.conf_dir.clone()).open().await?;
        let data = db.export(Some(self.images)).await?;

        let quantizer = USearchQuantizer::<32>::new()?;
        let invlists = ArrayInvertedLists::new(self.centers as u32);
        let mut ivf = IvfHnsw::new(quantizer, invlists, self.centers);

        ivf.train(data.as_slice().unwrap(), self.max_iter)?;
        ivf.save("quantizer.bin")?;
        Ok(())
    }
}
