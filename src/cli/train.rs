use anyhow::Result;
use clap::{Parser, ValueEnum};

use crate::cli::SubCommandExtend;
use crate::ivf::{HnswQuantizer, Quantizer};
use crate::kmodes::*;
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
    /// 初始化方法
    #[arg(short = 'I', long, default_value = "random")]
    pub init: InitMethod,
    /// 禁止使用二级聚类
    #[arg(short, long)]
    pub no_2level: bool,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
pub enum InitMethod {
    Random,
    KmeansPlusPlus,
}

impl SubCommandExtend for TrainCommand {
    async fn run(&self, opts: &Opts) -> Result<()> {
        let db = IMDBBuilder::new(opts.conf_dir.clone()).open().await?;
        let data = db.export(Some(self.images)).await?;

        let init_method = match self.init {
            InitMethod::Random => KModesInitMethod::Random,
            InitMethod::KmeansPlusPlus => KModesInitMethod::KmeansPlusPlus,
        };

        let centroids = if !self.no_2level {
            kmodes_2level::<32>(&data, self.centers, self.max_iter, init_method).centroids
        } else {
            kmodes_binary::<32>(&data, self.centers, self.max_iter, init_method).centroids
        };

        let quantizer = HnswQuantizer::init(&centroids)?;
        quantizer.save(&opts.conf_dir.join("quantizer.bin"))?;
        Ok(())
    }
}
