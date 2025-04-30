use anyhow::Result;
use clap::Parser;
use log::info;
use opencv::prelude::*;

use crate::cli::SubCommandExtend;
use crate::config::{Opts, OrbOptions};
use crate::orb::ORBDetector;
use crate::utils;

#[derive(Parser, Debug, Clone)]
pub struct ShowCommand {
    #[command(flatten)]
    pub orb: OrbOptions,
    /// 图片路径
    pub image: String,
    /// 不使用 GUI 展示，而是保存到文件
    pub output: Option<String>,
}

impl SubCommandExtend for ShowCommand {
    async fn run(&self, _opts: &Opts) -> Result<()> {
        let mut orb = ORBDetector::create(self.orb.clone());
        let (image, kps, _) = orb.detect_file(&self.image)?;
        info!("图像大小: {}x{}", image.cols(), image.rows());
        info!("特征点数量: {}", kps.len());

        let output = utils::draw_keypoints(&image, &kps)?;

        match &self.output {
            Some(file) => {
                utils::imwrite(file, &output)?;
            }
            _ => utils::imshow("result", &output)?,
        }
        Ok(())
    }
}
