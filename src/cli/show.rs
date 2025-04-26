use anyhow::Result;
use clap::Parser;

use crate::cli::SubCommandExtend;
use crate::config::{Opts, OrbOptions};
use crate::orb::Slam3ORB;
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
        let image = utils::imread(&self.image)?;

        let mut orb = Slam3ORB::from(&self.orb);
        let (kps, _) = utils::detect_and_compute(&mut orb, &image)?;
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
