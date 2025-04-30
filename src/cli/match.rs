use anyhow::Result;
use clap::Parser;
use log::info;
use opencv::core::*;
use opencv::prelude::*;
use opencv::{features2d, flann};

use crate::cli::SubCommandExtend;
use crate::config::{Opts, OrbOptions};
use crate::orb::ORBDetector;
use crate::utils;

#[derive(Parser, Debug, Clone)]
pub struct MatchCommand {
    #[command(flatten)]
    pub orb: OrbOptions,
    /// 图片1
    pub image1: String,
    /// 图片2
    pub image2: String,
    /// 不使用 GUI 展示，而是保存到文件
    pub output: Option<String>,
}

impl SubCommandExtend for MatchCommand {
    async fn run(&self, _opts: &Opts) -> Result<()> {
        let mut orb = ORBDetector::create(self.orb.clone());
        let (img1, kps1, des1) = orb.detect_file(&self.image1)?;
        let (img2, kps2, des2) = orb.detect_file(&self.image2)?;

        info!("图像1大小: {}x{}", img1.cols(), img1.rows());
        info!("特征点数量: {}", kps1.len());
        info!("图像2大小: {}x{}", img2.cols(), img2.rows());
        info!("特征点数量: {}", kps2.len());

        let mut matches = Vector::<Vector<DMatch>>::new();
        let mask = Mat::default();
        let flann = default_flann_matcher();
        flann.knn_train_match(&des1, &des2, &mut matches, 2, &mask, false)?;

        let mut matches_mask = vec![];
        for match_ in matches.iter() {
            if match_.len() != 2 {
                matches_mask.push(Vector::<i8>::from_iter([0, 0]));
                continue;
            }
            let (m, n) = (match_.get(0)?, match_.get(1)?);
            if m.distance < 0.7 * n.distance {
                matches_mask.push(Vector::<i8>::from_iter([1, 0]));
            } else {
                matches_mask.push(Vector::<i8>::from_iter([0, 0]));
            }
        }
        let matches_mask = Vector::<Vector<i8>>::from(matches_mask);

        let output = utils::draw_matches_knn(&img1, &kps1, &img2, &kps2, &matches, &matches_mask)?;
        match &self.output {
            Some(file) => {
                utils::imwrite(file, &output)?;
            }
            _ => utils::imshow("result", &output)?,
        }

        Ok(())
    }
}

fn default_flann_matcher() -> features2d::FlannBasedMatcher {
    let index_params = Ptr::new(flann::IndexParams::from(
        flann::LshIndexParams::new(6, 12, 1).expect("failed to build LshIndexParams"),
    ));
    let search_params =
        Ptr::new(flann::SearchParams::new_1(32, 0.0, true).expect("failed to build SearchParams"));
    features2d::FlannBasedMatcher::new(&index_params, &search_params)
        .expect("failed to build FlannBasedMatcher")
}
