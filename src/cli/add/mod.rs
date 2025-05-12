use std::path::PathBuf;
use std::sync::Arc;

use anyhow::anyhow;
use clap::Parser;
use indicatif::ProgressBar;
use regex::Regex;
use tasks::*;

mod tasks;
mod types;

use crate::IMDBBuilder;
use crate::cli::SubCommandExtend;
use crate::config::{Opts, OrbOptions};
use crate::orb::*;
use crate::utils::{ImageHash, pb_style};

#[derive(Parser, Debug, Clone)]
pub struct AddCommand {
    #[command(flatten)]
    pub orb: OrbOptions,
    /// 图片所在目录，也支持扫描 tar 归档文件
    pub path: PathBuf,
    /// 扫描的文件后缀名，多个后缀用逗号分隔
    #[arg(short, long, default_value = "jpg,png,webp")]
    pub suffix: String,
    /// 在添加到数据库之前使用正则表达式对图片路径进行处理
    /// 例：--replace '/path/to/image/(?<name>[0-9]+).jpg' '$name'
    #[arg(short, long, value_names = ["REGEX", "REPLACE"], verbatim_doc_comment)]
    pub replace: Vec<String>,
    /// 最少特征点，低于该值的图片会被过滤
    #[arg(short, long, default_value_t = 250)]
    pub min_keypoints: u32,
    // FIXME: 如何确保用户一直在使用相同的哈希算法？
    /// 图片去重使用的哈希算法
    #[arg(short = 'H', long, default_value = "blake3")]
    pub hash: ImageHash,
    /// 如果图片已添加，是否覆盖旧的记录
    #[arg(long)]
    pub overwrite: bool,
}

impl SubCommandExtend for AddCommand {
    async fn run(&self, opts: &Opts) -> anyhow::Result<()> {
        ORB_OPTIONS.get_or_init(|| self.orb.clone());

        let re_suf = format!("(?i)({})", self.suffix.replace(',', "|"));
        let re_suf = Regex::new(&re_suf).expect("failed to build regex");
        let replace = if self.replace.is_empty() {
            None
        } else {
            let re = Regex::new(&self.replace[0]).expect("failed to build regex");
            let replace = self.replace[1].clone();
            Some((re, replace))
        };

        let db = Arc::new(IMDBBuilder::new(opts.conf_dir.clone()).open().await?);

        let pb = ProgressBar::no_length().with_style(pb_style());

        if let Some(hash) = db.guess_hash().await {
            if hash != self.hash {
                return Err(anyhow!("哈希算法不一致"));
            }
        }

        let (t1, rx) = task_scan(self.path.clone(), pb.clone(), re_suf);
        let (t2, rx) = task_hash(rx, self.hash, pb.clone());
        let (t3, rx) = task_filter(rx, pb.clone(), db.clone(), self.overwrite, replace.clone());
        let (t4, rx) = task_calc(rx, pb.clone());
        let t5 = task_add(rx, pb.clone(), db, self.min_keypoints as i32, self.overwrite, replace);

        // 等待所有任务完成
        let _ = tokio::try_join!(t1, t2, t3, t4, t5);

        pb.finish_with_message("图片添加完成");

        Ok(())
    }
}
