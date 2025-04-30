use std::collections::HashMap;

use blake3::Hash;
use clap::Parser;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator};
use log::info;
use opencv::core::MatTraitConst;
use rayon::prelude::*;
use regex::Regex;
use tokio::task::{block_in_place, spawn_blocking};
use walkdir::WalkDir;

use crate::cli::SubCommandExtend;
use crate::config::{Opts, OrbOptions};
use crate::orb::*;
use crate::utils::pb_style;
use crate::{IMDBBuilder, utils};

#[derive(Parser, Debug, Clone)]
pub struct AddCommand {
    #[command(flatten)]
    pub orb: OrbOptions,
    /// 图片或目录的路径
    pub path: String,
    /// 扫描的文件后缀名，多个后缀用逗号分隔
    #[arg(short, long, default_value = "jpg,png")]
    pub suffix: String,
    /// 不添加完整文件路径到数据库，而是使用正则表达式提取出 name 分组作为图片的唯一标识
    /// 例：`/path/to/image/(?<name>[0-9]+).jpg`
    #[arg(short, long, verbatim_doc_comment)]
    pub regex: Option<String>,
    /// 最少特征点，低于该值的图片会被过滤
    #[arg(short, long, default_value_t = 250)]
    pub min_keypoints: u32,
}

impl SubCommandExtend for AddCommand {
    async fn run(&self, opts: &Opts) -> anyhow::Result<()> {
        ORB_OPTIONS.get_or_init(|| self.orb.clone());

        let re_name = self.regex.as_ref().map(|re| Regex::new(re).expect("failed to build regex"));
        let re_suf = format!("(?i)({})", self.suffix.replace(',', "|"));
        let re_suf = Regex::new(&re_suf).expect("failed to build regex");
        let db = IMDBBuilder::new(opts.conf_dir.clone()).open().await?;

        // 收集所有符合条件的文件路径
        info!("开始扫描目录: {}", self.path);
        let entries = WalkDir::new(&self.path)
            .into_iter()
            .filter_map(|entry| {
                entry.ok().and_then(|entry| {
                    let path = entry.path().to_path_buf();
                    if path.is_file()
                        && re_suf.is_match(&path.extension().unwrap_or_default().to_string_lossy())
                    {
                        path.to_str().map(|x| x.to_string())
                    } else {
                        None
                    }
                })
            })
            .collect::<Vec<_>>();
        info!("扫描完成，共 {} 张图片", entries.len());

        // NOTE: 由于异步 + rayon 的组合实在麻烦，这里采用了将计算拆分为多轮，避免在异步上下文中使用 rayon
        let entries = block_in_place(|| {
            let pb = ProgressBar::new(entries.len() as u64)
                .with_style(pb_style())
                .with_message("计算图片哈希中...");
            entries
                .into_par_iter()
                .progress_with(pb)
                .filter_map(|entry| utils::hash_file(&entry).ok().map(|hash| (hash, entry)))
                .collect::<HashMap<_, _>>()
        });
        info!("计算哈希值完成，共 {} 张不重复图片", entries.len());

        let pb = ProgressBar::new(entries.len() as u64)
            .with_style(pb_style())
            .with_message("检查已添加图片...");
        let mut images: Vec<(String, Hash)> = vec![];
        for (hash, filename) in entries.into_iter().progress_with(pb) {
            if !db.check_hash(hash.as_bytes()).await? {
                images.push((filename, hash));
            }
        }
        info!("检查完成，共 {} 张新图片", images.len());

        // 创建进度条
        let pb = ProgressBar::new(images.len() as u64)
            .with_style(pb_style())
            .with_message("添加图片中...");

        let (tx, mut rx) = tokio::sync::mpsc::channel(num_cpus::get());

        let task1 = spawn_blocking({
            let pb = pb.clone();
            move || {
                images.into_par_iter().progress_with(pb.clone()).for_each(|(image, hash)| {
                    if let Ok((_, _, des)) = ORB.with(|orb| orb.borrow_mut().detect_file(&image)) {
                        pb.set_message(image.clone());
                        tx.blocking_send((image, hash, des)).unwrap();
                    } else {
                        pb.println(format!("处理失败: {}", image));
                    }
                })
            }
        });

        let task2 = tokio::spawn({
            let pb = pb.clone();
            let min_keypoints = self.min_keypoints as i32;
            async move {
                while let Some((image, hash, des)) = rx.recv().await {
                    if des.rows() <= min_keypoints {
                        pb.println(format!("特征点少于 {}: {}", min_keypoints, image));
                        continue;
                    }

                    let path = match &re_name {
                        Some(re) => {
                            let captures = re.captures(&image);
                            if let Some(name) =
                                captures.and_then(|c| c.name("name").map(|m| m.as_str()))
                            {
                                name.to_string()
                            } else {
                                pb.println(format!("提取图片名失败: {}", image));
                                continue;
                            }
                        }
                        None => image,
                    };

                    if let Err(e) = db.add_image(&path, hash.as_bytes(), des).await {
                        pb.println(format!("添加图片失败: {}: {}", path, e));
                    }
                }
            }
        });

        // 等待所有任务完成
        let _ = tokio::try_join!(task1, task2);

        // 完成后的消息
        pb.finish_with_message("图片处理完成！");
        Ok(())
    }
}
