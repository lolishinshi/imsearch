use crate::cmd::SubCommandExtend;
use crate::config::{Opts, OutputFormat};
use crate::index::FaissSearchParams;
use crate::slam3_orb::Slam3ORB;
use crate::{IMDB, ORB, utils};
use anyhow::Result;
use blake3::Hash;
use clap::Parser;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator, ProgressStyle};
use log::info;
use opencv::core::MatTraitConst;
use rayon::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::task::{block_in_place, spawn_blocking};
use walkdir::WalkDir;

#[derive(Parser, Debug, Clone)]
pub struct AddImages {
    /// 图片或目录的路径
    pub path: String,
    /// 扫描的文件后缀名，多个后缀用逗号分隔
    #[arg(short, long, default_value = "jpg,png")]
    pub suffix: String,
}

#[derive(Parser, Debug, Clone)]
pub struct SearchImage {
    /// 被搜索的图片路径
    pub image: String,
    /// 搜索的倒排列表数量
    #[arg(short, long, default_value = "1")]
    pub nprobe: usize,
    /// 搜索的最大向量数量
    #[arg(short, long, default_value = "0")]
    pub max_codes: usize,
}

impl SubCommandExtend for AddImages {
    async fn run(&self, opts: &Opts) -> anyhow::Result<()> {
        let re = Regex::new(&self.suffix.replace(',', "|")).expect("failed to build regex");
        let db = IMDB::new(opts.conf_dir.clone()).await?;
        let pb_style = ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-");

        // 收集所有符合条件的文件路径
        info!("开始扫描目录: {}", self.path);
        let entries: Vec<PathBuf> = WalkDir::new(&self.path)
            .into_iter()
            .filter_map(|entry| {
                entry.ok().and_then(|entry| {
                    let path = entry.path().to_path_buf();
                    if path.is_file()
                        && re.is_match(&path.extension().unwrap_or_default().to_string_lossy())
                    {
                        Some(path)
                    } else {
                        None
                    }
                })
            })
            .collect();
        info!("扫描完成，共 {} 张图片", entries.len());

        // NOTE: 由于异步 + rayon 的组合实在麻烦，这里采用了将计算拆分为多轮，避免在异步上下文中使用 rayon
        let entries = block_in_place(|| {
            let pb = ProgressBar::new(entries.len() as u64)
                .with_style(pb_style.clone())
                .with_message("计算图片哈希中...");
            entries
                .into_par_iter()
                .progress_with(pb)
                .filter_map(|entry| utils::hash_file(&entry).ok().map(|hash| (hash, entry)))
                .collect::<HashMap<_, _>>()
        });
        info!("计算哈希值完成，共 {} 张不重复图片", entries.len());

        let pb = ProgressBar::new(entries.len() as u64)
            .with_style(pb_style.clone())
            .with_message("检查已添加图片...");
        let mut images: Vec<(PathBuf, Hash)> = vec![];
        for (hash, filename) in entries.into_iter().progress_with(pb) {
            if !db.check_hash(hash.as_bytes()).await? {
                images.push((filename, hash));
            }
        }
        info!("检查完成，共 {} 张新图片", images.len());

        // 创建进度条
        let pb = ProgressBar::new(images.len() as u64)
            .with_style(pb_style.clone())
            .with_message("添加图片中...");

        let (tx, mut rx) = tokio::sync::mpsc::channel(num_cpus::get());

        let task1 = spawn_blocking({
            let pb = pb.clone();
            move || {
                images.into_par_iter().progress_with(pb.clone()).for_each(|(image, hash)| {
                    if let Ok((_, des)) = utils::imread(image.to_string_lossy()).and_then(|image| {
                        ORB.with(|orb| utils::detect_and_compute(&mut orb.borrow_mut(), &image))
                    }) {
                        pb.set_message(image.display().to_string());
                        tx.blocking_send((image, hash, des)).unwrap();
                    } else {
                        pb.println(format!("处理失败: {}", image.display()));
                    }
                })
            }
        });

        let task2 = tokio::spawn({
            let pb = pb.clone();
            async move {
                while let Some((image, hash, des)) = rx.recv().await {
                    if des.rows() <= 10 {
                        pb.println(format!("特征点少于 10: {}", image.display()));
                        continue;
                    }
                    if let Err(e) =
                        db.add_image(image.to_string_lossy(), hash.as_bytes(), des).await
                    {
                        pb.println(format!("添加图片失败: {}: {}", image.display(), e));
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

impl SubCommandExtend for SearchImage {
    async fn run(&self, opts: &Opts) -> anyhow::Result<()> {
        let db = IMDB::new(opts.conf_dir.clone()).await?;
        let mut orb = Slam3ORB::from(opts);

        let index = db.get_index(opts.mmap);
        let params = FaissSearchParams { nprobe: self.nprobe, max_codes: self.max_codes };

        let (_, des) = block_in_place(|| {
            utils::imread(&self.image).and_then(|image| utils::detect_and_compute(&mut orb, &image))
        })?;

        let mut result =
            db.search(&index, des, opts.knn_k, opts.distance, opts.output_count, params).await?;

        result.truncate(opts.output_count);
        print_result(&result, opts)
    }
}

fn print_result(result: &[(f32, String)], opts: &Opts) -> Result<()> {
    match opts.output_format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(result)?)
        }
        OutputFormat::Table => {
            for (k, v) in result {
                println!("{:.2}\t{}", k, v);
            }
        }
    }
    Ok(())
}
