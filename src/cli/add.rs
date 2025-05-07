use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressIterator};
use log::info;
use opencv::core::MatTraitConst;
use regex::Regex;
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use tokio::sync::mpsc::{Sender, channel};
use tokio::task::{JoinHandle, block_in_place, spawn_blocking};
use tokio_tar::Archive;
use walkdir::WalkDir;

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
    #[arg(short, long, default_value = "jpg,png")]
    pub suffix: String,
    /// 不添加完整文件路径到数据库，而是使用正则表达式提取出 name 分组作为图片的唯一标识
    /// 例：`/path/to/image/(?<name>[0-9]+).jpg`
    #[arg(short, long, verbatim_doc_comment)]
    pub regex: Option<String>,
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

        let re_name = self.regex.as_ref().map(|re| Regex::new(re).expect("failed to build regex"));
        let re_suf = format!("(?i)({})", self.suffix.replace(',', "|"));
        let re_suf = Regex::new(&re_suf).expect("failed to build regex");
        let db = Arc::new(IMDBBuilder::new(opts.conf_dir.clone()).open().await?);

        let pb = ProgressBar::no_length().with_style(pb_style());

        // task1: 哈希计算
        let (hash_tx, mut hash_rx) = channel(num_cpus::get() * 2);
        let task1_hash: JoinHandle<Result<()>> = tokio::spawn({
            let hash = self.hash;
            let path = self.path.clone();
            let pb = pb.clone();

            async move {
                if path.is_file() {
                    hash_tar_file(path, hash, hash_tx, re_suf, pb).await
                } else {
                    hash_direcotry(path, hash, hash_tx, re_suf, pb).await
                }
            }
        });

        // task2: 检查已添加图片
        let (filter_tx, mut filter_rx) = channel(num_cpus::get() * 2);
        let task2_filter: JoinHandle<Result<()>> = tokio::spawn({
            let pb = pb.clone();
            let db = db.clone();
            let overwrite = self.overwrite;
            async move {
                while let Some((entry, data, hash)) = hash_rx.recv().await {
                    let exists = db.check_hash(&hash).await?;
                    if exists {
                        if overwrite {
                            db.update_image_path(&hash, &entry).await?;
                            pb.set_message(format!("更新图片路径: {}", entry));
                        } else {
                            pb.set_message(format!("跳过图片: {}", entry));
                        }
                        pb.inc(1);
                    } else {
                        filter_tx.send((entry, data, hash)).await?;
                    }
                }
                Ok(())
            }
        });

        // task3: 特征点计算
        let (feature_tx, mut feature_rx) = channel(num_cpus::get() * 2);
        let task3_feature = spawn_blocking({
            let pb = pb.clone();
            move || {
                let pb = &pb;
                let feature_tx = &feature_tx;
                rayon::scope(|s| {
                    while let Some((entry, data, hash)) = filter_rx.blocking_recv() {
                        s.spawn(move |_| {
                            if let Ok((_, _, des)) =
                                ORB.with(|orb| orb.borrow_mut().detect_bytes(&data))
                            {
                                feature_tx.blocking_send((entry, hash, des)).unwrap();
                            } else {
                                pb.set_message(format!("计算特征点失败: {}", entry));
                                pb.inc(1);
                            }
                        });
                    }
                });
            }
        });

        // task4: 添加图片
        let task4_add: JoinHandle<Result<()>> = tokio::spawn({
            let pb = pb.clone();
            let min_keypoints = self.min_keypoints as i32;
            let overwrite = self.overwrite;
            async move {
                while let Some((entry, hash, des)) = feature_rx.recv().await {
                    if des.rows() <= min_keypoints {
                        pb.set_message(format!("特征点少于 {}: {}", min_keypoints, entry));
                        pb.inc(1);
                        continue;
                    }

                    let path = match &re_name {
                        Some(re) => {
                            let captures = re.captures(&entry);
                            if let Some(name) = captures.and_then(|c| c.name("name")) {
                                name.as_str()
                            } else {
                                pb.println(format!("提取图片名失败: {}", entry));
                                pb.inc(1);
                                continue;
                            }
                        }
                        None => &entry,
                    };

                    if db.check_hash(&hash).await? {
                        if overwrite {
                            db.update_image_path(&hash, &path).await?;
                            pb.set_message(format!("更新图片路径: {}", path));
                        } else {
                            pb.set_message(format!("跳过图片: {}", path));
                        }
                    } else {
                        db.add_image(&path, &hash, des).await?;
                        pb.set_message(entry);
                    }

                    pb.inc(1);
                }
                Ok(())
            }
        });

        // 等待所有任务完成
        let _ = tokio::try_join!(task1_hash, task2_filter, task3_feature, task4_add);

        pb.finish_with_message("图片添加完成");

        Ok(())
    }
}

async fn hash_direcotry(
    path: impl AsRef<Path>,
    hash: ImageHash,
    hash_tx: Sender<(String, Vec<u8>, Vec<u8>)>,
    re_suf: Regex,
    pb: ProgressBar,
) -> Result<()> {
    info!("开始扫描目录: {}", path.as_ref().display());
    let pb2 = ProgressBar::no_length().with_style(pb_style());
    let entries = WalkDir::new(path)
        .into_iter()
        .progress_with(pb2)
        .filter_map(|entry| {
            entry.ok().and_then(|entry| {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if re_suf.is_match(&ext.to_string_lossy()) {
                            return Some(path.to_string_lossy().to_string());
                        }
                    }
                }
                None
            })
        })
        .collect::<Vec<_>>();
    info!("扫描完成，共 {} 张图片", entries.len());

    pb.set_length(entries.len() as u64);

    for entry in entries {
        let data = tokio::fs::read(&entry).await?;
        let hash_value = block_in_place(|| hash.hash_bytes(&data))?;
        hash_tx.send((entry, data, hash_value)).await?;
    }
    Ok(())
}

async fn hash_tar_file(
    path: PathBuf,
    hash: ImageHash,
    hash_tx: Sender<(String, Vec<u8>, Vec<u8>)>,
    re_suf: Regex,
    _pb: ProgressBar,
) -> Result<()> {
    let file = File::open(path).await?;
    let mut archive = Archive::new(file);
    let mut entries = archive.entries()?;

    while let Some(entry) = entries.next().await {
        let mut entry = entry?;
        // 此处存在非常烦人的 100 字节截断问题
        // 目前确认使用 astral-sh/tokio-tar + entry.path() 不会导致截断
        let path = entry.path()?;
        // 跳过目录
        if !entry.header().entry_type().is_file() {
            continue;
        }
        // 跳过不符合后缀的文件
        let Some(ext) = path.extension() else {
            continue;
        };
        if !re_suf.is_match(&ext.to_string_lossy()) {
            continue;
        }

        let path = path.to_string_lossy().to_string();

        let mut data = Vec::with_capacity(entry.header().size()? as usize);
        entry.read_to_end(&mut data).await?;

        let hash_value = block_in_place(|| hash.hash_bytes(&data))?;
        hash_tx.send((path, data, hash_value)).await?;
    }

    Ok(())
}
