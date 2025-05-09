use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Result, anyhow};
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
        let db = Arc::new(IMDBBuilder::new(opts.conf_dir.clone()).open().await?);

        let pb = ProgressBar::no_length().with_style(pb_style());

        if let Some(hash) = db.guess_hash().await {
            if hash != self.hash {
                return Err(anyhow!("哈希算法不一致"));
            }
        }

        // task1: 哈希计算
        let (hash_tx, mut hash_rx) = channel(num_cpus::get());
        let task1_hash = tokio::spawn({
            let hash = self.hash;
            let path = self.path.clone();
            let pb = pb.clone();

            async move {
                // NOTE: 这里刻意不使用 `?` 而是 unwrap，这是为了确保出错时正常崩溃
                // 如果上抛的话，上层就需要正确打印错误，太过麻烦，不如直接 panic
                if path.is_file() {
                    hash_tar_file(path, hash, hash_tx, re_suf, pb).await.unwrap();
                } else {
                    hash_direcotry(path, hash, hash_tx, re_suf, pb).await.unwrap();
                }
            }
        });

        // task2: 检查已添加图片
        let (filter_tx, mut filter_rx) = channel(num_cpus::get());
        let task2_filter = tokio::spawn({
            let pb = pb.clone();
            let db = db.clone();
            let overwrite = self.overwrite;
            async move {
                while let Some((entry, data, hash)) = hash_rx.recv().await {
                    let exists = db.check_hash(&hash).await.unwrap();
                    if exists {
                        if overwrite {
                            db.update_image_path(&hash, &entry).await.unwrap();
                            pb.set_message(format!("更新图片路径: {}", entry));
                        } else {
                            pb.set_message(format!("跳过已添加图片: {}", entry));
                        }
                        pb.inc(1);
                    } else {
                        filter_tx.send((entry, data, hash)).await.unwrap();
                    }
                }
            }
        });

        // task3: 特征点计算
        let (feature_tx, mut feature_rx) = channel(num_cpus::get());
        let task3_feature = spawn_blocking({
            let pb = pb.clone();
            let (tx, rx) =
                crossbeam_channel::bounded::<(String, Vec<u8>, Vec<u8>)>(num_cpus::get());
            move || {
                std::thread::scope(|s| {
                    for _ in 0..num_cpus::get() {
                        let pb = &pb;
                        let feature_tx = &feature_tx;
                        let rx = rx.clone();
                        s.spawn(move || {
                            while let Ok((entry, data, hash)) = rx.recv() {
                                if let Ok((_, _, des)) =
                                    ORB.with(|orb| orb.borrow_mut().detect_bytes(&data))
                                {
                                    feature_tx.blocking_send((entry, hash, des)).unwrap();
                                } else {
                                    pb.set_message(format!("计算特征点失败: {}", entry));
                                    pb.inc(1);
                                }
                            }
                        });
                    }
                    while let Some((entry, data, hash)) = filter_rx.blocking_recv() {
                        tx.send((entry, data, hash)).unwrap();
                    }
                    drop(tx);
                });
            }
        });

        // task4: 添加图片
        let replace = if self.replace.is_empty() {
            None
        } else {
            let re = Regex::new(&self.replace[0]).expect("failed to build regex");
            let replace = self.replace[1].clone();
            Some((re, replace))
        };
        let task4_add = tokio::spawn({
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

                    let path = match &replace {
                        Some((re, replace)) => &*re.replace(&entry, replace),
                        None => &*entry,
                    };

                    if db.check_hash(&hash).await.unwrap() {
                        if overwrite {
                            db.update_image_path(&hash, path).await.unwrap();
                            pb.set_message(format!("更新图片路径: {}", path));
                        } else {
                            pb.set_message(format!("跳过已添加图片: {}", path));
                        }
                    } else {
                        db.add_image(&path, &hash, des).await.unwrap();
                        pb.set_message(path.to_owned());
                    }

                    pb.inc(1);
                }
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

    futures::stream::iter(entries)
        .for_each_concurrent(num_cpus::get(), |entry| async {
            if let Ok(data) = tokio::fs::read(&entry).await {
                if let Ok(val) = block_in_place(|| hash.hash_bytes(&data)) {
                    hash_tx.send((entry, data, val)).await.unwrap();
                    return;
                }
            }
            pb.println(format!("计算哈希失败: {}", entry));
            pb.inc(1);
        })
        .await;

    Ok(())
}

async fn hash_tar_file(
    path: PathBuf,
    hash: ImageHash,
    hash_tx: Sender<(String, Vec<u8>, Vec<u8>)>,
    re_suf: Regex,
    pb: ProgressBar,
) -> Result<()> {
    let file = File::open(path).await?;
    let mut archive = Archive::new(file);
    let mut entries = archive.entries()?;

    // NOTE: tar 的 entries 必须按顺序读取，不能乱序并发
    let (tx, mut rx) = channel(1);
    let t1: JoinHandle<Result<()>> = tokio::spawn(async move {
        while let Some(entry) = entries.next().await {
            let mut entry = entry?;
            // 此处存在非常烦人的 100 字节截断问题
            // 目前确认使用 astral-sh/tokio-tar + entry.path() 不会导致截断
            let path = entry.path()?;
            // 跳过不符合条件的文件
            if !entry.header().entry_type().is_file() {
                continue;
            }
            let Some(ext) = path.extension() else {
                continue;
            };
            if !re_suf.is_match(&ext.to_string_lossy()) {
                continue;
            }

            let path = path.to_string_lossy().to_string();

            let mut data = Vec::with_capacity(entry.header().size()? as usize);
            entry.read_to_end(&mut data).await?;

            tx.send((path, data)).await?;
        }
        Ok(())
    });

    let t2 = tokio::spawn(async move {
        while let Some((path, data)) = rx.recv().await {
            match block_in_place(|| hash.hash_bytes(&data)) {
                Ok(hash_value) => hash_tx.send((path.clone(), data, hash_value)).await.unwrap(),
                Err(_) => {
                    pb.println(format!("计算哈希失败: {}", path.clone()));
                    pb.inc(1);
                }
            }
        }
    });

    let _ = tokio::try_join!(t1, t2);

    Ok(())
}
