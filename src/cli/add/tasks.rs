use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use either::Either;
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressIterator};
use log::info;
use opencv::core::MatTraitConst;
use regex::Regex;
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use tokio::sync::mpsc::{Receiver, Sender, channel};
use tokio::task::{JoinHandle, block_in_place, spawn_blocking};
use tokio_tar::Archive;
use walkdir::WalkDir;

use super::types::*;
use crate::IMDB;
use crate::orb::ORB;
use crate::utils::{ImageHash, pb_style};

pub fn task1_scan_and_hash(
    hash: ImageHash,
    path: PathBuf,
    pb: ProgressBar,
    regex_suf: Regex,
) -> (JoinHandle<()>, Receiver<HashedImageData>) {
    let (tx, rx) = channel::<HashedImageData>(num_cpus::get());
    let t = tokio::spawn({
        async move {
            // NOTE: 这里刻意不使用 `?` 而是 unwrap，这是为了确保出错时正常崩溃
            // 如果上抛的话，上层就需要正确打印错误，太过麻烦，不如直接 panic
            if path.is_file() {
                hash_tar_file(path, hash, tx, regex_suf, pb).await.unwrap();
            } else {
                hash_directory(path, hash, tx, regex_suf, pb).await.unwrap();
            }
        }
    });
    (t, rx)
}

pub fn task2_filter(
    mut lrx: Receiver<HashedImageData>,
    pb: ProgressBar,
    db: Arc<IMDB>,
    overwrite: bool,
    replace: Option<(Regex, String)>,
) -> (JoinHandle<()>, Receiver<HashedImageData>) {
    let (tx, rx) = channel::<HashedImageData>(num_cpus::get());
    let t = tokio::spawn({
        async move {
            while let Some(data) = lrx.recv().await {
                let exists = db.check_hash(&data.hash).await.unwrap();
                if exists {
                    if overwrite {
                        let path = match &replace {
                            Some((re, replace)) => &*re.replace(&data.path, replace),
                            None => &*data.path,
                        };
                        db.update_image_path(&data.hash, path).await.unwrap();
                        pb.set_message(format!("更新图片路径: {}", path));
                    } else {
                        pb.set_message(format!("跳过已添加图片: {}", data.path));
                    }
                    pb.inc(1);
                } else {
                    tx.send(data).await.unwrap();
                }
            }
        }
    });
    (t, rx)
}

pub fn task3_calc(
    mut lrx: Receiver<HashedImageData>,
    pb: ProgressBar,
) -> (JoinHandle<()>, Receiver<ProcessableImage>) {
    let (tx, rx) = channel::<ProcessableImage>(num_cpus::get());
    let (ttx, rrx) = crossbeam_channel::bounded::<HashedImageData>(num_cpus::get());
    let t = spawn_blocking(move || {
        std::thread::scope(|s| {
            for _ in 0..num_cpus::get() {
                let pb = &pb;
                let feature_tx = &tx;
                let rrx = rrx.clone();
                s.spawn(move || {
                    while let Ok(data) = rrx.recv() {
                        if let Ok((_, des)) = ORB.with(|orb| match data.data {
                            Either::Left(ref img) => orb.borrow_mut().detect_image(img),
                            Either::Right(ref bytes) => orb.borrow_mut().detect_bytes(bytes),
                        }) {
                            feature_tx
                                .blocking_send(ProcessableImage {
                                    path: data.path.clone(),
                                    hash: data.hash,
                                    descriptors: des,
                                })
                                .unwrap();
                        } else {
                            pb.set_message(format!("计算特征点失败: {}", data.path));
                            pb.inc(1);
                        }
                    }
                });
            }
            while let Some(data) = lrx.blocking_recv() {
                ttx.send(data).unwrap();
            }
            drop(ttx);
        });
    });
    (t, rx)
}

pub fn task4_add(
    mut lrx: Receiver<ProcessableImage>,
    pb: ProgressBar,
    db: Arc<IMDB>,
    min_keypoints: i32,
    overwrite: bool,
    replace: Option<(Regex, String)>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(data) = lrx.recv().await {
            if data.descriptors.rows() <= min_keypoints {
                pb.set_message(format!("特征点少于 {}: {}", min_keypoints, data.path));
                pb.inc(1);
                continue;
            }

            let path = match &replace {
                Some((re, replace)) => &*re.replace(&data.path, replace),
                None => &*data.path,
            };

            // 这里再检查一次，因为可能存在处理过程中新增的重复图片
            if db.check_hash(&data.hash).await.unwrap() {
                if overwrite {
                    db.update_image_path(&data.hash, path).await.unwrap();
                    pb.set_message(format!("更新图片路径: {}", path));
                } else {
                    pb.set_message(format!("跳过已添加图片: {}", path));
                }
            } else {
                db.add_image(path, &data.hash, data.descriptors).await.unwrap();
                pb.set_message(path.to_owned());
            }

            pb.inc(1);
        }
    })
}

async fn hash_directory(
    path: PathBuf,
    hash: ImageHash,
    tx: Sender<HashedImageData>,
    regex_suf: Regex,
    pb: ProgressBar,
) -> Result<()> {
    info!("开始扫描目录: {}", path.display());
    let pb2 = ProgressBar::no_length().with_style(pb_style());
    let entries = WalkDir::new(path)
        .into_iter()
        .progress_with(pb2)
        .filter_map(|entry| {
            entry.ok().and_then(|entry| {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if regex_suf.is_match(&ext.to_string_lossy()) {
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
                hash_and_send(data, entry, hash, &tx, &pb).await.unwrap();
            }
        })
        .await;

    Ok(())
}

async fn hash_tar_file(
    path: PathBuf,
    hash: ImageHash,
    tx: Sender<HashedImageData>,
    re_suf: Regex,
    pb: ProgressBar,
) -> Result<()> {
    let file = File::open(path).await?;
    let mut archive = Archive::new(file);
    let mut entries = archive.entries()?;

    // NOTE: tar 的 entries 必须按顺序读取，不能乱序并发
    let (ttx, mut rrx) = channel(1);
    let t1: JoinHandle<Result<()>> = tokio::spawn(async move {
        while let Some(entry) = entries.next().await {
            let mut entry = entry?;
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

            ttx.send((path, data)).await?;
        }
        Ok(())
    });

    let t2 = tokio::spawn(async move {
        while let Some((path, data)) = rrx.recv().await {
            hash_and_send(data, path, hash, &tx, &pb).await.unwrap();
        }
    });

    let _ = tokio::try_join!(t1, t2);

    Ok(())
}

async fn hash_and_send(
    data: Vec<u8>,
    path: String,
    hash: ImageHash,
    tx: &Sender<HashedImageData>,
    pb: &ProgressBar,
) -> Result<()> {
    match block_in_place(|| hash.hash_bytes(&data)) {
        Ok((img, val)) => match img {
            Some(img) => {
                tx.send(HashedImageData { path, data: Either::Left(img), hash: val }).await?
            }
            None => tx.send(HashedImageData { path, data: Either::Right(data), hash: val }).await?,
        },
        Err(_) => {
            pb.println(format!("计算哈希失败: {}", path));
            pb.inc(1);
        }
    }
    Ok(())
}
