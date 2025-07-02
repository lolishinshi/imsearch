use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use either::Either;
use futures::StreamExt;
use indicatif::ProgressBar;
use log::info;
use regex::Regex;
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use tokio::task::{JoinHandle, spawn_blocking};
use tokio_tar::Archive;
use walkdir::WalkDir;

use super::types::*;
use crate::IMDB;
use crate::orb::ORB;
use crate::utils::ImageHash;

pub fn task_scan(path: PathBuf, regex_suf: Regex) -> (JoinHandle<()>, Receiver<ImageData>) {
    let (tx, rx) = bounded(num_cpus::get());
    let t = tokio::spawn(async move {
        // NOTE: 这里刻意不使用 `?` 而是 unwrap，这是为了确保出错时正常崩溃
        // 如果上抛的话，上层就需要正确打印错误，太过麻烦，不如直接 panic
        if path.is_file() {
            scan_tar(path, tx, regex_suf).await.unwrap();
        } else {
            scan_directory(path, tx, regex_suf).await.unwrap();
        }
    });
    (t, rx)
}

pub fn task_hash(
    lrx: Receiver<ImageData>,
    hash: ImageHash,
    pb: ProgressBar,
) -> (JoinHandle<()>, Receiver<HashedImageData>) {
    let (tx, rx) = bounded(num_cpus::get());
    let t = spawn_blocking(move || {
        std::thread::scope(|s| {
            let threads = (num_cpus::get() / 2).max(1);
            for _ in 0..threads {
                s.spawn(|| {
                    while let Ok(data) = lrx.recv() {
                        match hash.hash_bytes(&data.data) {
                            Ok((img, val)) => {
                                tx.send(HashedImageData {
                                    path: data.path,
                                    data: img.map(Either::Left).unwrap_or(Either::Right(data.data)),
                                    hash: val,
                                })
                                .unwrap();
                            }
                            Err(_) => pb.println(format!("计算哈希失败: {}", data.path)),
                        }
                    }
                });
            }
        });
    });
    (t, rx)
}

pub fn task_filter(
    lrx: Receiver<HashedImageData>,
    pb: ProgressBar,
    db: Arc<IMDB>,
    duplicate: Duplicate,
    replace: Option<(Regex, String)>,
    distance: u32,
) -> (JoinHandle<()>, Receiver<HashedImageData>) {
    let (tx, rx) = bounded(num_cpus::get());
    let t = tokio::spawn(async move {
        futures::stream::iter(lrx)
            // 由于 check_hash 可能需要进行比较耗时的 KNN 查询
            // 因此这里使用 buffer_unordered 来并发处理
            // 但写入时需要使用 for_each 顺序写入，避免 sqlite3 的锁竞争
            .map(|data| async {
                let dup_id = db.check_hash(&data.hash, distance).await;
                (data, dup_id)
            })
            .buffer_unordered(8)
            .for_each(|(data, dup_id)| async {
                if let Some(id) = dup_id.unwrap() {
                    handle_duplicate(Either::Left(data), duplicate, id, replace.as_ref(), &db, &pb)
                        .await
                        .unwrap();
                    pb.inc(1);
                } else {
                    let tx = tx.clone();
                    spawn_blocking(move || tx.send(data).unwrap()).await.unwrap();
                }
            })
            .await;
    });
    (t, rx)
}

pub fn task_calc(
    lrx: Receiver<HashedImageData>,
    min_keypoints: u32,
    pb: ProgressBar,
) -> (JoinHandle<()>, Receiver<ProcessableImage>) {
    let (tx, rx) = unbounded();
    let t = spawn_blocking(move || {
        std::thread::scope(|s| {
            for _ in 0..num_cpus::get() {
                s.spawn(|| {
                    while let Ok(data) = lrx.recv() {
                        if let Ok((_, des)) = ORB.with(|orb| match data.data {
                            Either::Left(img) => orb.borrow_mut().detect_image(img),
                            Either::Right(bytes) => orb.borrow_mut().detect_bytes(&bytes),
                        }) {
                            if des.dim().0 <= min_keypoints as usize {
                                pb.set_message(format!(
                                    "特征点少于 {}: {}",
                                    min_keypoints, data.path
                                ));
                                pb.inc(1);
                            } else {
                                tx.send(ProcessableImage {
                                    path: data.path,
                                    hash: data.hash,
                                    descriptors: des,
                                })
                                .unwrap();
                            }
                        } else {
                            pb.set_message(format!("计算特征点失败: {}", data.path));
                            pb.inc(1);
                        }
                    }
                });
            }
        });
    });
    (t, rx)
}

pub fn task_add(
    lrx: Receiver<ProcessableImage>,
    pb: ProgressBar,
    db: Arc<IMDB>,
    duplicate: Duplicate,
    replace: Option<(Regex, String)>,
    phash_threshold: u32,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        futures::stream::iter(lrx)
            .for_each(|data| async {
                let path = match &replace {
                    Some((re, replace)) => &*re.replace(&data.path, replace),
                    None => &*data.path,
                };

                match db.check_hash(&data.hash, phash_threshold).await.unwrap() {
                    Some(id) => {
                        handle_duplicate(
                            Either::Right(data),
                            duplicate,
                            id,
                            replace.as_ref(),
                            &db,
                            &pb,
                        )
                        .await
                        .unwrap();
                    }
                    None => {
                        db.add_image(path, &data.hash, data.descriptors.view()).await.unwrap();
                        pb.set_message(path.to_owned());
                    }
                }

                pb.inc(1);
            })
            .await;
    })
}

async fn scan_directory(path: PathBuf, tx: Sender<ImageData>, regex_suf: Regex) -> Result<()> {
    info!("开始扫描目录: {}", path.display());

    futures::stream::iter(WalkDir::new(path))
        .filter_map(|entry| async {
            entry.ok().and_then(|entry| {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if regex_suf.is_match(&ext.to_string_lossy()) {
                            return Some(entry);
                        }
                    }
                }
                None
            })
        })
        .for_each_concurrent(32, |entry| {
            let tx = tx.clone();
            let path = entry.path().to_path_buf();
            async {
                let path_str = path.to_string_lossy().to_string();
                match path.extension().and_then(|ext| ext.to_str()) {
                    Some("tar") => scan_tar(path, tx, regex_suf.clone()).await.unwrap(),
                    _ => {
                        if let Ok(data) = tokio::fs::read(path).await {
                            spawn_blocking(move || {
                                tx.send(ImageData { path: path_str, data }).unwrap()
                            })
                            .await
                            .unwrap();
                        }
                    }
                }
            }
        })
        .await;

    Ok(())
}

async fn scan_tar(path: impl AsRef<Path>, tx: Sender<ImageData>, re_suf: Regex) -> Result<()> {
    let file = File::open(path).await?;
    let mut archive = Archive::new(file);
    let mut entries = archive.entries()?;

    // NOTE: tar 的 entries 必须按顺序读取，不能乱序并发
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

        let tx = tx.clone();
        spawn_blocking(move || tx.send(ImageData { path, data }).unwrap()).await?;
    }
    Ok(())
}

async fn handle_duplicate(
    data: Either<HashedImageData, ProcessableImage>,
    duplicate: Duplicate,
    duplicate_id: i64,
    replace: Option<&(Regex, String)>,
    db: &Arc<IMDB>,
    pb: &ProgressBar,
) -> Result<()> {
    let path = match data {
        Either::Left(data) => data.path,
        Either::Right(data) => data.path,
    };

    match duplicate {
        Duplicate::Overwrite => {
            let path = replace
                .map(|(re, replace)| re.replace(&path, replace))
                .unwrap_or(Cow::Borrowed(&path));
            db.update_image_path(duplicate_id, &path).await?;
            pb.set_message(format!("更新图片路径: {}", path));
        }
        Duplicate::Append => {
            let path = replace
                .map(|(re, replace)| re.replace(&path, replace))
                .unwrap_or(Cow::Borrowed(&path));
            db.append_image_path(duplicate_id, &path).await?;
            pb.set_message(format!("追加图片路径: {}", path));
        }
        Duplicate::Ignore => {
            pb.set_message(format!("跳过已添加图片: {}", path));
        }
    }
    Ok(())
}
