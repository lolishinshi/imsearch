use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, bounded};
use either::Either;
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressIterator};
use itertools::Itertools;
use log::info;
use opencv::core::MatTraitConst;
use parking_lot::Mutex;
use regex::Regex;
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use tokio::task::{JoinHandle, spawn_blocking};
use tokio_tar::Archive;
use walkdir::WalkDir;
use yastl::Pool;

use super::types::*;
use crate::IMDB;
use crate::orb::ORB;
use crate::utils::{ImageHash, pb_style, pb_style_speed};

static POOL: LazyLock<Pool> = LazyLock::new(|| Pool::new(num_cpus::get()));

pub fn task_scan(
    path: PathBuf,
    pb: ProgressBar,
    regex_suf: Regex,
) -> (JoinHandle<()>, Receiver<ImageData>) {
    let (tx, rx) = bounded(num_cpus::get());
    let t = tokio::spawn(async move {
        // NOTE: 这里刻意不使用 `?` 而是 unwrap，这是为了确保出错时正常崩溃
        // 如果上抛的话，上层就需要正确打印错误，太过麻烦，不如直接 panic
        if path.is_file() {
            scan_tar(path, tx, regex_suf, pb).await.unwrap();
        } else {
            scan_directory(path, tx, regex_suf, pb).await.unwrap();
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
        for chunk in &lrx.iter().chunks(num_cpus::get() * 10) {
            // 由于 hash 计算非常快，这里先将结果收集到 Vec 中，最后再发送
            // 否则可能因为 channel 堵塞导致线程无法及时结束
            let result = Arc::new(Mutex::new(Vec::new()));
            POOL.scoped(|s| {
                for data in chunk {
                    s.execute(|| match hash.hash_bytes(&data.data) {
                        Ok((img, val)) => {
                            result.lock().push(HashedImageData {
                                path: data.path,
                                data: img.map(Either::Left).unwrap_or(Either::Right(data.data)),
                                hash: val,
                            });
                        }
                        Err(_) => pb.println(format!("计算哈希失败: {}", data.path)),
                    });
                }
            });
            for data in result.lock().drain(..) {
                tx.send(data).unwrap();
            }
        }
    });
    (t, rx)
}

pub fn task_filter(
    lrx: Receiver<HashedImageData>,
    pb: ProgressBar,
    db: Arc<IMDB>,
    overwrite: bool,
    replace: Option<(Regex, String)>,
) -> (JoinHandle<()>, Receiver<HashedImageData>) {
    let (tx, rx) = bounded(num_cpus::get());
    let t = tokio::spawn(async move {
        while let Ok(data) = lrx.recv() {
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
                tx.send(data).unwrap();
            }
        }
    });
    (t, rx)
}

pub fn task_calc(
    lrx: Receiver<HashedImageData>,
    pb: ProgressBar,
) -> (JoinHandle<()>, Receiver<ProcessableImage>) {
    let (tx, rx) = bounded(num_cpus::get());
    let t = spawn_blocking(move || {
        for chunk in &lrx.iter().chunks(num_cpus::get() * 10) {
            POOL.scoped(|s| {
                for data in chunk {
                    s.execute(|| {
                        if let Ok((_, des)) = ORB.with(|orb| match data.data {
                            Either::Left(img) => orb.borrow_mut().detect_image(img),
                            Either::Right(bytes) => orb.borrow_mut().detect_bytes(&bytes),
                        }) {
                            tx.send(ProcessableImage {
                                path: data.path,
                                hash: data.hash,
                                descriptors: des,
                            })
                            .unwrap();
                        } else {
                            pb.set_message(format!("计算特征点失败: {}", data.path));
                            pb.inc(1);
                        }
                    });
                }
            });
        }
    });
    (t, rx)
}

pub fn task_add(
    lrx: Receiver<ProcessableImage>,
    pb: ProgressBar,
    db: Arc<IMDB>,
    min_keypoints: i32,
    overwrite: bool,
    replace: Option<(Regex, String)>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        while let Ok(data) = lrx.recv() {
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

async fn scan_directory(
    path: PathBuf,
    tx: Sender<ImageData>,
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
        .for_each_concurrent(32, |entry| async {
            if let Ok(data) = tokio::fs::read(&entry).await {
                tx.send(ImageData { path: entry, data }).unwrap();
            }
        })
        .await;

    Ok(())
}

async fn scan_tar(
    path: PathBuf,
    tx: Sender<ImageData>,
    re_suf: Regex,
    pb: ProgressBar,
) -> Result<()> {
    let file = File::open(path).await?;
    let mut archive = Archive::new(file);
    let mut entries = archive.entries()?;

    pb.set_style(pb_style_speed());

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

        tx.send(ImageData { path, data }).unwrap();
    }
    Ok(())
}
