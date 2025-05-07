use std::sync::Arc;

use clap::Parser;
use indicatif::ProgressBar;
use log::info;
use opencv::core::MatTraitConst;
use regex::Regex;
use tokio::sync::mpsc::channel;
use tokio::task::{block_in_place, spawn_blocking};
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

        let pb = ProgressBar::new(entries.len() as u64).with_style(pb_style());

        // task1: 哈希计算
        // NOTE: 对于机械硬盘来说，多线程读取会降低速度，不应该使用多线程。
        //       对于 SSD 来说，它的读取速度远大于计算速度，没必要使用多线程。
        let (hash_tx, mut hash_rx) = channel(num_cpus::get() * 2);
        let task1_hash = tokio::spawn({
            let hash = self.hash;
            let hash_tx = hash_tx.clone();
            async move {
                for entry in entries {
                    let data = tokio::fs::read(&entry).await.unwrap();
                    let hash = block_in_place(|| hash.hash_bytes(&data)).unwrap();
                    hash_tx.send((entry, data, hash)).await.unwrap();
                }
            }
        });

        // task2: 检查已添加图片
        let (filter_tx, mut filter_rx) = channel(num_cpus::get() * 2);
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
                            pb.set_message(format!("跳过图片: {}", entry));
                        }
                        pb.inc(1);
                    } else {
                        filter_tx.send((entry, data, hash)).await.unwrap();
                    }
                }
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
        let task4_add = tokio::spawn({
            let pb = pb.clone();
            let min_keypoints = self.min_keypoints as i32;
            let overwrite = self.overwrite;
            async move {
                while let Some((entry, hash, des)) = feature_rx.recv().await {
                    if des.rows() <= min_keypoints {
                        pb.println(format!("特征点少于 {}: {}", min_keypoints, entry));
                        pb.inc(1);
                        continue;
                    }

                    let path = match &re_name {
                        Some(re) => {
                            let captures = re.captures(&entry);
                            if let Some(name) =
                                captures.and_then(|c| c.name("name").map(|m| m.as_str()))
                            {
                                name.to_string()
                            } else {
                                pb.println(format!("提取图片名失败: {}", entry));
                                pb.inc(1);
                                continue;
                            }
                        }
                        None => entry.clone(),
                    };

                    if db.check_hash(&hash).await.unwrap() {
                        if overwrite {
                            db.update_image_path(&hash, &path).await.unwrap();
                            pb.set_message(format!("更新图片路径: {}", path));
                        } else {
                            pb.set_message(format!("跳过图片: {}", path));
                        }
                    } else {
                        db.add_image(&path, &hash, des).await.unwrap();
                        pb.set_message(entry);
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
