use std::collections::HashMap;
use std::fs::File;
use std::sync::Mutex;
use std::time::Instant;

use clap::Parser;
use hnsw_rs::prelude::*;
use imsearch::utils::{ImageHash, pb_style};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator};
use log::info;
use rayon::prelude::*;
use regex::Regex;
use walkdir::WalkDir;

pub struct Dist64BitHamming;

impl Distance<u64> for Dist64BitHamming {
    fn eval(&self, va: &[u64], vb: &[u64]) -> f32 {
        unsafe {
            let a = va.get_unchecked(0);
            let b = vb.get_unchecked(0);
            (a ^ b).count_ones() as f32 / 64.
        }
    }
}

/// phash 的去重效果分析工具
#[derive(Parser)]
pub struct Args {
    /// 包含图片的目录路径
    path: Vec<String>,
    /// 扫描的文件后缀名，多个后缀用逗号分隔
    #[arg(short, long, default_value = "jpg,png")]
    pub suffix: String,
    /// 判断重复的相似度距离
    #[arg(short, long, default_value_t = 4)]
    threshold: u8,
    /// 将重复图片列表保存到文件
    #[arg(short, long)]
    output: Option<String>,
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    let re_suf = format!("(?i)({})", args.suffix.replace(',', "|"));
    let re_suf = Regex::new(&re_suf).expect("failed to build regex");

    let images = args
        .path
        .iter()
        .flat_map(|path| {
            info!("开始扫描目录: {}", path);
            let pb = ProgressBar::new(0);
            pb.set_style(pb_style());
            WalkDir::new(path).into_iter().progress_with(pb).filter_map(|entry| {
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
        })
        .take(10000)
        .collect::<Vec<_>>();
    info!("扫描完成，共 {} 张图片", images.len());

    info!("对所有图片计算 blake3 和 phash……");
    let hashes = images
        .into_par_iter()
        .progress_with_style(pb_style())
        .filter_map(|image| {
            let b3hash = ImageHash::Blake3.hash_file(&image).ok()?;
            let phash = ImageHash::Phash.hash_file(&image).ok()?;
            let mut b = [0u8; 8];
            b.copy_from_slice(&phash);
            let phash = u64::from_le_bytes(b);
            Some((b3hash, (phash, image)))
        })
        .collect::<HashMap<_, _>>();

    info!("blake3 去重完成，共 {} 不重复图片", hashes.len());
    let (hashes, images): (Vec<_>, Vec<_>) = hashes.into_values().unzip();

    info!("开始进行 phash 去重……");
    let now = Instant::now();
    let index = Hnsw::<u64, Dist64BitHamming>::new(15, hashes.len(), 16, 40, Dist64BitHamming);
    let duplicates = Mutex::new(HashMap::new());
    // NOTE: 此处由于使用了 par_iter，结果会存在一定随机性
    hashes.par_iter().progress_with_style(pb_style()).enumerate().for_each(|(i, &hash)| {
        let result = index.search(&[hash], 1, 16);
        let result =
            result.into_iter().filter(|n| n.distance * 64. <= args.threshold as f32).next();
        if let Some(n) = result {
            duplicates.lock().unwrap().entry(n.d_id).or_insert(vec![]).push(i as u64);
        } else {
            index.insert((&[hash], i));
        }
    });
    let elapsed = now.elapsed();
    info!("phash 去重完成，耗时 {:?}", elapsed);
    let duplicates = duplicates.into_inner().unwrap();
    let total = duplicates.iter().map(|(_, value)| value.len()).sum::<usize>();

    info!("phash 去重完成，共 {} 组重复图片，{} 张重复图片", duplicates.len(), total);

    if let Some(output) = args.output {
        let mut file = File::create(output).unwrap();
        let duplicates = duplicates
            .iter()
            .map(|(key, value)| {
                let k = &images[*key as usize];
                let v = value.iter().map(|i| &images[*i as usize]).collect::<Vec<_>>();
                (k, v)
            })
            .collect::<HashMap<_, _>>();
        serde_json::to_writer_pretty(&mut file, &duplicates).unwrap();
    }
}
