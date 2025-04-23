use crate::cmd::SubCommandExtend;
use crate::config::{Opts, OutputFormat};
use crate::index::FaissSearchParams;
use crate::slam3_orb::Slam3ORB;
use crate::IMDB;
use crate::ORB;
use anyhow::Result;
use clap::Parser;
use indicatif::ParallelProgressIterator;
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use rayon::prelude::*;
use regex::Regex;
use std::path::PathBuf;
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
    fn run(&self, opts: &Opts) -> anyhow::Result<()> {
        let re = Regex::new(&self.suffix.replace(',', "|")).expect("failed to build regex");
        let db = IMDB::new(opts.conf_dir.clone(), false)?;

        // 收集所有符合条件的文件路径
        info!("开始扫描目录: {}", self.path);
        let entries: Vec<PathBuf> = WalkDir::new(&self.path)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.into_path())
            .filter(|path| {
                path.extension().map(|s| re.is_match(&*s.to_string_lossy())) == Some(true)
            })
            .collect();

        // 创建进度条
        let pb = ProgressBar::new(entries.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("#>-")
        );
        pb.set_message("处理图片中...");

        // 使用进度条处理图片
        entries
            .into_par_iter()
            .progress_with(pb.clone())
            .for_each(|entry| {
                let result =
                    ORB.with(|orb| db.add_image(entry.to_string_lossy(), &mut *orb.borrow_mut()));

                let status_msg = match &result {
                    Ok(add) => match add {
                        true => format!("Add {}", entry.display()),
                        false => format!("Update {}", entry.display()),
                    },
                    Err(e) => format!("Skip {}", e),
                };

                // 更新进度条消息
                pb.set_message(status_msg.clone());

                if let Err(e) = &result {
                    pb.println(format!("{}: {}", entry.display(), e));
                }
            });

        // 完成后的消息
        pb.finish_with_message("图片处理完成！");
        Ok(())
    }
}

impl SubCommandExtend for SearchImage {
    fn run(&self, opts: &Opts) -> anyhow::Result<()> {
        let db = IMDB::new(opts.conf_dir.clone(), true)?;
        let mut orb = Slam3ORB::from(opts);

        let index = db.get_index(opts.mmap, opts.strategy);
        let params = FaissSearchParams {
            nprobe: self.nprobe,
            max_codes: self.max_codes,
        };
        let mut result = db.search(
            &index,
            &self.image,
            &mut orb,
            opts.knn_k,
            opts.distance,
            params,
        )?;

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
