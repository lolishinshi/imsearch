use crate::cmd::SubCommandExtend;
use crate::config::{Opts, OutputFormat};
use crate::index::FaissSearchParams;
use crate::slam3_orb::Slam3ORB;
use crate::IMDB;
use crate::ORB;
use anyhow::Result;
use clap::Parser;
use rayon::prelude::*;
use regex::Regex;
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
        WalkDir::new(&self.path)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .par_bridge()
            .for_each(|entry| {
                let entry = entry.into_path();
                if entry
                    .extension()
                    .map(|s| re.is_match(&*s.to_string_lossy()))
                    != Some(true)
                {
                    return;
                }

                let result =
                    ORB.with(|orb| db.add_image(entry.to_string_lossy(), &mut *orb.borrow_mut()));
                match result {
                    Ok(add) => match add {
                        true => println!("[OK] Add {}", entry.display()),
                        false => println!("[OK] Update {}", entry.display()),
                    },
                    Err(e) => eprintln!("[ERR] {}: {}\n{}", entry.display(), e, e.backtrace()),
                }
            });
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
