use std::convert::Infallible;
use std::str::FromStr;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use log::debug;
use tokio::task::block_in_place;

use crate::cli::SubCommandExtend;
use crate::config::{Opts, OrbOptions, SearchOptions};
use crate::faiss::{FaissSearchParams, get_faiss_stats};
use crate::orb::Slam3ORB;
use crate::{IMDBBuilder, utils};

#[derive(Parser, Debug, Clone)]
pub struct SearchCommand {
    #[command(flatten)]
    pub orb: OrbOptions,
    #[command(flatten)]
    pub search: SearchOptions,
    /// 被搜索的图片路径
    pub image: String,
    /// 输出格式
    #[arg(long, value_name = "FORMAT", default_value = "table")]
    pub output_format: OutputFormat,
}

impl SubCommandExtend for SearchCommand {
    async fn run(&self, opts: &Opts) -> anyhow::Result<()> {
        let mut orb = Slam3ORB::from(&self.orb);
        let (_, des) = block_in_place(|| {
            utils::imread(&self.image, self.orb.img_max_width)
                .and_then(|image| utils::detect_and_compute(&mut orb, &image))
        })?;

        let db = IMDBBuilder::new(opts.conf_dir.clone()).mmap(!self.search.no_mmap).open().await?;
        let index = db.get_index();
        let params =
            FaissSearchParams { nprobe: self.search.nprobe, max_codes: self.search.max_codes };

        let result = db
            .search(&index, &[des], self.search.k, self.search.distance, self.search.count, params)
            .await?;

        let stats = get_faiss_stats();

        debug!("ndis             : {}", stats.nq);
        debug!("nprobe           : {}", stats.nlist);
        debug!("nheap_updates    : {}", stats.nheap_updates);
        debug!("quantization_time: {:.2}ms", stats.quantization_time);
        debug!("search_time      : {:.2}ms", stats.search_time);

        print_result(&result[0], self)
    }
}

fn print_result(result: &[(f32, String)], opts: &SearchCommand) -> Result<()> {
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

#[derive(ValueEnum, Debug, Clone)]
pub enum OutputFormat {
    Json,
    Table,
}

impl FromStr for OutputFormat {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "json" => Ok(Self::Json),
            "table" => Ok(Self::Table),
            _ => unreachable!(),
        }
    }
}
