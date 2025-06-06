use std::convert::Infallible;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use tokio::task::block_in_place;

use crate::IMDBBuilder;
use crate::cli::SubCommandExtend;
use crate::config::{Opts, OrbOptions, SearchOptions};
use crate::ivf::IvfHnsw;
use crate::orb::ORBDetector;

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
    /// 默认索引文件名
    #[arg(short = 'I', long, value_name = "NAME", default_value = "index")]
    pub index_name: String,
}

impl SubCommandExtend for SearchCommand {
    async fn run(&self, opts: &Opts) -> anyhow::Result<()> {
        let mut orb = ORBDetector::create(self.orb.clone());
        let (_, _, des) = block_in_place(|| orb.detect_file(&self.image))?;

        let db = IMDBBuilder::new(opts.conf_dir.clone())
            .score_type(self.search.score_type)
            .open()
            .await?;

        let index = Arc::new(IvfHnsw::open_disk(&opts.conf_dir)?);

        let SearchOptions { k, distance, count, nprobe, .. } = self.search;
        let result = db.search(index, des, k, distance, count, nprobe).await?;

        print_result(&result, self)
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
