use std::path::PathBuf;
use std::str::FromStr;

use crate::slam3_orb::Slam3ORB;
use crate::ImageDb;
use directories::ProjectDirs;
use once_cell::sync::Lazy;
use opencv::{core, features2d, flann};
use structopt::clap::AppSettings;
use structopt::StructOpt;

pub static OPTS: Lazy<Opts> = Lazy::new(Opts::from_args);
pub static CONF_DIR: Lazy<PathBuf> = Lazy::new(|| {
    let proj_dirs = ProjectDirs::from("", "aloxaf", "imsearch").expect("failed to get project dir");
    proj_dirs.config_dir().to_path_buf()
});
pub static THREAD_NUM: Lazy<usize> = Lazy::new(|| {
    std::env::var("RAYON_NUM_THREADS")
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(num_cpus::get())
});

fn default_config_dir() -> &'static str {
    CONF_DIR.to_str().unwrap()
}

#[derive(StructOpt)]
#[structopt(name = "imsearch", global_setting(AppSettings::ColoredHelp))]
pub struct Opts {
    /// Path to imsearch config
    #[structopt(short, long, default_value = default_config_dir())]
    pub conf_dir: String,

    /// The maximum number of features to retain
    #[structopt(short = "n", value_name = "N", long, default_value = "500")]
    pub orb_nfeatures: u32,
    /// Pyramid decimation ratio, greater than 1
    #[structopt(long, value_name = "SCALE", default_value = "1.2")]
    pub orb_scale_factor: f32,
    /// The number of pyramid levels
    #[structopt(long, value_name = "N", default_value = "8")]
    pub orb_nlevels: u32,
    /// Initial fast threshold
    #[structopt(long, value_name = "THRESHOLD", default_value = "20")]
    pub orb_ini_th_fast: u32,
    /// Minimum fast threshold
    #[structopt(long, value_name = "THRESHOLD", default_value = "7")]
    pub orb_min_th_fast: u32,

    /// The number of hash tables to use
    #[structopt(long, value_name = "NUMBER", default_value = "6")]
    pub lsh_table_number: u32,
    /// The length of the key in the hash tables
    #[structopt(long, value_name = "SIZE", default_value = "12")]
    pub lsh_key_size: u32,
    /// Number of levels to use in multi-probe (0 for standard LSH)
    #[structopt(long, value_name = "LEVEL", default_value = "1")]
    pub lsh_probe_level: u32,

    /// Specifies the maximum leafs to visit when searching for neighbours, -2 = auto, -1 = unlimit
    #[structopt(long, value_name = "CHECKS", default_value = "32")]
    pub search_checks: i32,
    /// Number of features to search per iteration
    #[structopt(long, value_name = "SIZE", default_value = "5000000")]
    pub batch_size: usize,
    /// Maximum distance allowed for match, from 0 ~ 255
    #[structopt(long, value_name = "N", default_value = "64")]
    pub distance: u32,

    /// How many results to show
    #[structopt(long, value_name = "COUNT", default_value = "10")]
    pub output_count: usize,
    /// Output format
    #[structopt(long, value_name = "FORMAT", default_value = "table", possible_values = &["table", "json"])]
    pub output_format: OutputFormat,
    /// Count of best matches found per each query descriptor
    #[structopt(long, value_name = "K", default_value = "3")]
    pub knn_k: usize,

    #[structopt(subcommand)]
    pub subcmd: SubCommand,
}

#[derive(StructOpt)]
pub enum SubCommand {
    /// Show all features point for an image
    ShowKeypoints(ShowKeypoints),
    /// Show matches between two image
    ShowMatches(ShowMatches),
    /// Add images to database
    AddImages(AddImages),
    /// Search image from database
    SearchImage(SearchImage),
    /// Start interactive REPL
    StartRepl(StartRepl),
}

#[derive(StructOpt)]
pub struct ShowKeypoints {
    /// Path to an image
    pub image: String,
    /// Optional output image
    pub output: Option<String>,
}

#[derive(StructOpt)]
pub struct ShowMatches {
    /// Path to image A
    pub image1: String,
    /// Path to image B
    pub image2: String,
    /// Optional output image
    pub output: Option<String>,
}

#[derive(StructOpt)]
pub struct AddImages {
    /// Path to an image or folder
    pub path: String,
    /// Scan image with these suffixes
    #[structopt(short, long, default_value = "jpg,png")]
    pub suffix: String,
}

#[derive(StructOpt)]
pub struct SearchImage {
    /// Path to the image to search
    pub image: String,
}

#[derive(StructOpt)]
pub struct StartRepl {
    /// Promot
    #[structopt(short, long, default_value = "")]
    pub prompt: String,
}

#[derive(StructOpt)]
pub enum OutputFormat {
    Json,
    Table,
}

impl FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "json" => Ok(Self::Json),
            "table" => Ok(Self::Table),
            _ => unreachable!(),
        }
    }
}

impl From<&Opts> for Slam3ORB {
    fn from(opts: &Opts) -> Self {
        Self::create(
            opts.orb_nfeatures as i32,
            opts.orb_scale_factor,
            opts.orb_nlevels as i32,
            opts.orb_ini_th_fast as i32,
            opts.orb_min_th_fast as i32,
        )
        .expect("failed to build Slam3Orb")
    }
}

impl From<&Opts> for features2d::FlannBasedMatcher {
    fn from(opts: &Opts) -> Self {
        let index_params = core::Ptr::new(flann::IndexParams::from(
            flann::LshIndexParams::new(
                opts.lsh_table_number as i32,
                opts.lsh_key_size as i32,
                opts.lsh_probe_level as i32,
            )
            .expect("failed to build LshIndexParams"),
        ));
        let search_params = core::Ptr::new(
            flann::SearchParams::new_1(opts.search_checks, 0.0, true)
                .expect("failed to build SearchParams"),
        );
        features2d::FlannBasedMatcher::new(&index_params, &search_params)
            .expect("failed to build FlannBasedMatcher")
    }
}

impl From<&Opts> for ImageDb {
    fn from(opts: &Opts) -> Self {
        Self::new(&opts.conf_dir).expect("failed to create ImageDb")
    }
}
