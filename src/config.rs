use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::cmd::*;
use crate::slam3_orb::{InterpolationFlags, Slam3ORB};
use directories::ProjectDirs;
use once_cell::sync::Lazy;
use opencv::{core, features2d, flann};
use structopt::clap::AppSettings;
use structopt::StructOpt;

static CONF_DIR: Lazy<ConfDir> = Lazy::new(|| {
    let proj_dirs = ProjectDirs::from("", "aloxaf", "imsearch").expect("failed to get project dir");
    ConfDir(proj_dirs.config_dir().to_path_buf())
});

fn default_config_dir() -> &'static str {
    CONF_DIR.path().to_str().unwrap()
}

#[derive(StructOpt, Debug, Clone)]
#[structopt(name = "imsearch", global_setting(AppSettings::ColoredHelp))]
pub struct Opts {
    /// Path to imsearch config
    #[structopt(short, long, default_value = default_config_dir())]
    pub conf_dir: ConfDir,

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
    /// Interpolation algorithm
    #[structopt(long, value_name = "FLAG", default_value = "Area")]
    pub orb_interpolation: InterpolationFlags,
    /// Record orientation info
    #[structopt(long)]
    pub orb_not_oriented: bool,

    /// Use mmap instead of read whole index to memory
    #[structopt(long)]
    pub mmap: bool,

    /// Number of features to search per iteration
    #[structopt(long, value_name = "SIZE", default_value = "5000000")]
    pub batch_size: usize,
    /// Maximum distance allowed for match, from 0 ~ 128
    #[structopt(long, value_name = "N", default_value = "64")]
    pub distance: u32,

    /// How to calculate the score
    #[structopt(long, value_name = "TYPE", default_value = "wilson", possible_values = &["wilson", "count"])]
    pub score_type: ScoreType,
    /// How many results to show
    #[structopt(long, value_name = "COUNT", default_value = "10")]
    pub output_count: usize,
    /// Output format
    #[structopt(long, value_name = "FORMAT", default_value = "table", possible_values = &["table", "json"])]
    pub output_format: OutputFormat,
    /// Count of best matches found per each query descriptor
    #[structopt(long, value_name = "K", default_value = "3")]
    pub knn_k: usize,
    /// How many bucket to search
    #[structopt(long, value_name = "N", default_value = "3")]
    pub nprobe: usize,

    #[structopt(subcommand)]
    pub subcmd: SubCommand,
}

#[derive(StructOpt, Debug, Clone)]
pub enum SubCommand {
    /// Show all features point for an image
    ShowKeypoints(ShowKeypoints),
    /// Show matches between two image
    ShowMatches(ShowMatches),
    /// Add images to database
    AddImages(AddImages),
    /// Search image from database
    SearchImage(SearchImage),
    /// Start Web server
    StartServer(StartServer),
    /// Build index
    BuildIndex(BuildIndex),
    /// Clear indexed (and unindexed) features
    ClearCache(ClearCache),
    /// Mark a range of features as trained
    MarkAsIndexed(MarkAsIndexed),
    /// Export data for trainning
    ExportData(ExportData),
}

#[derive(StructOpt, Debug, Clone)]
pub enum OutputFormat {
    Json,
    Table,
}

#[derive(StructOpt, Debug, Clone)]
pub enum ScoreType {
    Wilson,
    Count,
}

impl FromStr for ScoreType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "wilson" => Ok(Self::Wilson),
            "count" => Ok(Self::Count),
            _ => unreachable!(),
        }
    }
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

impl From<&Opts> for Slam3ORB {
    fn from(opts: &Opts) -> Self {
        Self::create(
            opts.orb_nfeatures as i32,
            opts.orb_scale_factor,
            opts.orb_nlevels as i32,
            opts.orb_ini_th_fast as i32,
            opts.orb_min_th_fast as i32,
            opts.orb_interpolation,
            !opts.orb_not_oriented,
        )
        .expect("failed to build Slam3Orb")
    }
}

impl From<&Opts> for features2d::FlannBasedMatcher {
    fn from(_opts: &Opts) -> Self {
        let index_params = core::Ptr::new(flann::IndexParams::from(
            flann::LshIndexParams::new(6, 12, 1).expect("failed to build LshIndexParams"),
        ));
        let search_params = core::Ptr::new(
            flann::SearchParams::new_1(32, 0.0, true).expect("failed to build SearchParams"),
        );
        features2d::FlannBasedMatcher::new(&index_params, &search_params)
            .expect("failed to build FlannBasedMatcher")
    }
}

#[derive(Debug, Clone)]
pub struct ConfDir(PathBuf);

impl ConfDir {
    pub fn path(&self) -> &Path {
        self.0.as_path()
    }

    pub fn database(&self) -> PathBuf {
        self.0.join("database")
    }

    pub fn index(&self) -> PathBuf {
        self.0.join("index")
    }

    pub fn version(&self) -> PathBuf {
        self.0.join("version")
    }
}

impl FromStr for ConfDir {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(PathBuf::from(s)))
    }
}
