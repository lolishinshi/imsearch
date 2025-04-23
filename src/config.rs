use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::cmd::*;
use crate::imdb::SearchStrategy;
use crate::slam3_orb::{InterpolationFlags, Slam3ORB};
use clap::{Parser, Subcommand, ValueEnum};
use directories::ProjectDirs;
use once_cell::sync::Lazy;
use opencv::{core, features2d, flann};

static CONF_DIR: Lazy<ConfDir> = Lazy::new(|| {
    let proj_dirs = ProjectDirs::from("", "aloxaf", "imsearch").expect("failed to get project dir");
    ConfDir(proj_dirs.config_dir().to_path_buf())
});

fn default_config_dir() -> &'static str {
    CONF_DIR.path().to_str().unwrap()
}

#[derive(Parser, Debug, Clone)]
#[command(name = "imsearch")]
pub struct Opts {
    /// imsearch 配置文件目录
    #[arg(short, long, default_value = default_config_dir())]
    pub conf_dir: ConfDir,

    /// ORB 特征点最大保留数量
    #[arg(short = 'n', value_name = "N", long, default_value = "500")]
    pub orb_nfeatures: u32,
    /// ORB 特征金字塔缩放因子
    #[arg(long, value_name = "SCALE", default_value = "1.2")]
    pub orb_scale_factor: f32,
    /// ORB 特征金字塔层数
    #[arg(long, value_name = "N", default_value = "8")]
    pub orb_nlevels: u32,
    /// ORB 特征点金字塔缩放插值方式
    #[arg(long, value_name = "FLAG", default_value = "Area")]
    pub orb_interpolation: InterpolationFlags,
    /// ORB FAST 角点检测器初始阈值
    #[arg(long, value_name = "THRESHOLD", default_value = "20")]
    pub orb_ini_th_fast: u32,
    /// ORB FAST 角点检测器最小阈值
    #[arg(long, value_name = "THRESHOLD", default_value = "7")]
    pub orb_min_th_fast: u32,
    /// ORB 特征点是否不需要方向信息
    #[arg(long)]
    pub orb_not_oriented: bool,

    /// 使用 mmap 模式加载索引，而不是一次性全部加载到内存
    #[arg(long)]
    pub mmap: bool,

    /// 构建索引时的批次大小
    #[arg(long, value_name = "SIZE", default_value = "5000000")]
    pub batch_size: usize,
    /// 两个相似向量的允许的最大距离，范围从 0 到 255
    #[arg(long, value_name = "N", default_value = "64")]
    pub distance: u32,

    /// 显示的结果数量
    #[arg(long, value_name = "COUNT", default_value = "10")]
    pub output_count: usize,
    /// 输出格式
    #[arg(long, value_name = "FORMAT", default_value = "table")]
    pub output_format: OutputFormat,
    /// 每个查询描述符找到的最佳匹配数量
    #[arg(long, value_name = "K", default_value = "3")]
    pub knn_k: usize,
    /// 搜索策略
    #[arg(long, value_name = "STRATEGY", default_value = "heap")]
    pub strategy: SearchStrategy,

    #[command(subcommand)]
    pub subcmd: SubCommand,
}

#[derive(Subcommand, Debug, Clone)]
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
