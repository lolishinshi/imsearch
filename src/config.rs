use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::LazyLock;

use clap::{Parser, Subcommand};
use directories::ProjectDirs;
use opencv::imgproc::InterpolationFlags;

use crate::cli::*;
use crate::slam3_orb::Slam3ORB;

static CONF_DIR: LazyLock<ConfDir> = LazyLock::new(|| {
    let proj_dirs = ProjectDirs::from("", "aloxaf", "imsearch").expect("failed to get project dir");
    ConfDir(proj_dirs.config_dir().to_path_buf())
});

fn default_config_dir() -> &'static str {
    CONF_DIR.path().to_str().unwrap()
}

#[derive(Parser, Debug, Clone)]
pub struct OrbOptions {
    /// ORB 特征点最大保留数量
    #[arg(short = 'n', value_name = "N", long, default_value_t = 500)]
    pub orb_nfeatures: u32,
    /// ORB 特征金字塔缩放因子
    #[arg(long, value_name = "SCALE", default_value_t = 1.2)]
    pub orb_scale_factor: f32,
    /// ORB 特征金字塔层数
    #[arg(long, value_name = "N", default_value_t = 8)]
    pub orb_nlevels: u32,
    /// ORB 特征点金字塔缩放插值方式
    #[arg(long, value_name = "FLAG", default_value = "area", value_parser = parse_interpolation)]
    pub orb_interpolation: InterpolationFlags,
    /// ORB FAST 角点检测器初始阈值
    #[arg(long, value_name = "THRESHOLD", default_value_t = 20)]
    pub orb_ini_th_fast: u32,
    /// ORB FAST 角点检测器最小阈值
    #[arg(long, value_name = "THRESHOLD", default_value_t = 7)]
    pub orb_min_th_fast: u32,
    /// ORB 特征点是否不需要方向信息
    #[arg(long)]
    pub orb_not_oriented: bool,
}

#[derive(Parser, Debug, Clone)]
pub struct SearchOptions {
    /// 不使用 mmap 模式加载索引，而是一次性全部加载到内存
    #[arg(long)]
    pub no_mmap: bool,
    /// 两个相似向量的允许的最大距离，范围从 0 到 255
    #[arg(long, value_name = "N", default_value_t = 64, value_parser = clap::value_parser!(u16).range(0..=255))]
    pub distance: u32,
    /// 显示的结果数量
    #[arg(long, value_name = "COUNT", default_value_t = 10)]
    pub count: usize,
    /// 每个查询描述符找到的最佳匹配数量
    #[arg(long, value_name = "K", default_value_t = 3)]
    pub k: usize,
    /// 搜索的倒排列表数量
    #[arg(long, default_value = "3")]
    pub nprobe: usize,
    /// 搜索的最大向量数量，0 表示不限制
    #[arg(long, default_value = "0")]
    pub max_codes: usize,
}

#[derive(Parser, Debug, Clone)]
#[command(name = "imsearch", version)]
pub struct Opts {
    #[command(subcommand)]
    pub subcmd: SubCommand,
    /// imsearch 配置文件目录
    #[arg(short, long, default_value = default_config_dir())]
    pub conf_dir: ConfDir,
}

#[derive(Subcommand, Debug, Clone)]
pub enum SubCommand {
    /// 展示一张图片上的所有特征点
    Show(ShowCommand),
    /// 展示两张图片之间的特征点匹配关系
    Match(MatchCommand),
    /// 添加图片特征点到数据库
    Add(AddCommand),
    /// 从数据库中搜索图片
    Search(SearchCommand),
    /// 启动 HTTP 搜索服务
    Server(ServerCommand),
    /// 使用已添加的特征点构建索引
    Build(BuildCOmmand),
    /// 清理数据库中的特征点，主要作用为减小数据库体积
    Clean(CleanCommand),
    /// 导出 npy 格式的特征点，供训练使用
    Export(ExportCommand),
    #[cfg(feature = "rocksdb")]
    /// 从 rocksdb 格式的旧数据库中更新为新的数据库格式
    UpdateDB(UpdateDBCommand),
}

impl From<&OrbOptions> for Slam3ORB {
    fn from(opts: &OrbOptions) -> Self {
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

#[derive(Debug, Clone)]
pub struct ConfDir(PathBuf);

impl ConfDir {
    pub fn path(&self) -> &Path {
        self.0.as_path()
    }

    /// 返回数据库文件的路径
    pub fn database(&self) -> PathBuf {
        self.0.join("imsearch.db")
    }

    /// 返回索引文件的路径
    pub fn index(&self) -> PathBuf {
        self.0.join("index")
    }

    /// 返回子索引文件的路径
    pub fn index_sub(&self) -> PathBuf {
        for i in 1.. {
            let path = self.0.join(format!("index.{}", i));
            if !path.exists() {
                return path;
            }
        }
        unreachable!()
    }

    pub fn index_sub_with(&self, i: usize) -> PathBuf {
        self.0.join(format!("index.{}", i))
    }

    /// 返回索引模板文件的路径
    pub fn index_template(&self) -> PathBuf {
        self.0.join("index.template")
    }

    /// 返回索引临时文件的路径
    pub fn index_tmp(&self) -> PathBuf {
        self.0.join("index.tmp")
    }
}

impl FromStr for ConfDir {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(PathBuf::from(s)))
    }
}

fn parse_interpolation(s: &str) -> Result<InterpolationFlags, String> {
    match s {
        "linear" => Ok(InterpolationFlags::INTER_LINEAR),
        "cubic" => Ok(InterpolationFlags::INTER_CUBIC),
        "area" => Ok(InterpolationFlags::INTER_AREA),
        "lanczos4" => Ok(InterpolationFlags::INTER_LANCZOS4),
        _ => Err(format!("无效的插值方式: {}", s)),
    }
}
