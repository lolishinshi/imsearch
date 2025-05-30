use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::LazyLock;

use clap::{Parser, Subcommand, ValueEnum};
use directories::ProjectDirs;
use opencv::imgproc::InterpolationFlags;

use crate::cli::*;

static CONF_DIR: LazyLock<ConfDir> = LazyLock::new(|| {
    let proj_dirs = ProjectDirs::from("", "aloxaf", "imsearch").expect("failed to get project dir");
    ConfDir { path: proj_dirs.config_dir().to_path_buf(), default: "index".to_string() }
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
    /// 图片最大尺寸，如果宽高**均**超过这个尺寸，则等比缩放
    #[arg(short = 'S', long, value_name = "HEIGHTxWIDHT", value_parser = parse_size, verbatim_doc_comment, default_value = "1080x768")]
    pub max_size: (i32, i32),
    /// 图片最大长宽比例，超过这个比例的图片，会按比例增加特征点数量
    #[arg(short = 'A', long, value_name = "RATIO", default_value_t = 5.)]
    pub max_aspect_ratio: f32,
    /// 最大特征点数量
    #[arg(long, default_value_t = 1000)]
    pub max_features: u32,
}

#[derive(Parser, Debug, Clone)]
pub struct SearchOptions {
    /// 不使用 mmap 模式加载索引，而是一次性全部加载到内存
    #[arg(long)]
    pub no_mmap: bool,
    /// 两个相似向量的允许的最大距离，范围从 0 到 255
    #[arg(long, value_name = "N", default_value_t = 64, value_parser = clap::value_parser!(u32).range(0..=255))]
    pub distance: u32,
    /// 显示的结果数量
    #[arg(long, value_name = "COUNT", default_value_t = 10)]
    pub count: usize,
    /// 每个查询描述符找到的最佳匹配数量
    #[arg(short, value_name = "K", default_value_t = 3)]
    pub k: usize,
    /// 搜索的倒排列表数量
    #[arg(long, default_value = "3")]
    pub nprobe: usize,
    /// HNSW 搜索时每次访问的节点数量
    #[arg(long, default_value = "16")]
    pub ef_search: usize,
    /// 评分方式
    #[arg(long, value_enum, default_value_t = ScoreType::Wilson)]
    pub score_type: ScoreType,
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
    /// 添加图片特征点到数据库
    Add(AddCommand),
    /// 从数据库中搜索图片
    Search(SearchCommand),
    /// 启动 HTTP 搜索服务
    Server(ServerCommand),
    /// 使用已添加的特征点构建索引
    Build(BuildCommand),
    /// 清理数据库中的特征点，主要作用为减小数据库体积
    Clean(CleanCommand),
    /// 训练索引
    Train(TrainCommand),
}

#[derive(Debug, Clone)]
pub struct ConfDir {
    path: PathBuf,
    default: String,
}

impl ConfDir {
    pub fn path(&self) -> &Path {
        self.path.as_path()
    }

    pub fn set_default(&mut self, default: String) {
        self.default = default;
    }

    /// 返回数据库文件的路径
    pub fn database(&self) -> PathBuf {
        self.path.join("imsearch.db")
    }

    /// 返回 phash 索引文件路径
    pub fn index_phash(&self) -> PathBuf {
        self.path.join("index.phash")
    }

    /// 返回索引文件的路径
    pub fn index(&self) -> PathBuf {
        self.path.join(&self.default)
    }

    /// 返回下一个子索引文件的路径
    pub fn next_sub_index(&self) -> PathBuf {
        for i in 1.. {
            let path = self.path.join(format!("index.{}", i));
            if !path.exists() {
                return path;
            }
        }
        unreachable!()
    }

    /// 返回所有子索引文件的路径
    pub fn all_sub_index(&self) -> Vec<PathBuf> {
        let mut paths = vec![];
        for i in 1.. {
            let path = self.path.join(format!("index.{}", i));
            if !path.exists() {
                break;
            }
            paths.push(path);
        }
        paths
    }

    /// 返回索引模板文件的路径
    pub fn template_index(&self) -> PathBuf {
        self.path.join("index.template")
    }

    /// 返回 ondisk ivf 文件的路径
    pub fn ondisk_ivf(&self) -> PathBuf {
        self.path.join("index.ivfdata")
    }

    /// 返回 ondisk ivf 文件的临时路径
    pub fn ondisk_ivf_tmp(&self) -> PathBuf {
        self.path.join("index.ivfdata.tmp")
    }
}

impl FromStr for ConfDir {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self { path: PathBuf::from(s), default: "index".to_string() })
    }
}

fn parse_size(s: &str) -> anyhow::Result<(i32, i32)> {
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() != 2 {
        return Err(anyhow::anyhow!("无效的尺寸: {}", s));
    }
    Ok((parts[0].parse()?, parts[1].parse()?))
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

#[derive(ValueEnum, Debug, Clone, Copy)]
pub enum ScoreType {
    /// 威尔逊评分
    Wilson,
    /// 计数评分
    Count,
}
