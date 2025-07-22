use std::fs::File;
use std::io::Read;

use anyhow::Result;
use axum_typed_multipart::TryFromField;
use clap::ValueEnum;
use indicatif::ProgressStyle;
use ndarray::{Array2, ArrayView2};
use opencv::core::*;
use opencv::{imgcodecs, imgproc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::dhash::d_hash;
use crate::orb::Slam3ORB;

pub fn detect_and_compute(
    orb: &mut Slam3ORB,
    image: &impl ToInputArray,
) -> opencv::Result<(Vec<KeyPoint>, Array2<u8>)> {
    let mask = Mat::default();
    let mut kps = Vector::<KeyPoint>::new();
    let mut des = Mat::default();
    orb.detect_and_compute(image, &mask, &mut kps, &mut des)?;
    let kps = kps.to_vec();
    let des = ArrayView2::from_shape((kps.len(), 32), des.data_bytes()?).unwrap();
    Ok((kps, des.to_owned()))
}

pub fn imdecode(buf: &[u8], (height, width): (i32, i32)) -> opencv::Result<Mat> {
    let mat = Mat::from_slice(buf)?;
    let mut img = imgcodecs::imdecode(&mat, imgcodecs::IMREAD_GRAYSCALE)?;
    if img.cols() > width && img.rows() > height {
        img = adjust_image_size(img, (height, width))?;
    }
    Ok(img)
}

pub fn imread<S: AsRef<str>>(filename: S, (height, width): (i32, i32)) -> opencv::Result<Mat> {
    let mut img = imgcodecs::imread(filename.as_ref(), imgcodecs::IMREAD_GRAYSCALE)?;
    if img.cols() > width && img.rows() > height {
        img = adjust_image_size(img, (height, width))?;
    }
    Ok(img)
}

// 在长宽比例中，选择最大的进行缩放
pub fn adjust_image_size(img: Mat, (height, width): (i32, i32)) -> opencv::Result<Mat> {
    let scale = (width as f64 / img.cols() as f64).max(height as f64 / img.rows() as f64);
    if scale >= 1. {
        return Ok(img);
    }
    let mut output = Mat::default();
    imgproc::resize(
        &img,
        &mut output,
        Size::default(),
        scale,
        scale,
        imgproc::InterpolationFlags::INTER_AREA as i32,
    )?;
    Ok(output)
}

/// 威尔逊得分
/// 基于：https://www.jianshu.com/p/4d2b45918958
pub fn wilson_score(scores: &[f32]) -> f32 {
    let count = scores.len() as f32;
    if count == 0. {
        return 0.;
    }
    let mean = scores.iter().sum::<f32>() / count;
    let var = scores.iter().map(|&a| (mean - a).powi(2)).sum::<f32>() / count;
    // 98% 置信度
    let z = 2.326f32;

    (mean + z.powi(2) / (2. * count) - ((z / (2. * count)) * (4. * count * var + z.powi(2)).sqrt()))
        / (1. + z.powi(2) / count)
}

pub fn pb_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
        )
        .unwrap()
        .progress_chars("#>-")
}

pub fn pb_style_speed() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) {msg}",
        )
        .unwrap()
        .progress_chars("#>-")
}

#[derive(
    Debug, Clone, Copy, Eq, PartialEq, ValueEnum, ToSchema, TryFromField, Serialize, Deserialize,
)]
pub enum ImageHash {
    /// 使用 blake3 哈希算法，长度 32 字节
    #[schema(rename = "blake3")]
    Blake3,
    /// 使用 dhash 哈希算法，长度 8 字节
    #[schema(rename = "dhash")]
    Dhash,
}

impl ImageHash {
    /// 对一个图片文件进行哈希，返回哈希值
    pub fn hash_file(&self, path: &str) -> Result<Vec<u8>> {
        match self {
            Self::Blake3 => {
                let mut file = File::open(path)?;
                let mut data = vec![];
                file.read_to_end(&mut data)?;
                Ok(blake3::hash(&data).as_bytes().to_vec())
            }
            Self::Dhash => {
                let img = imgcodecs::imread(path, imgcodecs::IMREAD_GRAYSCALE)?;
                let hash = d_hash(&img)?;
                Ok(hash.to_vec())
            }
        }
    }

    /// 对一个图片的字节序列进行哈希，返回解码后的图片和哈希值
    pub fn hash_bytes(&self, data: &[u8]) -> Result<(Option<Mat>, Vec<u8>)> {
        match self {
            Self::Blake3 => Ok((None, blake3::hash(data).as_bytes().to_vec())),
            Self::Dhash => {
                let mat = Mat::from_slice(data)?;
                let img = imgcodecs::imdecode(&mat, imgcodecs::IMREAD_GRAYSCALE)?;
                let hash = d_hash(&img)?;
                Ok((Some(img), hash.to_vec()))
            }
        }
    }
}

impl Default for ImageHash {
    fn default() -> Self {
        Self::Blake3
    }
}
