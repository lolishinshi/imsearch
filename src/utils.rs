use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use crate::slam3_orb::Slam3ORB;
use anyhow::Result;
use blake3::Hash;
use dashmap::DashMap;
use opencv::features2d;
use opencv::highgui;
use opencv::imgcodecs;
use opencv::prelude::*;
use opencv::types;
use opencv::{core, imgproc};

pub fn detect_and_compute(
    orb: &mut Slam3ORB,
    image: &dyn core::ToInputArray,
) -> Result<(types::VectorOfKeyPoint, Mat)> {
    let mask = Mat::default();
    let lap = types::VectorOfi32::from(vec![0, 0]);
    let mut kps = types::VectorOfKeyPoint::new();
    let mut des = Mat::default();
    orb.detect_and_compute(image, &mask, &mut kps, &mut des, &lap)?;
    Ok((kps, des))
}

pub fn imread<S: AsRef<str>>(filename: S) -> Result<Mat> {
    let mut img = imgcodecs::imread(filename.as_ref(), imgcodecs::IMREAD_GRAYSCALE)?;
    if img.cols() > 1920 || img.rows() > 1080 {
        img = adjust_image_size(&img, 1920, 1080)?;
    }
    Ok(img)
}

pub fn imshow(winname: &str, mat: &dyn core::ToInputArray) -> Result<()> {
    highgui::imshow(winname, mat)?;
    while highgui::get_window_property(
        winname,
        highgui::WindowPropertyFlags::WND_PROP_FULLSCREEN as i32,
    )? >= 0.0
    {
        highgui::wait_key(50)?;
    }
    Ok(())
}

pub fn imwrite(filename: &str, img: &dyn core::ToInputArray) -> Result<bool> {
    let flags = types::VectorOfi32::new();
    Ok(imgcodecs::imwrite(filename, img, &flags)?)
}

// TODO: 对于长图，应该要增加特征点数量
pub fn adjust_image_size(img: &Mat, width: i32, height: i32) -> Result<Mat> {
    if img.rows() <= height || img.cols() <= width {
        return Ok(img.clone());
    }
    let (ow, oh) = (img.cols() as f64, img.rows() as f64);
    let scale = (height as f64 / oh).min(width as f64 / ow);
    let mut output = Mat::default();
    imgproc::resize(
        img,
        &mut output,
        core::Size::default(),
        scale,
        scale,
        imgproc::InterpolationFlags::INTER_AREA as i32,
    )?;
    Ok(output)
}

pub fn draw_keypoints(
    image: &dyn core::ToInputArray,
    keypoints: &types::VectorOfKeyPoint,
) -> Result<Mat> {
    let mut output = core::Mat::default();
    features2d::draw_keypoints(
        image,
        keypoints,
        &mut output,
        core::Scalar::all(-1.0),
        features2d::DrawMatchesFlags::DEFAULT,
    )?;
    Ok(output)
}

pub fn draw_matches_knn(
    img1: &dyn core::ToInputArray,
    keypoints1: &types::VectorOfKeyPoint,
    img2: &dyn core::ToInputArray,
    keypoints2: &types::VectorOfKeyPoint,
    matches1to2: &types::VectorOfVectorOfDMatch,
    matches_mask: &types::VectorOfVectorOfi8,
) -> Result<Mat> {
    let mut output = core::Mat::default();
    features2d::draw_matches_knn(
        img1,
        keypoints1,
        img2,
        keypoints2,
        matches1to2,
        &mut output,
        core::Scalar::from((0., 255., 0.)),
        core::Scalar::from((255., 0., 0.)),
        matches_mask,
        features2d::DrawMatchesFlags::DEFAULT,
    )?;
    Ok(output)
}

#[derive(Debug, Default)]
pub struct TimeMeasure(pub DashMap<String, Duration>);

impl TimeMeasure {
    pub fn new() -> Self {
        Self(DashMap::new())
    }

    pub fn measure<F, R>(&self, key: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let r = f();
        *self.0.entry(key.to_owned()).or_insert(Duration::default()) += Instant::now() - start;
        r
    }
}

pub fn read_line(prompt: &str) -> Result<String> {
    print!("{}", prompt);
    std::io::stdout().flush()?;
    let v = std::io::stdin()
        .bytes()
        .take_while(|c| c.as_ref().ok() != Some(&b'\n'))
        .collect::<Result<Vec<_>, _>>()?;
    if v.is_empty() {
        anyhow::bail!("EOF");
    }
    Ok(String::from_utf8(v)?.trim().to_owned())
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

pub fn hash_file<P: AsRef<Path>>(path: P) -> Result<Hash> {
    let mut file = File::open(path)?;
    let mut data = vec![];
    file.read_to_end(&mut data)?;
    Ok(blake3::hash(&data))
}
