use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::Result;
use blake3::Hash;
use indicatif::ProgressStyle;
use opencv::core::*;
use opencv::{features2d, highgui, imgcodecs, imgproc};

use crate::orb::Slam3ORB;

pub fn detect_and_compute(
    orb: &mut Slam3ORB,
    image: &impl ToInputArray,
) -> Result<(Vector<KeyPoint>, Mat)> {
    let mask = Mat::default();
    let lap = Vector::<i32>::from(vec![0, 0]);
    let mut kps = Vector::<KeyPoint>::new();
    let mut des = Mat::default();
    orb.detect_and_compute(image, &mask, &mut kps, &mut des, &lap)?;
    Ok((kps, des))
}

pub fn imdecode(buf: &[u8], max_width: u32) -> Result<Mat> {
    let mat = Mat::from_slice(buf)?;
    let mut img = imgcodecs::imdecode(&mat, imgcodecs::IMREAD_GRAYSCALE)?;
    if img.cols() > max_width as i32 {
        img = adjust_image_size(&img, max_width as i32)?;
    }
    Ok(img)
}

pub fn imread<S: AsRef<str>>(filename: S, max_width: u32) -> Result<Mat> {
    let mut img = imgcodecs::imread(filename.as_ref(), imgcodecs::IMREAD_GRAYSCALE)?;
    if img.cols() > max_width as i32 {
        img = adjust_image_size(&img, max_width as i32)?;
    }
    Ok(img)
}

pub fn imshow(winname: &str, mat: &impl ToInputArray) -> Result<()> {
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

pub fn imwrite(filename: &str, img: &impl ToInputArray) -> Result<bool> {
    let flags = Vector::<i32>::new();
    Ok(imgcodecs::imwrite(filename, img, &flags)?)
}

// 调整图片的最大宽度，同时保持长宽比
pub fn adjust_image_size(img: &Mat, width: i32) -> Result<Mat> {
    if img.cols() <= width {
        return Ok(img.clone());
    }
    let scale = width as f64 / img.cols() as f64;
    let mut output = Mat::default();
    imgproc::resize(
        img,
        &mut output,
        Size::default(),
        scale,
        scale,
        imgproc::InterpolationFlags::INTER_AREA as i32,
    )?;
    Ok(output)
}

pub fn draw_keypoints(image: &impl ToInputArray, keypoints: &Vector<KeyPoint>) -> Result<Mat> {
    let mut output = Mat::default();
    features2d::draw_keypoints(
        image,
        keypoints,
        &mut output,
        Scalar::all(-1.0),
        features2d::DrawMatchesFlags::DEFAULT,
    )?;
    Ok(output)
}

pub fn draw_matches_knn(
    img1: &impl ToInputArray,
    keypoints1: &Vector<KeyPoint>,
    img2: &impl ToInputArray,
    keypoints2: &Vector<KeyPoint>,
    matches1to2: &Vector<Vector<DMatch>>,
    matches_mask: &Vector<Vector<i8>>,
) -> Result<Mat> {
    let mut output = Mat::default();
    features2d::draw_matches_knn(
        img1,
        keypoints1,
        img2,
        keypoints2,
        matches1to2,
        &mut output,
        Scalar::from((0., 255., 0.)),
        Scalar::from((255., 0., 0.)),
        matches_mask,
        features2d::DrawMatchesFlags::DEFAULT,
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

pub fn hash_file(path: impl AsRef<Path>) -> Result<Hash> {
    let mut file = File::open(path)?;
    let mut data = vec![];
    file.read_to_end(&mut data)?;
    Ok(blake3::hash(&data))
}

pub fn pb_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
        )
        .unwrap()
        .progress_chars("#>-")
}
