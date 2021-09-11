use crate::slam3_orb::Slam3ORB;
use opencv::features2d;
use opencv::highgui;
use opencv::imgcodecs;
use opencv::prelude::*;
use opencv::types;
use opencv::{core, imgproc};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::time::{Duration, Instant};

pub fn detect_and_compute(
    orb: &mut Slam3ORB,
    image: &dyn core::ToInputArray,
) -> opencv::Result<(types::VectorOfKeyPoint, Mat)> {
    let mask = Mat::default();
    let lap = types::VectorOfi32::from(vec![0, 0]);
    let mut kps = types::VectorOfKeyPoint::new();
    let mut des = Mat::default();
    orb.detect_and_compute(image, &mask, &mut kps, &mut des, &lap)?;
    Ok((kps, des))
}

pub fn imread(filename: &str) -> opencv::Result<Mat> {
    let mut img = imgcodecs::imread(filename, imgcodecs::IMREAD_GRAYSCALE)?;
    if img.cols() > 1920 || img.rows() > 1080 {
        img = adjust_image_size(&img, 1920, 1080)?;
    }
    Ok(img)
}

pub fn imshow(winname: &str, mat: &dyn core::ToInputArray) -> opencv::Result<()> {
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

pub fn imwrite(filename: &str, img: &dyn core::ToInputArray) -> opencv::Result<bool> {
    let flags = types::VectorOfi32::new();
    imgcodecs::imwrite(filename, img, &flags)
}

pub fn adjust_image_size(img: &Mat, width: i32, height: i32) -> opencv::Result<Mat> {
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
) -> opencv::Result<Mat> {
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
) -> opencv::Result<Mat> {
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

pub struct TimeMeasure(pub HashMap<String, Duration>);

impl TimeMeasure {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn measure<F, R>(&mut self, key: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let r = f();
        *self.0.entry(key.to_owned()).or_insert(Duration::default()) += Instant::now() - start;
        r
    }
}

pub fn read_line(prompt: &str) -> anyhow::Result<String> {
    print!("{}", prompt);
    std::io::stdout().flush()?;
    let v = std::io::stdin()
        .bytes()
        .take_while(|c| c.as_ref().ok() != Some(&b'\n'))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(String::from_utf8(v)?.trim().to_owned())
}
