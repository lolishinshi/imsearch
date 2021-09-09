use crate::slam3_orb::Slam3ORB;
use opencv::core;
use opencv::features2d;
use opencv::highgui;
use opencv::imgcodecs;
use opencv::prelude::*;
use opencv::types;

pub fn detect_and_compute(
    orb: &mut Slam3ORB,
    image: &dyn core::ToInputArray,
) -> opencv::Result<(types::VectorOfKeyPoint, Mat)> {
    let mask = Mat::default();
    let lap = types::VectorOfi32::from(vec![0, 0]);
    let mut kps = types::VectorOfKeyPoint::new();
    let mut des = Mat::default();
    orb.detect_and_compute(image, &mask, &mut kps, &mut des, &lap)?;
    return Ok((kps, des));
}

pub fn imread(filename: &str) -> opencv::Result<Mat> {
    imgcodecs::imread(filename, imgcodecs::IMREAD_GRAYSCALE)
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
