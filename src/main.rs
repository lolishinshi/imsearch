use imsearch::config::{Opts, ShowKeypoints, ShowMatches, SubCommand};
use imsearch::slam3_orb::Slam3ORB;
use imsearch::utils;
use imsearch::ImageDb;
use opencv::prelude::*;
use opencv::{core, features2d, types};
use structopt::StructOpt;

fn show_keypoints(opts: &Opts, config: &ShowKeypoints) -> opencv::Result<()> {
    let image = utils::imread(&config.image)?;

    let mut orb = Slam3ORB::from(opts);
    let (kps, _) = utils::detect_and_compute(&mut orb, &image)?;
    let output = utils::draw_keypoints(&image, &kps)?;

    match &config.output {
        Some(file) => {
            utils::imwrite(&file, &output)?;
        }
        _ => utils::imshow("result", &output)?,
    }
    Ok(())
}

fn show_matches(opts: &Opts, config: &ShowMatches) -> opencv::Result<()> {
    let img1 = utils::imread(&config.image1)?;
    let img2 = utils::imread(&config.image2)?;

    let mut orb = Slam3ORB::from(opts);
    let (kps1, des1) = utils::detect_and_compute(&mut orb, &img1)?;
    let (kps2, des2) = utils::detect_and_compute(&mut orb, &img2)?;

    let mut matches = types::VectorOfVectorOfDMatch::new();
    let mask = core::Mat::default();
    let flann = features2d::FlannBasedMatcher::from(opts);
    flann.knn_train_match(&des1, &des2, &mut matches, 2, &mask, false)?;

    let mut matches_mask = vec![];
    for match_ in matches.iter() {
        if match_.len() != 2 {
            matches_mask.push(types::VectorOfi8::from_iter([0, 0]));
            continue;
        }
        let (m, n) = (match_.get(0)?, match_.get(1)?);
        if m.distance < 0.7 * n.distance {
            matches_mask.push(types::VectorOfi8::from_iter([1, 0]));
        } else {
            matches_mask.push(types::VectorOfi8::from_iter([0, 0]));
        }
    }
    let matches_mask = types::VectorOfVectorOfi8::from(matches_mask);

    let output = utils::draw_matches_knn(&img1, &kps1, &img2, &kps2, &matches, &matches_mask)?;
    match &config.output {
        Some(file) => {
            utils::imwrite(&file, &output)?;
        }
        _ => utils::imshow("result", &output)?,
    }

    Ok(())
}

fn main() {
    let opts: Opts = Opts::from_args();

    match &opts.subcmd {
        SubCommand::ShowKeypoints(config) => {
            show_keypoints(&opts, &config).unwrap();
        }
        SubCommand::ShowMatches(config) => {
            show_matches(&opts, &config).unwrap();
        }
        _ => todo!(),
    }
}
