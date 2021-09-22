use std::cell::RefCell;
use std::path::PathBuf;
use std::time::Instant;

use imsearch::config::*;
use imsearch::slam3_orb::Slam3ORB;
use imsearch::utils;
use imsearch::IMDB;
use once_cell::sync::Lazy;
use opencv::prelude::*;
use opencv::{core, features2d, types};
use rayon::prelude::*;
use regex::Regex;
use structopt::StructOpt;
use walkdir::WalkDir;

pub static OPTS: Lazy<Opts> = Lazy::new(Opts::from_args);
thread_local! {
    static ORB: RefCell<Slam3ORB> = RefCell::new(Slam3ORB::from(&*OPTS));
}

fn show_keypoints(opts: &Opts, config: &ShowKeypoints) -> opencv::Result<()> {
    let image = utils::imread(&config.image)?;

    let mut orb = Slam3ORB::from(opts);
    let (kps, _) = utils::detect_and_compute(&mut orb, &image)?;
    let output = utils::draw_keypoints(&image, &kps)?;

    match &config.output {
        Some(file) => {
            utils::imwrite(file, &output)?;
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
            utils::imwrite(file, &output)?;
        }
        _ => utils::imshow("result", &output)?,
    }

    Ok(())
}

fn add_images(opts: &Opts, config: &AddImages) -> anyhow::Result<()> {
    let re = Regex::new(&config.suffix.replace(',', "|")).expect("failed to build regex");
    let db = IMDB::new(opts.conf_dir.clone(), false)?;
    WalkDir::new(&config.path)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .par_bridge()
        .for_each(|entry| {
            let entry = entry.into_path();
            if entry
                .extension()
                .map(|s| re.is_match(&*s.to_string_lossy()))
                != Some(true)
            {
                return;
            }
            println!("Adding {}", entry.display());
            ORB.with(|orb| db.add_image(entry.to_string_lossy(), &mut *orb.borrow_mut()))
                .expect("Failed to add image");
        });
    Ok(())
}

fn search_image(opts: &Opts, config: &SearchImage) -> anyhow::Result<()> {
    let db = IMDB::new(opts.conf_dir.clone(), true)?;
    let mut orb = Slam3ORB::from(opts);
    let index = db.get_index(opts.mmap);
    let result = db.search(&index, &config.image, &mut orb, 3, opts.distance)?;
    print_result(&result)
}

fn start_repl(opts: &Opts, config: &StartRepl) -> anyhow::Result<()> {
    let db = IMDB::new(opts.conf_dir.clone(), true)?;
    let mut orb = Slam3ORB::from(opts);

    log::debug!("Reading index");
    let index = db.get_index(opts.mmap);

    log::debug!("Start REPL");
    while let Ok(line) = utils::read_line(&config.prompt) {
        if !PathBuf::from(&line).exists() {
            continue;
        }

        log::debug!("Searching {:?}", line);
        let start = Instant::now();
        let result = db.search(&index, line, &mut orb, 3, opts.distance)?;

        log::debug!("Take time: {:.2}s", (Instant::now() - start).as_secs_f32());

        print_result(&result)?;
    }

    Ok(())
}

fn build_index(opts: &Opts) -> anyhow::Result<()> {
    let db = IMDB::new(opts.conf_dir.clone(), true)?;
    db.build_database(opts.batch_size)
}

fn main() {
    env_logger::init();

    match &OPTS.subcmd {
        SubCommand::ShowKeypoints(config) => {
            show_keypoints(&*OPTS, config).unwrap();
        }
        SubCommand::ShowMatches(config) => {
            show_matches(&*OPTS, config).unwrap();
        }
        SubCommand::AddImages(config) => {
            add_images(&*OPTS, config).unwrap();
        }
        SubCommand::SearchImage(config) => {
            search_image(&*OPTS, config).unwrap();
        }
        SubCommand::StartRepl(config) => {
            start_repl(&*OPTS, config).unwrap();
        }
        SubCommand::BuildIndex => {
            build_index(&*OPTS).unwrap();
        }
    }
}

fn print_result(result: &[(f32, String)]) -> anyhow::Result<()> {
    match OPTS.output_format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(result)?)
        }
        OutputFormat::Table => {
            for (k, v) in result {
                println!("{:.2}\t{}", k, v);
            }
        }
    }
    Ok(())
}
