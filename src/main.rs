use imsearch::config::*;
use imsearch::flann::Flann;
use imsearch::slam3_orb::Slam3ORB;
use imsearch::utils;
use imsearch::utils::read_line;
use imsearch::ImageDb;
use itertools::Itertools;
use opencv::prelude::*;
use opencv::{core, features2d, types};
use rayon::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use walkdir::WalkDir;

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
    let db = ImageDb::from(opts);
    WalkDir::new(&config.path)
        .into_iter()
        .par_bridge()
        .for_each(|entry| {
            let entry = entry.unwrap().into_path();
            if entry
                .extension()
                .map(|s| re.is_match(&*s.to_string_lossy()))
                != Some(true)
            {
                return;
            }
            println!("Adding {}", entry.display());
            db.add(entry.to_string_lossy()).unwrap_or_else(|e| {
                eprintln!("ERROR: {}", e);
            });
        });
    Ok(())
}

fn search_image(opts: &Opts, config: &SearchImage) -> anyhow::Result<()> {
    let db = ImageDb::from(opts);
    let result = db.search(&config.image)?;
    print_result(&result)
}

fn start_repl(opts: &Opts, config: &StartRepl) -> anyhow::Result<()> {
    let db = ImageDb::from(opts);
    log::debug!("Reading data");
    let train_des = db.get_all_des()?;
    log::debug!("Building index");
    let mut flann = Flann::new(
        &train_des,
        opts.lsh_table_number,
        opts.lsh_key_size,
        opts.lsh_probe_level,
        opts.search_checks,
    )?;
    let mut orb = Slam3ORB::from(&*OPTS);

    while let Ok(s) = read_line(&config.prompt) {
        log::debug!("searching {:?}", s);
        if !PathBuf::from(&s).exists() {
            continue;
        }

        let start = Instant::now();

        let img = utils::imread(&s)?;
        let (_, query_des) = utils::detect_and_compute(&mut orb, &img)?;

        let mut results = HashMap::new();

        let matches = flann.knn_search(&query_des, OPTS.knn_k)?;
        for match_ in matches.into_iter() {
            for point in match_.into_iter() {
                let des = train_des.row(point.index as i32)?;
                let id = db.search_image_id_by_des(&des)?;
                *results.entry(id).or_insert(0.) +=
                    point.distance_squared / 500.0 / OPTS.knn_k as f32;
            }
        }

        let mut result = results
            .iter()
            .map(|(image_id, score)| {
                db.search_image_path_by_id(*image_id)
                    .map(|image_path| (*score, image_path))
            })
            .collect::<Result<Vec<_>, _>>()?;
        result.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let result = result.into_iter().take(OPTS.output_count).collect_vec();

        log::debug!("Take time: {:.2}s", (Instant::now() - start).as_secs_f32());

        print_result(&result)?;
    }

    Ok(())
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
