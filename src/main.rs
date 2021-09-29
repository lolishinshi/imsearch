use std::cell::RefCell;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use imsearch::config::*;
use imsearch::slam3_orb::Slam3ORB;
use imsearch::utils;
use imsearch::IMDB;
use log::debug;
use ndarray_npy::write_npy;
use once_cell::sync::Lazy;
use opencv::prelude::*;
use opencv::{core, features2d, imgcodecs, types};
use rayon::prelude::*;
use regex::Regex;
use rouille::{post_input, router, try_or_400, Response};
use structopt::StructOpt;
use walkdir::WalkDir;

pub static OPTS: Lazy<Opts> = Lazy::new(Opts::from_args);
thread_local! {
    static ORB: RefCell<Slam3ORB> = RefCell::new(Slam3ORB::from(&*OPTS));
}

fn show_keypoints(opts: &Opts, config: &ShowKeypoints) -> Result<()> {
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

fn show_matches(opts: &Opts, config: &ShowMatches) -> Result<()> {
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

fn add_images(opts: &Opts, config: &AddImages) -> Result<()> {
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

            let result =
                ORB.with(|orb| db.add_image(entry.to_string_lossy(), &mut *orb.borrow_mut()));
            match result {
                Ok(add) => match add {
                    true => println!("[OK] Add {}", entry.display()),
                    false => println!("[OK] Update {}", entry.display()),
                },
                Err(e) => eprintln!("[ERR] {}: {}\n{}", entry.display(), e, e.backtrace()),
            }
        });
    Ok(())
}

fn search_image(opts: &Opts, config: &SearchImage) -> Result<()> {
    let db = IMDB::new(opts.conf_dir.clone(), true)?;
    let mut orb = Slam3ORB::from(opts);

    let mut index = db.get_index(opts.mmap);
    index.set_nprobe(opts.nprobe);

    let start = Instant::now();
    let mut result = db.search(&index, &config.image, &mut orb, 3, opts.distance)?;
    debug!("search time: {:.2}s", start.elapsed().as_secs_f32());

    result.truncate(opts.output_count);
    print_result(&result)
}

fn start_repl(opts: &Opts, config: &StartRepl) -> Result<()> {
    let db = IMDB::new(opts.conf_dir.clone(), true)?;
    let mut orb = Slam3ORB::from(opts);

    let mut index = db.get_index(opts.mmap);
    index.set_nprobe(opts.nprobe);

    debug!("start REPL");
    while let Ok(line) = utils::read_line(&config.prompt) {
        if !PathBuf::from(&line).exists() {
            continue;
        }

        let start = Instant::now();
        let mut result = db.search(&index, line, &mut orb, 3, opts.distance)?;
        debug!("search time: {:.2}s", start.elapsed().as_secs_f32());

        result.truncate(opts.output_count);
        print_result(&result)?;
    }

    Ok(())
}

fn build_index(opts: &Opts) -> Result<()> {
    let db = IMDB::new(opts.conf_dir.clone(), false)?;
    db.build_index(opts.batch_size)
}

fn mark_as_trained(opts: &Opts, config: &MarkAsIndexed) -> Result<()> {
    let db = IMDB::new(opts.conf_dir.clone(), false)?;
    db.mark_as_indexed(config.max_feature_id, opts.batch_size)
}

fn clear_cache(opts: &Opts, config: &ClearCache) -> Result<()> {
    let db = IMDB::new(opts.conf_dir.clone(), false)?;
    db.clear_cache(config.unindexed)
}

fn export_data(opts: &Opts) -> Result<()> {
    let db = IMDB::new(opts.conf_dir.clone(), true)?;
    let data = db.export()?;
    write_npy("train.npy", &data)?;
    Ok(())
}

fn start_server(opts: &Opts, config: &StartServer) -> Result<()> {
    let db = IMDB::new(opts.conf_dir.clone(), true)?;

    let mut index = db.get_index(opts.mmap);
    index.set_nprobe(opts.nprobe);

    let opts = opts.clone();

    debug!("starting server at {}", &config.addr);
    rouille::start_server(&config.addr, move |request| {
        router!(request,
            (POST) (/search) => {
                let data = try_or_400!(post_input!(request, {
                    file: rouille::input::post::BufferedFile,
                }));
                let mat = try_or_400!(Mat::from_slice(&data.file.data));
                let img = try_or_400!(imgcodecs::imdecode(&mat, imgcodecs::IMREAD_GRAYSCALE));

                let result = ORB.with(|orb| {
                    utils::detect_and_compute(&mut *orb.borrow_mut(), &img)
                        .and_then(|(_, descriptors)| {
                        db.search_des(&index, descriptors, opts.knn_k, opts.distance)
                    })
                });

                match result {
                    Ok(mut result) => {
                        result.truncate(opts.output_count);
                        Response::json(&result)
                    },
                    Err(err) => {
                        // TODO: 此处错误处理很简陋
                        Response::json(&err.to_string()).with_status_code(400)
                    },
                }
            },
            _ => {
                Response::empty_404()
            }
        )
    });
}

fn main() {
    env_logger::init();

    let fdlimit = fdlimit::raise_fd_limit();
    debug!("raise fdlimit to {:?}", fdlimit);

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
        SubCommand::StartServer(config) => {
            start_server(&*OPTS, config).unwrap();
        }
        SubCommand::ClearCache(config) => {
            clear_cache(&*OPTS, config).unwrap();
        }
        SubCommand::MarkAsIndexed(config) => {
            mark_as_trained(&*OPTS, config).unwrap();
        }
        SubCommand::ExportData => {
            export_data(&*OPTS).unwrap();
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
