use anyhow::Result;
use image_search::ImageDb;
use walkdir::WalkDir;

use std::env;
use std::path::PathBuf;

fn main() {
    let mut db = ImageDb::open("./image.db").unwrap();

    let args = env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        println!("Usage: {} add    DIR", args[0]);
        println!("       {} search FILE...", args[0]);
        return;
    }

    let path = PathBuf::from(&args[2]);
    match args[1].as_str() {
        "add" => {
            for entry in WalkDir::new(path) {
                let path = entry.unwrap().into_path();
                if path.is_dir() {
                    continue;
                }
                println!("Indexing {}", path.display());
                db.add(path.to_string_lossy()).unwrap();
            }
        }
        "search" => {
            db.search(path.to_string_lossy()).unwrap();
        }
        _ => unimplemented!(),
    }
}
