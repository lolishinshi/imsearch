use crate::cmd::SubCommandExtend;
use crate::config::{Opts, OutputFormat};
use crate::slam3_orb::Slam3ORB;
use crate::IMDB;
use crate::ORB;
use anyhow::Result;
use rayon::prelude::*;
use regex::Regex;
use structopt::StructOpt;
use walkdir::WalkDir;

#[derive(StructOpt, Debug, Clone)]
pub struct AddImages {
    /// Path to an image or folder
    pub path: String,
    /// Scan image with these suffixes
    #[structopt(short, long, default_value = "jpg,png")]
    pub suffix: String,
}

#[derive(StructOpt, Debug, Clone)]
pub struct SearchImage {
    /// Path to the image to search
    pub image: String,
}

impl SubCommandExtend for AddImages {
    fn run(&self, opts: &Opts) -> anyhow::Result<()> {
        let re = Regex::new(&self.suffix.replace(',', "|")).expect("failed to build regex");
        let db = IMDB::new(opts.conf_dir.clone(), false)?;
        WalkDir::new(&self.path)
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
}

impl SubCommandExtend for SearchImage {
    fn run(&self, opts: &Opts) -> anyhow::Result<()> {
        let db = IMDB::new(opts.conf_dir.clone(), true)?;
        let mut orb = Slam3ORB::from(opts);

        let mut index = db.get_multi_index(opts.mmap);
        index.set_nprobe(opts.nprobe);

        let mut result = db.search(&index, &self.image, &mut orb, 3, opts.distance)?;

        result.truncate(opts.output_count);
        print_result(&result, opts)
    }
}

fn print_result(result: &[(f32, String)], opts: &Opts) -> Result<()> {
    match opts.output_format {
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
