pub mod cmd;
pub mod config;
pub mod db;
pub mod imdb;
pub mod index;
pub mod matrix;
pub mod slam3_orb;
pub mod utils;

pub use imdb::IMDB;

use crate::config::Opts;
use crate::slam3_orb::Slam3ORB;
use clap::Parser;
use once_cell::sync::Lazy;
use std::cell::RefCell;

pub static OPTS: Lazy<Opts> = Lazy::new(config::Opts::parse);
thread_local! {
    pub static ORB: RefCell<Slam3ORB> = RefCell::new(Slam3ORB::from(&*OPTS));
}
