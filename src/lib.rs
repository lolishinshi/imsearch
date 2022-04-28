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
use once_cell::sync::Lazy;
use std::cell::RefCell;
use structopt::StructOpt;

pub static OPTS: Lazy<Opts> = Lazy::new(config::Opts::from_args);
thread_local! {
    pub static ORB: RefCell<Slam3ORB> = RefCell::new(Slam3ORB::from(&*OPTS));
}
