pub mod cmd;
pub mod config;
pub mod db;
pub mod imdb;
pub mod index;
pub mod matrix;
#[cfg(feature = "rocksdb")]
pub mod rocks;
pub mod slam3_orb;
pub mod utils;

pub use imdb::IMDB;

use crate::config::Opts;
use crate::slam3_orb::Slam3ORB;
use clap::Parser;
use std::cell::RefCell;
use std::sync::LazyLock;

pub static OPTS: LazyLock<Opts> = LazyLock::new(config::Opts::parse);

thread_local! {
    pub static ORB: RefCell<Slam3ORB> = RefCell::new(Slam3ORB::from(&*OPTS));
}
