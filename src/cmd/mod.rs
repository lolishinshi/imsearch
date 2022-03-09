mod show;

use anyhow::Result;
pub use show::*;
use crate::config::Opts;


pub trait SubCommand {
    fn run(&self, opts: &Opts) -> Result<()>;
}
