mod add;
mod build;
mod clean;
mod search;
pub mod server;
mod train;

pub use add::*;
pub use build::*;
pub use clean::*;
pub use search::*;
pub use server::*;
pub use train::*;

use crate::config::Opts;

pub trait SubCommandExtend {
    fn run(&self, opts: &Opts) -> impl std::future::Future<Output = anyhow::Result<()>> + Send;
}
