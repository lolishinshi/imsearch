mod add;
mod build;
mod clean;
mod export;
mod search;
pub mod server;

pub use add::*;
pub use build::*;
pub use clean::*;
pub use export::*;
pub use search::*;
pub use server::*;

use crate::config::Opts;

pub trait SubCommandExtend {
    fn run(&self, opts: &Opts) -> impl std::future::Future<Output = anyhow::Result<()>> + Send;
}
