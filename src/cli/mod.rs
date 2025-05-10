mod add;
mod build;
mod clean;
mod export;
mod r#match;
mod search;
pub mod server;
mod show;

pub use add::*;
pub use build::*;
pub use clean::*;
pub use export::*;
pub use r#match::*;
pub use search::*;
pub use server::*;
pub use show::*;

use crate::config::Opts;

pub trait SubCommandExtend {
    fn run(&self, opts: &Opts) -> impl std::future::Future<Output = anyhow::Result<()>> + Send;
}
