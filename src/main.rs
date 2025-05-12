use std::env;

use clap::Parser;
use imsearch::cli::SubCommandExtend;
use imsearch::config::*;
use imsearch::faiss::faiss_version;
use log::{info, warn};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match &*env::var("RUST_LOG").unwrap_or_default() {
        "debug" => {
            warn!(
                "不加限制的 debug 模式会导致导致 sqlx 等第三方库输出大量日志，建议使用 RUST_LOG=imsearch=debug"
            );
        }
        "" => unsafe {
            env::set_var("RUST_LOG", "imsearch=info");
        },
        _ => {}
    }

    env_logger::init();

    info!("faiss 版本: {}", faiss_version());

    let opts = Opts::parse();

    match &opts.subcmd {
        SubCommand::Show(config) => {
            config.run(&opts).await?;
        }
        SubCommand::Match(config) => {
            config.run(&opts).await?;
        }
        SubCommand::Add(config) => {
            config.run(&opts).await?;
        }
        SubCommand::Search(config) => {
            config.run(&opts).await?;
        }
        SubCommand::Build(config) => {
            config.run(&opts).await?;
        }
        SubCommand::Server(config) => {
            config.run(&opts).await?;
        }
        SubCommand::Clean(config) => {
            config.run(&opts).await?;
        }
        SubCommand::Export(config) => {
            config.run(&opts).await?;
        }
    }

    Ok(())
}
