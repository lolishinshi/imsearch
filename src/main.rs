use std::env;

use clap::Parser;
use imsearch::cli::SubCommandExtend;
use imsearch::config::*;
use log::{info, warn};
use tikv_jemallocator::Jemalloc;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn simd_version() -> &'static str {
    if cfg!(target_feature = "avx512vpopcntdq") {
        "avx512vpopcntdq"
    } else if cfg!(target_feature = "avx512f") {
        "avx512f"
    } else if cfg!(target_feature = "avx2") {
        "avx2"
    } else if cfg!(target_feature = "sse4.2") {
        "sse4.2"
    } else if cfg!(target_feature = "sse4.1") {
        "sse4.1"
    } else if cfg!(target_feature = "sse3") {
        "sse3"
    } else if cfg!(target_feature = "sse2") {
        "sse2"
    } else if cfg!(target_feature = "sse") {
        "sse"
    } else {
        "none"
    }
}

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

    info!("SIMD 版本: {}", simd_version());

    let opts = Opts::parse();

    match &opts.subcmd {
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
        SubCommand::Train(config) => {
            config.run(&opts).await?;
        }
    }

    Ok(())
}
