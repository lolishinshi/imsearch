use clap::Parser;
use imsearch::cli::SubCommandExtend;
use imsearch::config::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

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
        #[cfg(feature = "rocksdb")]
        SubCommand::UpdateDB(config) => {
            config.run(&opts).await?;
        }
    }

    Ok(())
}
