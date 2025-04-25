use imsearch::OPTS;
use imsearch::cmd::SubCommandExtend;
use imsearch::config::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    match &OPTS.subcmd {
        SubCommand::ShowKeypoints(config) => {
            config.run(&OPTS).await?;
        }
        SubCommand::ShowMatches(config) => {
            config.run(&OPTS).await?;
        }
        SubCommand::AddImages(config) => {
            config.run(&OPTS).await?;
        }
        SubCommand::SearchImage(config) => {
            config.run(&OPTS).await?;
        }
        SubCommand::BuildIndex(config) => {
            config.run(&OPTS).await?;
        }
        SubCommand::StartServer(config) => {
            config.run(&OPTS).await?;
        }
        SubCommand::ClearCache(config) => {
            config.run(&OPTS).await?;
        }
        SubCommand::ExportData(config) => {
            config.run(&OPTS).await?;
        }
        #[cfg(feature = "rocksdb")]
        SubCommand::UpdateDB(config) => {
            config.run(&*OPTS).await?;
        }
    }

    Ok(())
}
