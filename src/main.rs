use imsearch::OPTS;
use imsearch::cmd::SubCommandExtend;
use imsearch::config::*;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    match &OPTS.subcmd {
        SubCommand::ShowKeypoints(config) => {
            config.run(&OPTS)?;
        }
        SubCommand::ShowMatches(config) => {
            config.run(&OPTS)?;
        }
        SubCommand::AddImages(config) => {
            config.run(&OPTS)?;
        }
        SubCommand::SearchImage(config) => {
            config.run(&OPTS)?;
        }
        SubCommand::BuildIndex(config) => {
            config.run(&OPTS)?;
        }
        SubCommand::StartServer(config) => {
            config.run(&OPTS)?;
        }
        SubCommand::ClearCache(config) => {
            config.run(&OPTS)?;
        }
        SubCommand::ExportData(config) => {
            config.run(&OPTS)?;
        }
        SubCommand::MergeIndex(config) => {
            config.run(&OPTS).unwrap();
        }
        #[cfg(feature = "rocksdb")]
        SubCommand::UpdateDB(config) => {
            config.run(&*OPTS)?;
        }
    }

    Ok(())
}
