use imsearch::cmd::SubCommandExtend;
use imsearch::config::*;
use imsearch::OPTS;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    fdlimit::raise_fd_limit()?;
    // debug!("raise fdlimit to {:?}", fdlimit);

    match &OPTS.subcmd {
        SubCommand::ShowKeypoints(config) => {
            config.run(&*OPTS).unwrap();
        }
        SubCommand::ShowMatches(config) => {
            config.run(&*OPTS).unwrap();
        }
        SubCommand::AddImages(config) => {
            config.run(&*OPTS).unwrap();
        }
        SubCommand::SearchImage(config) => {
            config.run(&*OPTS).unwrap();
        }
        SubCommand::BuildIndex(config) => {
            config.run(&*OPTS).unwrap();
        }
        SubCommand::StartServer(config) => {
            config.run(&*OPTS).unwrap();
        }
        SubCommand::ClearCache(config) => {
            config.run(&*OPTS).unwrap();
        }
        SubCommand::MarkAsIndexed(config) => {
            config.run(&*OPTS).unwrap();
        }
        SubCommand::ExportData(config) => {
            config.run(&*OPTS).unwrap();
        }
    }

    Ok(())
}
