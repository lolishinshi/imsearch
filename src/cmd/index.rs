use crate::cmd::SubCommandExtend;
use crate::{Opts, IMDB};
use anyhow::Result;
use ndarray_npy::write_npy;
use structopt::StructOpt;

#[derive(StructOpt, Debug, Clone)]
pub struct MarkAsIndexed {
    /// Mark feature in [0, max_feature_id) as trained
    #[structopt(long)]
    pub max_feature_id: u64,
}

#[derive(StructOpt, Debug, Clone)]
pub struct ClearCache {
    /// Also clear unindexed features
    #[structopt(long)]
    pub unindexed: bool,
}

#[derive(StructOpt, Debug, Clone)]
pub struct BuildIndex {
    /// Skip index < start
    #[structopt(long)]
    pub start: Option<u64>,
    /// Skip index >= end
    #[structopt(long)]
    pub end: Option<u64>,
}

#[derive(StructOpt, Debug, Clone)]
pub struct ExportData {}

impl SubCommandExtend for MarkAsIndexed {
    fn run(&self, opts: &Opts) -> Result<()> {
        let db = IMDB::new(opts.conf_dir.clone(), false)?;
        db.mark_as_indexed(self.max_feature_id, opts.batch_size)
    }
}

impl SubCommandExtend for ClearCache {
    fn run(&self, opts: &Opts) -> Result<()> {
        let db = IMDB::new(opts.conf_dir.clone(), false)?;
        db.clear_cache(self.unindexed)
    }
}

impl SubCommandExtend for BuildIndex {
    fn run(&self, opts: &Opts) -> Result<()> {
        let db = IMDB::new(opts.conf_dir.clone(), false)?;
        db.build_index(opts.batch_size, self.start, self.end)
    }
}

impl SubCommandExtend for ExportData {
    fn run(&self, opts: &Opts) -> Result<()> {
        let db = IMDB::new(opts.conf_dir.clone(), true)?;
        let data = db.export()?;
        write_npy("train.npy", &data)?;
        Ok(())
    }
}
