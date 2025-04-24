use crate::cmd::SubCommandExtend;
use crate::index::FaissIndex;
use crate::{Opts, IMDB};
use anyhow::Result;
use clap::Parser;
use ndarray_npy::write_npy;

#[derive(Parser, Debug, Clone)]
pub struct MarkAsIndexed {
    /// Mark feature in [0, max_feature_id) as trained
    #[arg(long)]
    pub max_feature_id: u64,
}

#[derive(Parser, Debug, Clone)]
pub struct ClearCache {
    /// Also clear unindexed features
    #[arg(long)]
    pub unindexed: bool,
}

#[derive(Parser, Debug, Clone)]
pub struct BuildIndex {
    /// Skip index < start
    #[arg(long)]
    pub start: Option<u64>,
    /// Skip index >= end
    #[arg(long)]
    pub end: Option<u64>,
}

#[derive(Parser, Debug, Clone)]
pub struct ExportData {}

#[derive(Parser, Debug, Clone)]
pub struct MergeIndex {
    #[arg(long)]
    pub index1: String,
    #[arg(long)]
    pub index2: String,
}

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

impl SubCommandExtend for MergeIndex {
    fn run(&self, opts: &Opts) -> Result<()> {
        let mut index1 = FaissIndex::from_file(&self.index1, false);
        let index2 = FaissIndex::from_file(&self.index2, true);
        index1.merge_from(&index2, 0);

        index1.write_file("merge_index");

        Ok(())
    }
}
