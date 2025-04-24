use crate::cmd::SubCommandExtend;
use crate::index::FaissIndex;
use crate::{IMDB, Opts};
use anyhow::Result;
use clap::Parser;
use ndarray_npy::write_npy;

#[derive(Parser, Debug, Clone)]
pub struct ClearCache {
    /// Also clear unindexed features
    #[arg(long)]
    pub unindexed: bool,
}

#[derive(Parser, Debug, Clone)]
pub struct BuildIndex {}

#[derive(Parser, Debug, Clone)]
pub struct ExportData {}

#[derive(Parser, Debug, Clone)]
pub struct MergeIndex {
    #[arg(long)]
    pub index1: String,
    #[arg(long)]
    pub index2: String,
}

impl SubCommandExtend for ClearCache {
    #[tokio::main]
    async fn run(&self, opts: &Opts) -> Result<()> {
        let db = IMDB::new(opts.conf_dir.clone()).await?;
        db.clear_cache(self.unindexed).await
    }
}

impl SubCommandExtend for BuildIndex {
    #[tokio::main]
    async fn run(&self, opts: &Opts) -> Result<()> {
        let db = IMDB::new(opts.conf_dir.clone()).await?;
        db.build_index(opts.batch_size).await
    }
}

impl SubCommandExtend for ExportData {
    #[tokio::main]
    async fn run(&self, opts: &Opts) -> Result<()> {
        let db = IMDB::new(opts.conf_dir.clone()).await?;
        let data = db.export().await?;
        write_npy("train.npy", &data)?;
        Ok(())
    }
}

impl SubCommandExtend for MergeIndex {
    fn run(&self, _opts: &Opts) -> Result<()> {
        let mut index1 = FaissIndex::from_file(&self.index1, false);
        let index2 = FaissIndex::from_file(&self.index2, true);
        index1.merge_from(&index2, 0);
        index1.write_file("merge_index");
        Ok(())
    }
}
