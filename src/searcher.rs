use std::iter;
use std::path::Path;

use crate::db::ImageDB;
use crate::slam3_orb::Slam3ORB;
use crate::utils;
use anyhow::Result;
use opencv::prelude::{MatTraitConst, MatTraitConstManual};

pub struct Neighbor {
    pub id: usize,
    pub distance: u32,
}

pub trait Matrix {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn as_ptr(&self) -> *const u8;
}

pub trait KnnSearcher {
    type Error;

    /// Add points
    fn add<T: Matrix>(&mut self, points: T) -> Result<bool, Self::Error>;
    /// Add points with specific id
    fn add_with_ids<T: Matrix>(&mut self, points: T, ids: &[usize]) -> Result<(), Self::Error>;
    /// Search K nearest neighbours
    fn search<T: Matrix>(&self, points: T, k: usize) -> Result<Vec<Vec<Neighbor>>, Self::Error>;
    /// Train index, may be no-op
    fn train<T>(&mut self, points: T) -> Result<(), Self::Error>;
    /// Build index, may be no-op
    fn build(&mut self) -> Result<(), Self::Error>;
    /// Read index from file
    fn read<P: AsRef<Path>>(&self, path: P) -> Result<Self, Self::Error>
    where
        Self: Sized;
    /// Write index to file
    fn write<P: AsRef<Path>>(&self, path: P) -> Result<(), Self::Error>;
}

pub struct ImSearcher {
    db: ImageDB,
}

impl ImSearcher {
    pub fn new<P: AsRef<Path>>(conf_dir: P, read_only: bool) -> Result<Self> {
        let db = ImageDB::open(conf_dir, read_only)?;
        Ok(Self { db })
    }

    pub fn add_image<S: AsRef<str>>(&self, image_path: S, orb: &mut Slam3ORB) -> Result<bool> {
        if self.db.image_exists(image_path.as_ref())? {
            return Ok(false);
        }

        let image = utils::imread(image_path.as_ref())?;
        let (_, descriptors) = utils::detect_and_compute(orb, &image)?;

        self.db.add_image(image_path.as_ref(), descriptors)
    }

    pub fn search_image(&self) {

    }
}
