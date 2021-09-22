use std::convert::TryInto;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::config::ConfDir;
use crate::matrix::Matrix;
use anyhow::Result;
use rocksdb::{BoundColumnFamily, IteratorMode, Options, ReadOptions, WriteBatch, DB};

#[derive(Debug, Hash, Eq, PartialEq)]
pub(super) enum ImageColumnFamily {
    /// HashMap<u64, Box<[u8]>>
    IdToFeature,
    /// HashMap<u64, String>
    IdToImage,
    /// HashSet<Hash>
    ImageList,
    /// See `MetaData`
    MetaData,
    /// Just like IdToFeature, but only contains features which haven't been stored
    NewFeature,
}

pub(super) enum MetaData {
    TotalFeatures,
    TotalImages,
}

impl AsRef<str> for ImageColumnFamily {
    fn as_ref(&self) -> &str {
        match self {
            Self::IdToFeature => "id_to_feature",
            Self::IdToImage => "id_to_id",
            Self::ImageList => "image_list",
            Self::MetaData => "meta_data",
            Self::NewFeature => "new_feature",
        }
    }
}

impl AsRef<[u8]> for MetaData {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::TotalFeatures => b"total_features",
            Self::TotalImages => b"total_images",
        }
    }
}

pub struct ImageDB {
    db: DB,
    total_images: AtomicU64,
    total_features: AtomicU64,
}

impl ImageDB {
    /// Open the database, will create if not exists
    pub fn open(path: &ConfDir, read_only: bool) -> Result<Self> {
        super::update::check_db_update(path)?;

        let mut options = Options::default();
        options.increase_parallelism(num_cpus::get() as i32);
        options.set_keep_log_file_num(100);

        let db = match read_only {
            true => DB::open_for_read_only(&options, path.database(), false)?,
            false => DB::open(&options, path.database())?,
        };

        let meta_data = db.cf_handle(ImageColumnFamily::MetaData.as_ref()).unwrap();
        let total_images = db.get_cf(&meta_data, MetaData::TotalImages)?.unwrap();
        let total_features = db.get_cf(&meta_data, MetaData::TotalFeatures)?.unwrap();

        // meta_data borrows db here
        drop(meta_data);

        Ok(Self {
            db,
            total_images: AtomicU64::new(bytes_to_u64(total_images)),
            total_features: AtomicU64::new(bytes_to_u64(total_features)),
        })
    }

    /// Check whether an image exists
    pub fn image_exists(&self, hash: &[u8]) -> Result<bool> {
        Ok(self
            .db
            .get_cf(&self.cf(ImageColumnFamily::ImageList), hash)?
            .is_some())
    }

    /// Add an image and its features to database
    ///
    /// return false if the image is already inserted
    pub fn add_image<S, T>(&self, path: S, hash: &[u8], features: T) -> Result<bool>
    where
        S: AsRef<str>,
        T: Matrix,
    {
        let new_feature = self.cf(ImageColumnFamily::NewFeature);
        let id_to_image = self.cf(ImageColumnFamily::IdToImage);
        let image_list = self.cf(ImageColumnFamily::ImageList);

        if self.db.get_cf(&image_list, hash)?.is_some() {
            return Ok(false);
        }

        // insert id => feature to NewFeature
        // insert id => image
        let mut batch = WriteBatch::default();
        for feature in features.iter_lines() {
            let id = self.total_features.fetch_add(1, Ordering::SeqCst);
            batch.put_cf(&new_feature, id.to_le_bytes(), feature);
            batch.put_cf(&id_to_image, id.to_le_bytes(), path.as_ref());
        }
        // insert image list
        batch.put_cf(&image_list, hash, []);

        self.db.write(batch)?;

        let total_images = self.total_images.fetch_add(1, Ordering::SeqCst);
        let total_features = self.total_features.load(Ordering::SeqCst);

        // update total_images and total_features
        let meta_data = self.cf(ImageColumnFamily::MetaData);
        self.db.put_cf(
            &meta_data,
            MetaData::TotalImages,
            total_images.to_le_bytes(),
        )?;
        self.db.put_cf(
            &meta_data,
            MetaData::TotalFeatures,
            total_features.to_le_bytes(),
        )?;

        Ok(true)
    }

    /// Return an iterator of features
    pub fn features(&self, indexed: bool) -> impl Iterator<Item = (u64, Box<[u8]>)> + '_ {
        let family = match indexed {
            true => ImageColumnFamily::IdToFeature,
            false => ImageColumnFamily::NewFeature,
        };
        self.db
            .iterator_cf_opt(&self.cf(family), Self::read_opts(), IteratorMode::Start)
            .map(|item| (bytes_to_u64(item.0), item.1))
    }

    /// Find image according to feature id
    pub fn find_image(&self, id: u64) -> Result<String> {
        Ok(self
            .db
            .get_cf(&self.cf(ImageColumnFamily::IdToImage), id.to_le_bytes())
            .map(|data| String::from_utf8(data.unwrap()).unwrap())?)
    }

    /// Mark a list of features as trained
    pub fn mark_as_trained(&self, ids: &[u64]) -> Result<()> {
        let new_feature = self.cf(ImageColumnFamily::NewFeature);
        let id_to_feature = self.cf(ImageColumnFamily::IdToFeature);

        let mut batch = WriteBatch::default();
        // TODO: use multi_get_cf
        for id in ids {
            let id = id.to_le_bytes();
            let feature = self.db.get_pinned_cf(&new_feature, id)?.unwrap();
            batch.delete_cf(&new_feature, id);
            batch.put_cf(&id_to_feature, id, feature);
        }
        self.db.write(batch)?;

        Ok(())
    }

    fn read_opts() -> ReadOptions {
        let mut options = ReadOptions::default();
        options.set_verify_checksums(false);
        options
    }

    /// get column family handle
    fn cf(&self, cf: ImageColumnFamily) -> Arc<BoundColumnFamily> {
        // TODO: cache
        self.db.cf_handle(cf.as_ref()).unwrap()
    }
}

fn bytes_to_u64<T: AsRef<[u8]>>(bytes: T) -> u64 {
    u64::from_le_bytes(
        bytes
            .as_ref()
            .try_into()
            .expect("byte cannot convert to usize"),
    )
}
