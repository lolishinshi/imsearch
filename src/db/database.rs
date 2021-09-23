use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::config::ConfDir;
use crate::db::utils::{bytes_to_i32, bytes_to_u64};
use crate::matrix::Matrix;
use anyhow::Result;
use rocksdb::{BoundColumnFamily, IteratorMode, Options, ReadOptions, WriteBatch, DB};

#[derive(Debug, Hash, Eq, PartialEq)]
pub(super) enum ImageColumnFamily {
    /// HashMap<u64, Box<[u8]>>
    IdToFeature,
    /// HashMap<u64, u64>
    IdToImageId,
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

impl ImageColumnFamily {
    pub fn all() -> Vec<Self> {
        vec![
            Self::ImageList,
            Self::IdToImageId,
            Self::NewFeature,
            Self::IdToFeature,
            Self::MetaData,
            Self::IdToImage,
        ]
    }
}

impl AsRef<str> for ImageColumnFamily {
    fn as_ref(&self) -> &str {
        match self {
            Self::IdToFeature => "id_to_feature",
            Self::IdToImageId => "id_to_image_id",
            Self::IdToImage => "id_to_image",
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
        options.set_keep_log_file_num(10);
        options.set_level_compaction_dynamic_level_bytes(true);
        options.set_max_total_wal_size(1 << 29);

        let column_family = ImageColumnFamily::all();

        let db = match read_only {
            true => DB::open_cf_for_read_only(&options, path.database(), &column_family, false)?,
            false => DB::open_cf(&options, path.database(), &column_family)?,
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
    pub fn find_image_id(&self, hash: &[u8]) -> Result<Option<i32>> {
        let image_list = self.cf(ImageColumnFamily::ImageList);
        Ok(self
            .db
            .get_cf(&image_list, hash)?
            .map(|data| bytes_to_i32(data)))
    }

    /// update image path
    pub fn update_image_path(&self, image_id: i32, path: &str) -> Result<()> {
        let id_to_image = self.cf(ImageColumnFamily::IdToImage);
        Ok(self.db.put_cf(&id_to_image, image_id.to_le_bytes(), path)?)
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
        let id_to_image_id = self.cf(ImageColumnFamily::IdToImageId);
        let id_to_image = self.cf(ImageColumnFamily::IdToImage);
        let image_list = self.cf(ImageColumnFamily::ImageList);

        if self.db.get_cf(&image_list, hash)?.is_some() {
            return Ok(false);
        }

        let mut batch = WriteBatch::default();

        // insert image_id => image_path
        let image_id = self.total_images.fetch_add(1, Ordering::SeqCst) as i32;
        batch.put_cf(&id_to_image, image_id.to_le_bytes(), path.as_ref());

        // insert feature_id => feature to NewFeature
        // insert feature_id => image_id
        for feature in features.iter_lines() {
            let id = self.total_features.fetch_add(1, Ordering::SeqCst);
            batch.put_cf(&new_feature, id.to_le_bytes(), feature);
            batch.put_cf(&id_to_image_id, id.to_le_bytes(), image_id.to_le_bytes());
        }
        // insert image_hash => image_id
        batch.put_cf(&image_list, hash, image_id.to_le_bytes());

        let total_images = self.total_images.load(Ordering::SeqCst);
        let total_features = self.total_features.load(Ordering::SeqCst);

        // update total_images and total_features
        let meta_data = self.cf(ImageColumnFamily::MetaData);
        batch.put_cf(
            &meta_data,
            MetaData::TotalImages,
            total_images.to_le_bytes(),
        );
        batch.put_cf(
            &meta_data,
            MetaData::TotalFeatures,
            total_features.to_le_bytes(),
        );

        self.db.write(batch)?;

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
    pub fn find_image_path(&self, feature_id: u64) -> Result<String> {
        let id_to_image_id = self.cf(ImageColumnFamily::IdToImageId);
        let id_to_image = self.cf(ImageColumnFamily::IdToImage);
        let image_id = self
            .db
            .get_cf(&id_to_image_id, feature_id.to_le_bytes())
            .map(|data| bytes_to_i32(data.unwrap()))?;
        Ok(self
            .db
            .get_cf(&id_to_image, image_id.to_le_bytes())
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

    pub fn total_features(&self) -> u64 {
        self.total_features.load(Ordering::SeqCst)
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
