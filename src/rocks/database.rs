use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::config::ConfDir;
use crate::rocks::utils::{bytes_to_i32, bytes_to_u64, default_options};
use anyhow::Result;
use log::{debug, info};
use rocksdb::{BoundColumnFamily, ColumnFamilyDescriptor, DB, IteratorMode, ReadOptions};

#[derive(Debug, Hash, Eq, PartialEq)]
pub(super) enum ImageColumnFamily {
    /// HashMap<FeatureId, Box<[u8]>>
    IdToFeature,
    /// HashMap<FeatureId, ImageId>
    IdToImageId,
    /// HashMap<ImageId, String>
    IdToImage,
    /// HashMap<Hash, ImageId>
    ImageList,
    /// See `MetaData`
    MetaData,
    /// Just like IdToFeature, but only contains features which haven't been indexed
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

    pub fn descriptors() -> Vec<ColumnFamilyDescriptor> {
        Self::all()
            .into_iter()
            .map(|cf| ColumnFamilyDescriptor::new(cf.as_ref(), default_options()))
            .collect()
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
        let options = default_options();
        let cfs = ImageColumnFamily::all();
        let cf_descriptors = ImageColumnFamily::descriptors();

        let database_path = path.path().join("database");
        info!("open database at {}", database_path.display());

        let db = match read_only {
            true => DB::open_cf_for_read_only(&options, database_path, &cfs, false)?,
            false => DB::open_cf_descriptors(&options, database_path, cf_descriptors)?,
        };

        let meta_data = db.cf_handle(ImageColumnFamily::MetaData.as_ref()).unwrap();
        let total_images = bytes_to_u64(db.get_cf(&meta_data, MetaData::TotalImages)?.unwrap());
        let total_features = bytes_to_u64(db.get_cf(&meta_data, MetaData::TotalFeatures)?.unwrap());

        debug!("total images: {}", total_images);
        debug!("total features: {}", total_features);

        // meta_data borrows db here
        drop(meta_data);

        Ok(Self {
            db,
            total_images: AtomicU64::new(total_images),
            total_features: AtomicU64::new(total_features),
        })
    }

    /// 迭代图片 ID、哈希、路径
    pub fn images(&self) -> impl Iterator<Item = Result<(i32, Box<[u8]>, String)>> + '_ {
        self.db
            .iterator_cf_opt(
                &self.cf(ImageColumnFamily::ImageList),
                Self::read_opts(),
                IteratorMode::Start,
            )
            .map(|item| {
                let item = item?;
                let image_id = bytes_to_i32(item.1);
                let image_path = self
                    .db
                    .get_cf(&self.cf(ImageColumnFamily::IdToImage), image_id.to_le_bytes())?
                    .unwrap();
                Ok((image_id, item.0, String::from_utf8(image_path).unwrap()))
            })
    }

    /// 迭代图片 ID、特征 ID
    pub fn features(&self) -> impl Iterator<Item = (i32, u64)> + '_ {
        self.db
            .iterator_cf_opt(
                &self.cf(ImageColumnFamily::IdToFeature),
                Self::read_opts(),
                IteratorMode::Start,
            )
            .filter_map(|item| {
                item.and_then(|item| {
                    let image_id =
                        self.db.get_cf(&self.cf(ImageColumnFamily::IdToImageId), &item.0)?.unwrap();
                    Ok((bytes_to_i32(image_id), bytes_to_u64(item.0)))
                })
                .ok()
            })
    }

    pub fn total_features(&self) -> u64 {
        self.total_features.load(Ordering::SeqCst)
    }

    pub fn total_images(&self) -> u64 {
        self.total_images.load(Ordering::SeqCst)
    }

    fn read_opts() -> ReadOptions {
        let mut options = ReadOptions::default();
        options.set_verify_checksums(false);
        options.set_readahead_size(32 << 20);
        options
    }

    /// get column family handle
    fn cf(&self, cf: ImageColumnFamily) -> Arc<BoundColumnFamily> {
        // TODO: cache
        self.db.cf_handle(cf.as_ref()).unwrap()
    }
}
