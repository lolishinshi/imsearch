use anyhow::Result;
use rocksdb::{BoundColumnFamily, IteratorMode, Options, WriteBatch, DB};
use std::collections::HashMap;
use std::convert::TryInto;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[derive(Debug, Hash, Eq, PartialEq)]
enum ImageColumnFamily {
    IdToFeature,
    IdToImage,
    ImageList,
    MetaData,
}

enum MetaData {
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
    total_images: AtomicUsize,
    total_features: AtomicUsize,
}

impl ImageDB {
    /// Open the database, will create if not exists
    pub fn open<P: AsRef<Path>>(path: P, read_only: bool) -> Result<Self> {
        check_db_update(path.as_ref());

        let mut options = Options::default();
        options.increase_parallelism(num_cpus::get() as i32);
        options.set_keep_log_file_num(100);

        let db = match read_only {
            true => DB::open_for_read_only(&options, path, false)?,
            false => DB::open(&options, path)?,
        };

        let meta_data = db.cf_handle(ImageColumnFamily::MetaData.as_ref()).unwrap();
        let total_images = db.get_cf(&meta_data, MetaData::TotalImages)?.unwrap();
        let total_features = db.get_cf(&meta_data, MetaData::TotalFeatures)?.unwrap();

        // meta_data borrows db here
        drop(meta_data);

        Ok(Self {
            db,
            total_images: AtomicUsize::new(bytes_to_usize(total_images)),
            total_features: AtomicUsize::new(bytes_to_usize(total_features)),
        })
    }

    // add an image and its features to database
    pub fn add_image<S, I, T>(&self, path: S, features: I) -> Result<()>
    where
        S: AsRef<str>,
        I: IntoIterator<Item = T>,
        T: AsRef<[u8]>,
    {
        let mut batch = WriteBatch::default();

        let mut feature_cnt = 0;
        for feature in features {
            let id = self.total_features.fetch_add(1, Ordering::SeqCst);
            let feature = feature.as_ref();
            batch.put_cf(&self.cf(ImageColumnFamily::IdToFeature), id.to_le_bytes(), feature);
            batch.put_cf(&self.cf(ImageColumnFamily::IdToImage), id.to_le_bytes(), path.as_ref());
            feature_cnt += 1;
        }
        batch.put_cf(&self.cf(ImageColumnFamily::ImageList), path.as_ref().as_bytes(), []);

        self.db.write(batch)?;

        let total_images = self.total_images.fetch_add(1, Ordering::SeqCst);
        let total_features = self.total_features.load(Ordering::SeqCst);

        self.db.put_cf(&self.cf(ImageColumnFamily::MetaData), MetaData::TotalImages.as_ref(), total_images.to_le_bytes())?;
        self.db.put_cf(&self.cf(ImageColumnFamily::MetaData), MetaData::TotalFeatures.as_ref(), total_features.to_le_bytes())?;

        Ok(())
    }

    // get column family handle
    fn cf(&self, cf: ImageColumnFamily) -> Arc<BoundColumnFamily> {
        // TODO: cache
        self.db.cf_handle(cf.as_ref()).unwrap()
    }
}

/// check whether the database needs update
fn check_db_update(path: &Path) -> Result<()> {
    let version_file = path.join("version");

    // v1, v2 => v3
    if !version_file.exists() {
        println!("START UPGRADING (2 -> 3) AFTER 10 SECS!!!");
        thread::sleep(Duration::from_secs(10));
        let image_db = DB::open_default(path.join("image.db"))?;
        let features_db = DB::open_default(path.join("features.db"))?;

        let new_db = DB::open_default("database")?;
        new_db.create_cf(ImageColumnFamily::IdToFeature, &Options::default())?;
        new_db.create_cf(ImageColumnFamily::IdToImage, &Options::default())?;

        let index_feature = new_db
            .cf_handle(ImageColumnFamily::IdToFeature.as_ref())
            .unwrap();
        let index_image = new_db
            .cf_handle(ImageColumnFamily::IdToImage.as_ref())
            .unwrap();
        let image_list = new_db
            .cf_handle(ImageColumnFamily::ImageList.as_ref())
            .unwrap();
        let meta_data = new_db
            .cf_handle(ImageColumnFamily::MetaData.as_ref())
            .unwrap();

        let mut total_features = 0usize;
        for (idx, data) in features_db.iterator(IteratorMode::Start).enumerate() {
            print!("\r{}", idx);
            let idx = idx.to_le_bytes();
            let feature = data.0;
            let image_id = data.1;
            let image_path = image_db.get(image_id)?.unwrap();

            new_db.put_cf(&index_feature, idx, feature)?;
            new_db.put_cf(&index_image, idx, &image_path)?;
            new_db.put_cf(&image_list, image_path, [])?;

            total_features += 1;
        }
        let total_images = image_db.iterator(IteratorMode::Start).count();

        new_db.put_cf(
            &meta_data,
            MetaData::TotalFeatures,
            total_features.to_le_bytes(),
        )?;
        new_db.put_cf(
            &meta_data,
            MetaData::TotalImages,
            total_images.to_le_bytes(),
        )?;
    }

    Ok(std::fs::write(version_file, "3")?)
}

fn bytes_to_usize<T: AsRef<[u8]>>(bytes: T) -> usize {
    usize::from_le_bytes(
        bytes
            .as_ref()
            .try_into()
            .expect("byte cannot convert to usize"),
    )
}
