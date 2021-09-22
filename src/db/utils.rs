use std::convert::TryInto;

use crate::db::database::ImageColumnFamily;
use crate::db::database::MetaData;
use rocksdb::{Error, Options, DB};

pub fn init_column_family(db: &DB) -> Result<(), Error> {
    let opts = &Options::default();
    db.create_cf(ImageColumnFamily::NewFeature, &opts)?;
    db.create_cf(ImageColumnFamily::IdToFeature, &opts)?;
    db.create_cf(ImageColumnFamily::IdToImageId, &opts)?;
    db.create_cf(ImageColumnFamily::IdToImage, &opts)?;
    db.create_cf(ImageColumnFamily::ImageList, &opts)?;
    db.create_cf(ImageColumnFamily::MetaData, &opts)?;

    let meta_data = db.cf_handle(ImageColumnFamily::MetaData.as_ref()).unwrap();
    db.put_cf(&meta_data, MetaData::TotalImages, 0u64.to_le_bytes())?;
    db.put_cf(&meta_data, MetaData::TotalFeatures, 0u64.to_le_bytes())?;

    Ok(())
}

pub fn bytes_to_u64<T: AsRef<[u8]>>(bytes: T) -> u64 {
    u64::from_le_bytes(
        bytes
            .as_ref()
            .try_into()
            .expect("byte cannot convert to u64"),
    )
}

pub fn bytes_to_i32<T: AsRef<[u8]>>(bytes: T) -> i32 {
    i32::from_le_bytes(
        bytes
            .as_ref()
            .try_into()
            .expect("byte cannot convert to i32"),
    )
}
