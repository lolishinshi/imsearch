use crate::db::database::ImageColumnFamily;
use crate::db::database::MetaData;
use rocksdb::{Error, Options, DB};

pub fn init_column_family(db: &DB) -> Result<(), Error> {
    let opts = &Options::default();
    db.create_cf(ImageColumnFamily::NewFeature, &opts)?;
    db.create_cf(ImageColumnFamily::IdToFeature, &opts)?;
    db.create_cf(ImageColumnFamily::IdToImage, &opts)?;
    db.create_cf(ImageColumnFamily::ImageList, &opts)?;
    db.create_cf(ImageColumnFamily::MetaData, &opts)?;

    let meta_data = db.cf_handle(ImageColumnFamily::MetaData.as_ref()).unwrap();
    db.put_cf(&meta_data, MetaData::TotalImages, 0u64.to_le_bytes())?;
    db.put_cf(&meta_data, MetaData::TotalFeatures, 0u64.to_le_bytes())?;

    Ok(())
}
