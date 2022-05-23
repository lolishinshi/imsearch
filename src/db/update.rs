use super::database::{ImageColumnFamily, MetaData};
use crate::config::ConfDir;
use crate::db::utils::{default_options, init_column_family};
use crate::utils::hash_file;
use anyhow::Result;
use log::info;
use rocksdb::{IteratorMode, DB};

/// check whether the database needs update
pub fn check_db_update(path: &ConfDir) -> Result<()> {
    info!("checking database update");
    let version_file = path.version();

    // v1, v2 => v3
    if !version_file.exists() && path.path().join("image.db").exists() {
        // There is bug with v1 & v2 database, no need to merge
        // println!("START UPGRADING (2 -> 3) AFTER 10 SECS!!!");
        // thread::sleep(Duration::from_secs(10));
        // update_from_2_to_3(path)?;
        std::fs::write(path.version(), "3")?;
    }
    if !version_file.exists() {
        std::fs::create_dir_all(path.path())?;
        std::fs::write(path.version(), "3")?;
    }

    let version = std::fs::read_to_string(version_file)?;

    match version.as_str() {
        "3" => {}
        _ => {}
    }

    // init
    if !path.database().exists() {
        let db = DB::open_default(path.database())?;
        init_column_family(&db)?;
    }

    Ok(())
}

#[allow(unused)]
fn update_from_2_to_3(path: &ConfDir) -> Result<()> {
    let mut opts = default_options();

    let image_db = DB::open_for_read_only(&opts, path.path().join("image.db"), true)?;
    let features_db = DB::open_for_read_only(&opts, path.path().join("feature.db"), true)?;
    let new_db = DB::open(&opts, path.database())?;

    init_column_family(&new_db)?;

    let new_feature = new_db
        .cf_handle(ImageColumnFamily::NewFeature.as_ref())
        .unwrap();
    let index_image = new_db
        .cf_handle(ImageColumnFamily::IdToImageId.as_ref())
        .unwrap();
    let image_list = new_db
        .cf_handle(ImageColumnFamily::ImageList.as_ref())
        .unwrap();
    let meta_data = new_db
        .cf_handle(ImageColumnFamily::MetaData.as_ref())
        .unwrap();
    let image_id = new_db
        .cf_handle(ImageColumnFamily::IdToImage.as_ref())
        .unwrap();

    let mut total_features = 0u64;
    for (idx, data) in features_db.iterator(IteratorMode::Start).enumerate() {
        // features_db contains: feature([u8; 32]) => image_id(i32)
        print!("\r{}", idx);
        let idx = idx.to_le_bytes();

        new_db.put_cf(&new_feature, idx, data.0)?;
        new_db.put_cf(&index_image, idx, data.1)?;

        total_features += 1;
    }

    println!();

    let mut total_images = 0u64;
    for (idx, data) in image_db.iterator(IteratorMode::Start).enumerate() {
        // image_db contains:
        //  image_id(i32)      => image_path(String)
        //  image_path(String) => image_id(u32)      [skip]
        if data.1.len() == 4 {
            continue;
        }
        print!("\r{}", idx);
        let hash = hash_file(String::from_utf8(data.1.to_vec())?)?;
        new_db.put_cf(&image_list, hash.as_bytes(), [])?;
        new_db.put_cf(&image_id, data.0, data.1)?;
        total_images += 1;
    }

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

    Ok(())
}
