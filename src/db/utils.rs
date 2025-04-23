use std::convert::TryInto;

use crate::db::database::ImageColumnFamily;
use crate::db::database::MetaData;
use rocksdb::{
    BlockBasedIndexType, BlockBasedOptions, Cache, DBCompressionType, Error, Options, DB,
};

pub fn default_options() -> Options {
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_size(16 << 10);
    block_opts.set_block_cache(&Cache::new_lru_cache(1 << 30));
    block_opts.set_index_type(BlockBasedIndexType::TwoLevelIndexSearch);
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_metadata_block_size(4 << 10);
    block_opts.set_cache_index_and_filter_blocks(true);
    block_opts.set_pin_top_level_index_and_filter(true);
    block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);

    let mut options = Options::default();
    options.set_block_based_table_factory(&block_opts);
    options.set_compression_type(DBCompressionType::Zstd);
    options.increase_parallelism(num_cpus::get() as i32);
    options.create_if_missing(true);
    options.set_keep_log_file_num(10);
    options.set_level_compaction_dynamic_level_bytes(true);
    options.set_max_total_wal_size(512 << 20);
    options.set_compaction_readahead_size(16 << 20);
    options.set_skip_stats_update_on_db_open(true);

    options
}

pub fn init_column_family(db: &DB) -> Result<(), Error> {
    let opts = default_options();
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
