use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::TryInto;
use std::path::Path;

use crate::config::OPTS;
use crate::knn::KnnSearcher;
use crate::slam3_orb::Slam3ORB;
use crate::utils;
use crate::utils::TimeMeasure;
use anyhow::Result;
use itertools::Itertools;
use opencv::core;
use opencv::prelude::*;
use rocksdb::{IteratorMode, ReadOptions, WriteBatch, DB};

thread_local! {
    static ORB: RefCell<Slam3ORB> = RefCell::new(Slam3ORB::from(&*OPTS));
}

pub struct ImageDb {
    image_db: DB,
    feature_db: DB,
}

impl ImageDb {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conf_dir = path.as_ref().to_path_buf();
        let image_db = DB::open_default(conf_dir.join("image.db"))?;
        let feature_db = DB::open_default(conf_dir.join("feature.db"))?;
        let result = Self {
            image_db,
            feature_db,
        };
        result.database_update()?;
        Ok(result)
    }

    fn database_update(&self) -> Result<()> {
        if self.image_db.get(b"VERSION")?.is_none() {
            let total_feature = self.feature_db.iterator(IteratorMode::Start).count() as i32;
            self.image_db
                .put(b"TOTAL_FEATURE", total_feature.to_le_bytes())?;
            self.image_db.put(b"VERSION", 1i32.to_le_bytes())?;
        }
        Ok(())
    }

    fn fetch_and_add(&self, key: &[u8], count: i32) -> Result<i32> {
        let old = self
            .image_db
            .get_pinned(key)?
            .map(|slice| i32::from_le_bytes(slice.as_ref().try_into().unwrap()))
            .unwrap_or(0);
        self.image_db.put(key, (old + count).to_le_bytes())?;
        Ok(old)
    }

    fn search_image_id_by_path(&self, path: &str) -> Result<Option<i32>> {
        Ok(self
            .image_db
            .get_pinned(path.as_bytes())?
            .map(|slice| i32::from_le_bytes(slice.as_ref().try_into().unwrap())))
    }

    pub fn search_image_path_by_id(&self, id: i32) -> Result<String> {
        Ok(self
            .image_db
            .get(id.to_le_bytes())
            .map(|slice| String::from_utf8(slice.unwrap()).unwrap())?)
    }

    pub fn search_image_id_by_des(&self, des: &Mat) -> Result<i32> {
        Ok(self
            .feature_db
            .get(des.data_typed::<u8>()?)
            .map(|slice| i32::from_le_bytes(slice.unwrap().try_into().unwrap()))?)
    }

    pub fn add<S: AsRef<str>>(&self, path: S) -> Result<()> {
        if self.search_image_id_by_path(path.as_ref())?.is_some() {
            return Ok(());
        }

        let img = utils::imread(path.as_ref())?;
        let (_, des) = ORB.with(|f| utils::detect_and_compute(&mut *f.borrow_mut(), &img))?;

        let id = self.fetch_and_add(b"TOTAL_IMAGE", 1)?;
        self.image_db
            .put(path.as_ref().as_bytes(), id.to_le_bytes())?;
        self.image_db
            .put(id.to_le_bytes(), path.as_ref().as_bytes())?;

        let mut batch = WriteBatch::default();
        for i in 0..des.rows() {
            let row = des.row(i)?;
            let data = row.data_typed::<u8>()?;
            batch.put(data, id.to_le_bytes());
        }
        self.fetch_and_add(b"TOTAL_FEATURE", des.rows())?;
        self.feature_db.write(batch)?;

        Ok(())
    }

    pub fn get_all_descriptors(&self) -> Result<Vec<Mat>> {
        let mut readopts = ReadOptions::default();
        readopts.set_verify_checksums(false);
        readopts.set_readahead_size(32 << 20); // 32MiB

        Ok(self
            .feature_db
            .iterator_opt(IteratorMode::Start, readopts)
            .chunks(OPTS.batch_size)
            .into_iter()
            .map(|chunk| iter_to_mat(chunk, OPTS.batch_size as i32, 32))
            .collect::<Result<Vec<_>, _>>()?)
    }

    pub fn search<S: AsRef<str>>(&self, path: S) -> Result<Vec<(f32, String)>> {
        let img = utils::imread(path.as_ref())?;
        let (_, query_des) = ORB.with(|f| utils::detect_and_compute(&mut *f.borrow_mut(), &img))?;

        let mut results = HashMap::new();

        let mut readopts = ReadOptions::default();
        readopts.set_verify_checksums(false);
        readopts.set_readahead_size(32 << 20); // 32MiB

        let mut time = TimeMeasure::new();

        for chunk in &self
            .feature_db
            .iterator_opt(IteratorMode::Start, readopts)
            .chunks(OPTS.batch_size)
        {
            let train_des =
                time.measure("reading", || iter_to_mat(chunk, OPTS.batch_size as i32, 32))?;

            let mut flann = time.measure("building", || {
                let mut v = KnnSearcher::new(
                    &train_des,
                    OPTS.lsh_table_number,
                    OPTS.lsh_key_size,
                    OPTS.lsh_probe_level,
                    OPTS.search_checks,
                );
                v.build();
                v
            });

            let matches = time.measure("searching", || flann.knn_search(&query_des, OPTS.knn_k));

            time.measure("recording", || -> Result<()> {
                for match_ in matches {
                    for point in match_ {
                        if point.distance > OPTS.distance {
                            continue;
                        }
                        let des = train_des.row(point.index as i32)?;
                        let id = self.search_image_id_by_des(&des)?;
                        *results.entry(id).or_insert(0.) +=
                            255.0 / point.distance.max(1) as f32 / OPTS.knn_k as f32;
                    }
                }
                Ok(())
            })?;
        }

        let results = time.measure("recoding", || -> Result<_> {
            let mut results = results
                .iter()
                .map(|(image_id, score)| {
                    self.search_image_path_by_id(*image_id)
                        .map(|image_path| (*score, image_path))
                })
                .collect::<Result<Vec<_>>>()?;
            results.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            Ok(results)
        })?;

        let total_image = self.fetch_and_add(b"TOTAL_IMAGE", 0)?;
        let total_feature = self.fetch_and_add(b"TOTAL_FEATURE", 0)?;

        log::debug!("Total image  : {}", total_image);
        log::debug!("Total feature: {}", total_feature);
        log::debug!("Reading data : {:.2}s", time.0["reading"].as_secs_f32());
        log::debug!("Building LSH : {:.2}s", time.0["building"].as_secs_f32());
        log::debug!("Searching KNN: {:.2}s", time.0["searching"].as_secs_f32());
        log::debug!("Recording    : {:.2}s", time.0["recording"].as_secs_f32());

        Ok(results.into_iter().take(OPTS.output_count).collect())
    }

    pub fn total_feature(&self) -> Result<i32> {
        self.fetch_and_add(b"TOTAL_FEATURE", 0)
    }
}

fn iter_to_mat(
    iter: impl Iterator<Item = (Box<[u8]>, Box<[u8]>)>,
    row: i32,
    col: i32,
) -> opencv::Result<Mat> {
    let mut matrix =
        Mat::new_rows_cols_with_default(row, col, core::CV_8U, core::Scalar::default())?;
    let mut idx = 0;
    for line in iter {
        let row = matrix.at_row_mut(idx as i32)?;
        row.copy_from_slice(&line.0);
        idx += 1;
    }
    matrix.row_range(&core::Range::new(0, idx)?)
}
