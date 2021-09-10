use std::cell::RefCell;
use std::path::Path;

use crate::config::OPTS;
use crate::slam3_orb::Slam3ORB;
use crate::utils;
use anyhow::Result;
use itertools::Itertools;
use opencv::features2d;
use opencv::prelude::*;
use opencv::types;
use rocksdb::{IteratorMode, ReadOptions, WriteBatch, DB};
use std::convert::TryInto;

thread_local! {
    static ORB: RefCell<Slam3ORB> = RefCell::new(Slam3ORB::from(&*OPTS));
    static FLANN: RefCell<features2d::FlannBasedMatcher> = RefCell::new(features2d::FlannBasedMatcher::from(&*OPTS));
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
        Ok(Self {
            image_db,
            feature_db,
        })
    }

    fn incr_total_image(&self) -> Result<i32> {
        let old = self
            .image_db
            .get_pinned(b"TOTAL_IMAGE")?
            .map(|slice| i32::from_le_bytes(slice.as_ref().try_into().unwrap()))
            .unwrap_or(0);
        self.image_db.put(b"TOTAL_IMAGE", (old + 1).to_le_bytes())?;
        Ok(old)
    }

    fn search_image_id_by_path(&self, path: &str) -> Result<Option<i32>> {
        Ok(self
            .image_db
            .get_pinned(path.as_bytes())?
            .map(|slice| i32::from_le_bytes(slice.as_ref().try_into().unwrap())))
    }

    fn search_image_path_by_id(&self, id: i32) -> Result<String> {
        Ok(self
            .image_db
            .get(id.to_le_bytes())
            .map(|slice| String::from_utf8(slice.unwrap()).unwrap())?)
    }

    fn search_image_id_by_des(&self, des: &Mat) -> Result<i32> {
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

        let id = self.incr_total_image()?;
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
        self.feature_db.write(batch)?;

        Ok(())
    }

    pub fn search<S: AsRef<str>>(&self, path: S) -> Result<Vec<(usize, String)>> {
        let img = utils::imread(path.as_ref())?;
        let (_, query_des) = ORB.with(|f| utils::detect_and_compute(&mut *f.borrow_mut(), &img))?;

        let results = dashmap::DashMap::new();

        let mut readopts = ReadOptions::default();
        readopts.set_verify_checksums(false);
        readopts.set_readahead_size(32 << 20); // 32MiB

        crossbeam_utils::thread::scope(|scope| -> Result<()> {
            for chunk in &self
                .feature_db
                .iterator_opt(IteratorMode::Start, readopts)
                .chunks(OPTS.batch_size)
            {
                let raw_data = chunk.map(|f| f.0).collect::<Vec<_>>();
                let query_des = query_des.clone();
                let results = &results;

                scope.spawn(move |_| -> Result<()> {
                    let train_des = Mat::from_slice_2d(&raw_data)?;
                    drop(raw_data);

                    let mut matches = types::VectorOfVectorOfDMatch::default();
                    let mask = Mat::default();

                    FLANN.with(|f| {
                        f.borrow().knn_train_match(
                            &query_des,
                            &train_des,
                            &mut matches,
                            OPTS.knn_k,
                            &mask,
                            false,
                        )
                    })?;

                    for match_ in matches.iter() {
                        for point in match_.iter() {
                            let des = train_des.row(point.train_idx)?;
                            let id = self.search_image_id_by_des(&des)?;
                            *results.entry(id).or_insert(0) += 1;
                        }
                    }
                    Ok(())
                });
            }
            Ok(())
        }).unwrap()?;

        let mut results = results
            .iter()
            .filter(|entry| *entry.value() > 2)
            .map(|entry| self.search_image_path_by_id(*entry.key()).map(|k| (*entry.value(), k)))
            .collect::<Result<Vec<_>>>()?;
        results.sort_unstable_by_key(|v| std::cmp::Reverse(v.0));

        Ok(results.into_iter().take(OPTS.output_count).collect())
    }
}
