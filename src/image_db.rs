use std::cell::RefCell;
use std::path::Path;

use crate::config::{OPTS, THREAD_NUM};
use crate::flann::Flann;
use crate::slam3_orb::Slam3ORB;
use crate::utils;
use anyhow::Result;
use itertools::Itertools;
use opencv::core;
use opencv::prelude::*;
use rocksdb::{IteratorMode, ReadOptions, WriteBatch, DB};
use std::convert::TryInto;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

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

    pub fn get_all_des(&self) -> Result<Mat> {
        let mut readopts = ReadOptions::default();
        readopts.set_verify_checksums(false);
        readopts.set_readahead_size(32 << 20); // 32MiB
        let iter = self.feature_db.iterator_opt(IteratorMode::Start, readopts);
        Ok(iter_to_mat(iter)?)
    }

    pub fn search<S: AsRef<str>>(&self, path: S) -> Result<Vec<(f32, String)>> {
        let img = utils::imread(path.as_ref())?;
        let (_, query_des) = ORB.with(|f| utils::detect_and_compute(&mut *f.borrow_mut(), &img))?;

        let results = dashmap::DashMap::new();
        let max_threads = AtomicUsize::new(*THREAD_NUM);

        let mut readopts = ReadOptions::default();
        readopts.set_verify_checksums(false);
        readopts.set_readahead_size(32 << 20); // 32MiB

        crossbeam_utils::thread::scope(|scope| -> Result<()> {
            for chunk in &self
                .feature_db
                .iterator_opt(IteratorMode::Start, readopts)
                .chunks(OPTS.batch_size)
            {
                while max_threads.load(Ordering::SeqCst) == 0 {
                    std::thread::sleep(Duration::from_millis(50));
                }
                max_threads.fetch_sub(1, Ordering::SeqCst);

                let train_des = iter_to_mat(chunk)?;
                let query_des = query_des.clone();
                let results = &results;
                let max_threads = &max_threads;

                scope.spawn(move |_| -> Result<()> {
                    let mut flann = Flann::new(
                        &train_des,
                        OPTS.lsh_table_number,
                        OPTS.lsh_key_size,
                        OPTS.lsh_probe_level,
                        OPTS.search_checks,
                    )?;
                    let matches = flann.knn_search(&query_des, OPTS.knn_k)?;

                    for match_ in matches.into_iter() {
                        for point in match_.into_iter() {
                            let des = train_des.row(point.index as i32)?;
                            let id = self.search_image_id_by_des(&des)?;
                            *results.entry(id).or_insert(0.) +=
                                point.distance_squared / 500.0 / OPTS.knn_k as f32;
                        }
                    }

                    max_threads.fetch_add(1, Ordering::SeqCst);

                    Ok(())
                });
            }
            Ok(())
        })
        .unwrap()?;

        let mut results = results
            .iter()
            .map(|entry| {
                self.search_image_path_by_id(*entry.key())
                    .map(|k| (*entry.value(), k))
            })
            .collect::<Result<Vec<_>>>()?;
        results.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        Ok(results.into_iter().take(OPTS.output_count).collect())
    }
}

fn iter_to_mat(iter: impl Iterator<Item = (Box<[u8]>, Box<[u8]>)>) -> opencv::Result<Mat> {
    let mut matrix = Mat::new_nd_vec_with_default(
        &core::Vector::from_iter([OPTS.batch_size as i32, 32]),
        core::CV_8U,
        core::Scalar::default(),
    )?;
    let mut idx = 0;
    for line in iter {
        let row = matrix.at_row_mut(idx as i32)?;
        row.copy_from_slice(&line.0);
        idx += 1;
    }
    matrix.row_range(&core::Range::new(0, idx)?)
}
