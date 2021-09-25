use std::collections::HashMap;

use crate::config::ConfDir;
use crate::db::ImageDB;
use crate::index::FaissIndex;
use crate::matrix::{Matrix, Matrix2D};
use crate::slam3_orb::Slam3ORB;
use crate::utils;
use crate::utils::{hash_file, wilson_score};
use anyhow::Result;
use itertools::Itertools;
use log::{debug, info};

pub struct IMDB {
    conf_dir: ConfDir,
    db: ImageDB,
}

impl IMDB {
    pub fn new(conf_dir: ConfDir, read_only: bool) -> Result<Self> {
        let db = ImageDB::open(&conf_dir, read_only)?;
        Ok(Self { db, conf_dir })
    }

    pub fn add_image<S: AsRef<str>>(&self, image_path: S, orb: &mut Slam3ORB) -> Result<bool> {
        let hash = hash_file(image_path.as_ref())?;
        if let Some(id) = self.db.find_image_id(hash.as_bytes())? {
            self.db.update_image_path(id, image_path.as_ref())?;
            return Ok(false);
        }

        let image = utils::imread(image_path.as_ref())?;
        let (_, descriptors) = utils::detect_and_compute(orb, &image)?;

        self.db
            .add_image(image_path.as_ref(), hash.as_bytes(), descriptors)
    }

    pub fn train_index(&mut self) {
        let mut _index = self.get_index(false);
    }

    pub fn build_index(&self, chunk_size: usize) -> Result<()> {
        let mut index = self.get_index(false);
        let mut features = FeatureWithId::new();

        if !index.is_trained() {
            panic!("index hasn't been trained");
        }

        let mut tmp_file = self.conf_dir.index();
        tmp_file.set_extension(".tmp");

        let add_index = |index: &mut FaissIndex, features: &FeatureWithId| -> Result<()> {
            index.add_with_ids(features.features(), &features.ids());
            index.write_file(&*tmp_file.to_str().unwrap());
            self.db.mark_as_trained(&features.ids_u64())?;
            std::fs::rename(&tmp_file, self.conf_dir.index())?;
            Ok(())
        };

        for (id, feature) in self.db.features(false) {
            features.add(id as i64, &*feature);
            if features.len() == chunk_size {
                info!("Building index: {} + {}", index.ntotal(), chunk_size);
                add_index(&mut index, &features)?;
                features.clear();
            }
        }

        if features.len() != 0 {
            info!("Building index: END");
            add_index(&mut index, &features)?;
        }

        Ok(())
    }

    fn create_index(&self) -> FaissIndex {
        let total_features = self.db.total_features();
        let desc = match total_features {
            // 0 ~ 1M
            0..=1000000 => {
                let k = 4 * (total_features as f32).sqrt() as u32;
                format!("BIVF{}", k)
            }
            // 1M ~ 10M
            1000001..=10000000 => String::from("BIVF65536"),
            // 10M ~ 100M
            10000001..=100000000 => String::from("BIVF262144"),
            // 100M ~ 10G
            100000001..=10000000000 => String::from("BIVF1048576"),
            _ => unimplemented!(),
        };
        debug!("creating index with {}", desc);
        FaissIndex::new(256, &desc)
    }

    pub fn get_index(&self, mmap: bool) -> FaissIndex {
        let index_file = &*self.conf_dir.index();
        if index_file.exists() {
            if !mmap {
                debug!("reading index from {}", index_file.display());
            }
            let index = FaissIndex::from_file(index_file.to_str().unwrap(), mmap);
            debug!("indexed features: {}", index.ntotal());
            index
        } else {
            self.create_index()
        }
    }

    pub fn search_des<M: Matrix>(
        &self,
        index: &FaissIndex,
        descriptors: M,
        knn: usize,
        max_distance: u32,
    ) -> Result<Vec<(f32, String)>> {
        let mut counter = HashMap::new();

        for neighbors in index.search(&descriptors, knn) {
            for neighbor in neighbors {
                if neighbor.distance > max_distance {
                    continue;
                }
                let image_index = self.db.find_image_path(neighbor.index as u64)?;
                counter
                    .entry(image_index)
                    .or_insert(vec![])
                    .push(1. - neighbor.distance as f32 / 256.);
            }
        }

        let mut results = counter
            .into_iter()
            // TODO: score type
            .map(|(image, scores)| (100. * wilson_score(&*scores), image))
            .collect::<Vec<_>>();
        results.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        Ok(results)
    }

    pub fn search<S: AsRef<str>>(
        &self,
        index: &FaissIndex,
        image_path: S,
        orb: &mut Slam3ORB,
        knn: usize,
        max_distance: u32,
    ) -> Result<Vec<(f32, String)>> {
        debug!("searching {} nearest neighbors", knn);
        let image = utils::imread(image_path.as_ref())?;
        let (_, descriptors) = utils::detect_and_compute(orb, &image)?;

        self.search_des(index, descriptors, knn, max_distance)
    }
}

#[derive(Debug)]
struct FeatureWithId(Vec<i64>, Matrix2D);

impl FeatureWithId {
    pub fn new() -> Self {
        Self(vec![], Matrix2D::new(32))
    }

    pub fn add(&mut self, id: i64, feature: &[u8]) {
        self.0.push(id);
        self.1.push(feature);
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn clear(&mut self) {
        self.0.clear();
        self.1.clear();
    }

    pub fn features(&self) -> &Matrix2D {
        &self.1
    }

    pub fn ids(&self) -> &[i64] {
        &self.0
    }

    pub fn ids_u64(&self) -> Vec<u64> {
        self.0.iter().map(|&n| n as u64).collect_vec()
    }
}
