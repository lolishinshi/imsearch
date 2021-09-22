use std::collections::HashMap;

use crate::config::ConfDir;
use crate::db::ImageDB;
use crate::knn::FaissSearcher;
use crate::matrix::Matrix2D;
use crate::slam3_orb::Slam3ORB;
use crate::utils;
use crate::utils::{hash_file, wilson_score};
use anyhow::Result;
use itertools::Itertools;

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
        if self.db.image_exists(hash.as_bytes())? {
            return Ok(false);
        }

        let image = utils::imread(image_path.as_ref())?;
        let (_, descriptors) = utils::detect_and_compute(orb, &image)?;

        self.db
            .add_image(image_path.as_ref(), hash.as_bytes(), descriptors)
    }

    pub fn build_database(&self, chunk_size: usize) -> Result<()> {
        let index_file = &*self.conf_dir.index();

        let mut index = FaissSearcher::from_file(index_file.to_str().unwrap(), false);
        let mut features = FeatureWithId::new();

        for (id, feature) in self.db.features(false) {
            features.add(id as i64, &*feature);
            if features.len() == chunk_size {
                log::debug!("Building index: {}", chunk_size);
                index.add_with_ids(&features.1, &features.0);
                self.db
                    .mark_as_trained(&features.0.iter().map(|&n| n as u64).collect_vec())?;
                features.clear();
            }
        }

        if features.len() != 0 {
            log::debug!("Building index: END");
            index.add_with_ids(&features.1, &features.0);
        }

        index.write_file(index_file.to_str().unwrap());

        Ok(())
    }

    pub fn get_index(&self, mmap: bool) -> FaissSearcher {
        let index_file = &*self.conf_dir.index();
        match index_file.exists() {
            true => FaissSearcher::from_file(index_file.to_str().unwrap(), mmap),
            _ => FaissSearcher::new(256, "IVF1048576"),
        }
    }

    pub fn search<S: AsRef<str>>(
        &self,
        index: &FaissSearcher,
        image_path: S,
        orb: &mut Slam3ORB,
        knn: usize,
        max_distance: u32,
    ) -> Result<Vec<(f32, String)>> {
        let image = utils::imread(image_path.as_ref())?;
        let (_, descriptors) = utils::detect_and_compute(orb, &image)?;

        let mut counter = HashMap::new();

        for neighbors in index.search(&descriptors, knn) {
            for neighbor in neighbors {
                if neighbor.distance > max_distance {
                    continue;
                }
                counter
                    .entry(neighbor.index)
                    .or_insert(vec![])
                    .push(1. - neighbor.distance as f32 / 256.);
            }
        }

        let mut results = counter
            .iter()
            .map(|(idx, scores)| {
                self.db
                    .find_image(*idx as u64)
                    .map(|path| (100. * wilson_score(scores), path))
            })
            .collect::<Result<Vec<_>, _>>()?;
        results.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        Ok(results)
    }
}

struct FeatureWithId(pub Vec<i64>, pub Matrix2D);

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
}
