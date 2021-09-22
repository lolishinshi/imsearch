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
        if let Some(id) = self.db.find_image_id(hash.as_bytes())? {
            self.db.update_image_path(id, image_path.as_ref())?;
            return Ok(false);
        }

        let image = utils::imread(image_path.as_ref())?;
        let (_, descriptors) = utils::detect_and_compute(orb, &image)?;

        self.db
            .add_image(image_path.as_ref(), hash.as_bytes(), descriptors)
    }

    pub fn build_database(&self, chunk_size: usize) -> Result<()> {
        let mut index = self.get_index(false);
        let mut features = FeatureWithId::new();

        for (id, feature) in self.db.features(false) {
            features.add(id as i64, &*feature);
            if features.len() == chunk_size {
                log::debug!("Building index: {}", chunk_size);
                index.add_with_ids(features.features(), &features.ids());
                self.db.mark_as_trained(&features.ids_u64())?;
                features.clear();
            }
        }

        if features.len() != 0 {
            log::debug!("Building index: END");
            index.add_with_ids(features.features(), &features.ids());
            self.db.mark_as_trained(&features.ids_u64())?;
        }

        index.write_file(&*self.conf_dir.index().to_str().unwrap());

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
