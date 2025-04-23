use std::collections::HashMap;
use std::time::Instant;

use crate::config::ConfDir;
use crate::db::ImageDB;
use crate::index::{FaissIndex, FaissSearchParams};
use crate::matrix::{Matrix, Matrix2D};
use crate::slam3_orb::Slam3ORB;
use crate::utils;
use crate::utils::{hash_file, wilson_score};
use anyhow::Result;
use itertools::Itertools;
use log::{debug, info};
use ndarray::prelude::*;

pub struct IMDB {
    conf_dir: ConfDir,
    db: ImageDB,
}

impl IMDB {
    /// 创建或打开一个新的 IMDB 实例
    ///
    /// # Arguments
    ///
    /// * `conf_dir` - 配置目录
    /// * `read_only` - 是否只读模式
    pub fn new(conf_dir: ConfDir, read_only: bool) -> Result<Self> {
        let db = ImageDB::open(&conf_dir, read_only)?;
        Ok(Self { db, conf_dir })
    }

    /// 添加图片到数据库
    ///
    /// # Arguments
    ///
    /// * `image_path` - 图片路径
    /// * `orb` - ORB 特征点检测器
    pub fn add_image<S: AsRef<str>>(&self, image_path: S, orb: &mut Slam3ORB) -> Result<bool> {
        let hash = hash_file(image_path.as_ref())?;
        if let Some(id) = self.db.find_image_id_by_hash(hash.as_bytes())? {
            self.db.update_image_path(id, image_path.as_ref())?;
            return Ok(false);
        }

        let image = utils::imread(image_path.as_ref())?;
        let (_, descriptors) = utils::detect_and_compute(orb, &image)?;

        self.db
            .add_image(image_path.as_ref(), hash.as_bytes(), descriptors)
    }

    /// 构建索引
    ///
    /// # Arguments
    ///
    /// * `chunk_size` - 每次添加到索引的特征数量
    /// * `start` - 开始的特征 ID
    /// * `end` - 结束的特征 ID
    pub fn build_index(
        &self,
        chunk_size: usize,
        start: Option<u64>,
        end: Option<u64>,
    ) -> Result<()> {
        let mut index = self.get_index(false, false);
        let mut features = FeatureWithId::new();

        if !index.is_trained() {
            panic!("index hasn't been trained");
        }

        let mut tmp_file = self.conf_dir.index();
        tmp_file.set_extension(".tmp");

        let add_index = |index: &mut FaissIndex, features: &FeatureWithId| -> Result<()> {
            index.add_with_ids(features.features(), features.ids());
            index.write_file(&*tmp_file.to_str().unwrap());
            self.db.mark_as_indexed(&features.ids_u64())?;
            std::fs::rename(&tmp_file, self.conf_dir.index())?;
            Ok(())
        };

        // TODO: 丢弃迭代器以允许 RocksDB 从硬盘上删除不需要的数据
        for item in self.db.features(false) {
            let (id, feature) = item?;
            if start.is_some() && id < start.unwrap() {
                continue;
            }
            if end.is_some() && id >= end.unwrap() {
                continue;
            }
            features.add(id as i64, &*feature);
            if features.len() == chunk_size {
                info!("Building index: {} + {}", index.ntotal(), chunk_size);
                add_index(&mut index, &features)?;
                features.clear();
            }
        }

        if !features.len() != 0 {
            info!("Building index: END");
            add_index(&mut index, &features)?;
        }

        Ok(())
    }

    /// 将索引标记为已索引
    ///
    /// # Arguments
    ///
    /// * `max_feature_id` - 最大特征 ID
    /// * `chunk_size` - 每次标记的特征数量
    pub fn mark_as_indexed(&self, max_feature_id: u64, chunk_size: usize) -> Result<()> {
        let mut idx = vec![];

        for item in self.db.features(false) {
            let (id, _) = item?;
            if id >= max_feature_id {
                continue;
            }
            idx.push(id);
            if idx.len() == chunk_size {
                info!("mark as indexed: {}", chunk_size);
                self.db.mark_as_indexed(&idx)?;
                idx.clear();
            }
        }

        if !idx.is_empty() {
            info!("mark as indexed: {}", idx.len());
            self.db.mark_as_indexed(&idx)?;
        }

        Ok(())
    }

    /// 清除索引缓存
    ///
    /// # Arguments
    ///
    /// * `unindexed` - 是否清除未索引的缓存
    pub fn clear_cache(&self, unindexed: bool) -> Result<()> {
        self.db.clear_cache(true)?;
        if unindexed {
            self.db.clear_cache(false)?;
        }
        Ok(())
    }

    /// 导出所有特征到一个二维数组
    pub fn export(&self) -> Result<Array2<u8>> {
        let mut arr = Array2::zeros((0, 32));
        for item in self.db.features(false) {
            let (_, feature) = item?;
            let tmp = ArrayView::from(&feature);
            arr.push(Axis(0), tmp)?;
        }
        Ok(arr)
    }

    /// 创建索引
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

    /// 获取索引，如果索引文件存在，则从文件中加载索引，否则创建一个新的索引
    ///
    /// # Arguments
    ///
    /// * `mmap` - 是否使用 mmap 模式加载索引
    /// * `per_invlist_search` - 是否使用 per_invlist_search 搜索策略
    pub fn get_index(&self, mmap: bool, per_invlist_search: bool) -> FaissIndex {
        let index_file = &*self.conf_dir.index();
        let mut index = if index_file.exists() {
            if !mmap {
                debug!("reading index from {}", index_file.display());
            }
            let index = FaissIndex::from_file(index_file.to_str().unwrap(), mmap);
            debug!("indexed features: {}", index.ntotal());
            index
        } else {
            self.create_index()
        };

        if per_invlist_search {
            index.set_per_invlit_search(true);
            index.set_use_heap(false);
        } else {
            index.set_use_heap(true);
        }

        index
    }

    /// 在索引中搜索描述符，返回 Vec<(分数, 图片路径)>
    ///
    /// # Arguments
    ///
    /// * `index` - 索引
    /// * `descriptors` - 描述符
    /// * `knn` - KNN 搜索的数量
    /// * `max_distance` - 最大距离
    /// * `params` - 搜索参数
    pub fn search_des<M: Matrix>(
        &self,
        index: &FaissIndex,
        descriptors: M,
        knn: usize,
        max_distance: u32,
        params: FaissSearchParams,
    ) -> Result<Vec<(f32, String)>> {
        debug!("searching {} nearest neighbors", knn);
        let instant = Instant::now();
        let mut counter = HashMap::new();

        for neighbors in index.search(&descriptors, knn, params) {
            for neighbor in neighbors {
                if neighbor.distance > max_distance {
                    continue;
                }
                let image_index = self.db.find_image_path(neighbor.index as u64)?;
                counter
                    .entry(image_index)
                    .or_insert_with(Vec::new)
                    .push(1. - neighbor.distance as f32 / 256.);
            }
        }

        let mut results = counter
            .into_iter()
            // TODO: score type
            .map(|(image, scores)| (100. * wilson_score(&*scores), image))
            .collect::<Vec<_>>();
        results.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        debug!("search time: {:.2}s", instant.elapsed().as_secs_f32());

        Ok(results)
    }

    /// 在索引中搜索图片
    ///
    /// # Arguments
    ///
    /// * `index` - 索引
    /// * `image_path` - 图片路径
    /// * `orb` - ORB 特征点检测器
    /// * `knn` - KNN 搜索的数量
    /// * `max_distance` - 最大距离
    /// * `params` - 搜索参数
    pub fn search<S: AsRef<str>>(
        &self,
        index: &FaissIndex,
        image_path: S,
        orb: &mut Slam3ORB,
        knn: usize,
        max_distance: u32,
        params: FaissSearchParams,
    ) -> Result<Vec<(f32, String)>> {
        let image = utils::imread(image_path.as_ref())?;
        let (_, descriptors) = utils::detect_and_compute(orb, &image)?;

        self.search_des(index, descriptors, knn, max_distance, params)
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
