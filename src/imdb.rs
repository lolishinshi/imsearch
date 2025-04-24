use std::collections::HashMap;
use std::str::FromStr;
use std::time::Instant;

use crate::config::ConfDir;
use crate::db::*;
use crate::index::{FaissIndex, FaissSearchParams};
use crate::matrix::{Matrix, Matrix2D};
use crate::utils;
use anyhow::Result;
use futures::{StreamExt, TryStreamExt};
use log::{debug, info};
use ndarray::prelude::*;
use opencv::core::{Mat, MatTraitConstManual};
use tokio::task::block_in_place;

#[derive(Debug, Clone)]
pub struct IMDB {
    conf_dir: ConfDir,
    db: Database,
}

impl IMDB {
    /// 创建或打开一个新的 IMDB 实例
    ///
    /// # Arguments
    ///
    /// * `conf_dir` - 配置目录
    pub async fn new(conf_dir: ConfDir) -> Result<Self> {
        if !conf_dir.path().exists() {
            std::fs::create_dir_all(conf_dir.path())?;
        }
        let db = init_db(conf_dir.database()).await?;
        Ok(Self { db, conf_dir })
    }

    /// 添加图片到数据库
    pub async fn add_image(
        &self,
        filename: impl AsRef<str>,
        hash: &[u8],
        descriptors: Mat,
    ) -> Result<bool> {
        let mut tx = self.db.begin_with("BEGIN IMMEDIATE").await?;
        let id = crud::add_image(&mut *tx, hash, filename.as_ref()).await?;
        crud::add_vector(&mut *tx, id, descriptors.data_typed::<u8>()?).await?;
        crud::add_vector_stats(&mut *tx, id, descriptors.height() as i64).await?;
        tx.commit().await?;
        Ok(true)
    }

    /// 检查图片是否存在
    pub async fn check_hash(&self, hash: &[u8]) -> Result<bool> {
        Ok(crud::check_image_hash(&self.db, hash).await?)
    }

    /// 构建索引
    ///
    /// # Arguments
    ///
    /// * `chunk_size` - 每次添加到索引的**图片**数量
    pub async fn build_index(&self, chunk_size: usize) -> Result<()> {
        let mut index = self.get_index(false, SearchStrategy::Heap);

        if !index.is_trained() {
            panic!("该索引未训练！");
        }

        let stream = crud::get_vectors(&self.db).await?;
        let mut stream = stream.chunks(chunk_size);

        while let Some(chunk) = stream.next().await {
            let mut features = FeatureWithId::new();
            let mut images = vec![];

            for record in chunk {
                let record = record?;
                for (i, feature) in record.vector.chunks(32).enumerate() {
                    if feature.len() != 32 {
                        panic!("特征长度不正确");
                    }
                    features.add(record.total_vector_count - i as i64, feature);
                }
                images.push(record.id);
            }

            info!("构建索引: {} + {}", index.ntotal(), features.len());

            tokio::task::block_in_place(|| {
                index.add_with_ids(features.features(), features.ids());
                index.write_file(&self.conf_dir.index_tmp());
            });

            crud::set_indexed_batch(&self.db, &images).await?;
            std::fs::rename(&self.conf_dir.index_tmp(), self.conf_dir.index())?;
        }

        Ok(())
    }

    /// 清除索引缓存
    ///
    /// # Arguments
    ///
    /// * `unindexed` - 是否清除未索引的缓存
    pub async fn clear_cache(&self, unindexed: bool) -> Result<()> {
        crud::delete_vectors(&self.db, true).await?;
        if unindexed {
            crud::delete_vectors(&self.db, false).await?;
        }
        Ok(())
    }

    /// 导出所有特征到一个二维数组
    pub async fn export(&self) -> Result<Array2<u8>> {
        let mut arr = Array2::zeros((0, 32));
        let mut stream = crud::get_vectors(&self.db).await?;
        while let Some(record) = stream.try_next().await? {
            for vector in record.vector.chunks(32) {
                if vector.len() != 32 {
                    panic!("特征长度不正确");
                }
                let tmp = ArrayView::from(vector);
                arr.push(Axis(0), tmp)?;
            }
        }
        Ok(arr)
    }

    /// 获取索引
    ///
    /// # Arguments
    ///
    /// * `mmap` - 是否使用 mmap 模式加载索引
    /// * `strategy` - 搜索策略
    pub fn get_index(&self, mmap: bool, strategy: SearchStrategy) -> FaissIndex {
        let index_file = &*self.conf_dir.index();
        let mut index = if index_file.exists() {
            if !mmap {
                debug!("正在加载索引 {}", index_file.display());
            }
            let index = FaissIndex::from_file(index_file.to_str().unwrap(), mmap);
            debug!("已添加特征点 : {}", index.ntotal());
            debug!("倒排列表数量 : {}", index.nlist());
            debug!("不平衡度     : {}", index.imbalance_factor());
            index.print_stats();
            index
        } else {
            panic!("索引文件不存在，请先构建索引");
        };

        index.set_use_heap(false);
        index.set_per_invlit_search(false);

        match strategy {
            SearchStrategy::PerInvlist => index.set_per_invlit_search(true),
            SearchStrategy::Heap => index.set_use_heap(true),
            SearchStrategy::Count => { /* 全部关闭就是计数排序 */ }
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
    pub async fn search<M: Matrix>(
        &self,
        index: &FaissIndex,
        descriptors: M,
        knn: usize,
        max_distance: u32,
        params: FaissSearchParams,
    ) -> Result<Vec<(f32, String)>> {
        debug!("对 {} 条向量搜索 {} 个最近邻, {:?}", descriptors.height(), knn, params);
        let mut instant = Instant::now();
        let mut counter = HashMap::new();

        // TODO: 这里应该用 spawn_blocking 还是 block_in_place 呢？
        let all_neighbors = block_in_place(|| index.search(&descriptors, knn, params));
        debug!("搜索耗时      ：{}ms", instant.elapsed().as_millis());
        instant = Instant::now();

        for neighbors in all_neighbors {
            for neighbor in neighbors {
                if neighbor.distance > max_distance as i32 {
                    continue;
                }
                let path = crud::get_image_path_by_vector_id(&self.db, neighbor.index).await?;
                counter
                    .entry(path)
                    .or_insert_with(Vec::new)
                    .push(1. - neighbor.distance as f32 / 256.);
            }
        }
        debug!("检索数据库耗时：{:.2}ms", instant.elapsed().as_millis());
        instant = Instant::now();

        let mut results = counter
            .into_iter()
            .map(|(path, scores)| (100. * utils::wilson_score(&*scores), path))
            .collect::<Vec<_>>();
        results.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        debug!("计算并排序耗时: {:.2}ms", instant.elapsed().as_millis());

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

    pub fn features(&self) -> &Matrix2D {
        &self.1
    }

    pub fn ids(&self) -> &[i64] {
        &self.0
    }
}

/// Faiss 搜索策略
#[derive(Debug, Clone, Copy)]
pub enum SearchStrategy {
    /// 倒序列表优先
    PerInvlist,
    /// 使用堆排序
    Heap,
    /// 使用计数排序
    Count,
}

impl FromStr for SearchStrategy {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "per_invlist" => Ok(SearchStrategy::PerInvlist),
            "heap" => Ok(SearchStrategy::Heap),
            "count" => Ok(SearchStrategy::Count),
            _ => Err(anyhow::anyhow!("invalid search strategy")),
        }
    }
}
