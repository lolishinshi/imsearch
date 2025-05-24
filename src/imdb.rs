use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use anyhow::Result;
use futures::prelude::*;
use indicatif::ProgressBar;
use log::{debug, info};
use ndarray::prelude::*;
use opencv::core::{Mat, MatTraitConstManual};
use opencv::prelude::*;
use tokio::sync::Mutex;
use tokio::task::{block_in_place, spawn_blocking};

use crate::config::{ConfDir, ScoreType};
use crate::db::*;
use crate::faiss::{FaissIndex, FaissSearchParams, Neighbor};
use crate::index::IndexManager;
use crate::utils::{self, ImageHash, pb_style};

#[derive(Debug, Clone)]
pub struct BuildOptions {
    pub on_disk: bool,
    pub batch_size: usize,
    pub no_merge: bool,
    pub ef_search: usize,
}

#[derive(Debug, Clone)]
pub struct IMDBBuilder {
    conf_dir: ConfDir,
    wal: bool,
    cache: bool,
    score_type: ScoreType,
}

impl IMDBBuilder {
    pub fn new(conf_dir: ConfDir) -> Self {
        Self { conf_dir, wal: true, cache: false, score_type: ScoreType::Wilson }
    }

    /// 数据库是否开启 WAL，开启会影响删除
    pub fn wal(mut self, wal: bool) -> Self {
        self.wal = wal;
        self
    }

    /// 是否使用缓存来加速 id 查询，会导致第一次查询速度变慢
    pub fn cache(mut self, cache: bool) -> Self {
        self.cache = cache;
        self
    }

    pub fn score_type(mut self, score_type: ScoreType) -> Self {
        self.score_type = score_type;
        self
    }

    pub async fn open(self) -> Result<IMDB> {
        if !self.conf_dir.path().exists() {
            std::fs::create_dir_all(self.conf_dir.path())?;
        }
        let db = init_db(self.conf_dir.database(), self.wal).await?;
        if let Ok((image_count, vector_count)) = crud::get_count(&db).await {
            info!("图片数量  : {}", image_count);
            info!("特征点数量：{}", vector_count);
        }
        let imdb = IMDB {
            db,
            conf_dir: self.conf_dir.clone(),
            total_vector_count: RwLock::new(vec![]),
            cache: self.cache,
            index: IndexManager::new(self.conf_dir),
            score_type: self.score_type,
        };
        imdb.load_total_vector_count().await?;
        Ok(imdb)
    }
}

#[derive(Debug)]
pub struct IMDB {
    conf_dir: ConfDir,
    db: Database,
    /// 是否使用缓存来加速 id 查询，会导致第一次查询速度变慢
    cache: bool,
    /// 每张图片特征点 ID 的累加数量，用于加速计算
    total_vector_count: RwLock<Vec<i64>>,
    index: IndexManager,
    score_type: ScoreType,
}

impl IMDB {
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
        crud::add_vector_stats(&mut *tx, id, descriptors.rows() as i64).await?;
        tx.commit().await?;
        Ok(true)
    }

    /// 检查图片是否存在
    pub async fn check_hash(&self, hash: &[u8]) -> Result<bool> {
        Ok(crud::check_image_hash(&self.db, hash).await?)
    }

    /// 更新图片路径
    pub async fn update_image_path(&self, hash: &[u8], path: &str) -> Result<()> {
        Ok(crud::update_image_path(&self.db, hash, path).await?)
    }

    /// 追加图片路径
    pub async fn append_image_path(&self, hash: &[u8], path: &str) -> Result<()> {
        Ok(crud::append_image_path(&self.db, hash, path).await?)
    }

    /// 猜测先前使用的哈希算法
    pub async fn guess_hash(&self) -> Option<ImageHash> {
        crud::guess_hash(&self.db).await.ok()
    }

    /// 清除索引缓存
    pub async fn clear_cache(&self, all: bool) -> Result<()> {
        if all {
            crud::delete_vectors_all(&self.db).await?;
        } else {
            crud::delete_vectors(&self.db).await?;
        }
        Ok(())
    }

    /// 导出若干图片的特征点到一个二维数组
    pub async fn export(&self, count: Option<usize>) -> Result<Array2<u8>> {
        let count = count.unwrap_or(usize::MAX);
        let mut arr = Array2::zeros((0, 32));
        let mut i = 0;
        let records = crud::get_vectors(&self.db, count, 0).await?;
        for record in records {
            for vector in record.vector.chunks(32) {
                if vector.len() != 32 {
                    panic!("特征长度不正确");
                }
                let tmp = ArrayView::from(vector);
                arr.push(Axis(0), tmp)?;
            }
            i += 1;
            if i >= count {
                break;
            }
        }

        Ok(arr)
    }

    /// 获取用于搜索的索引
    pub fn get_index(&self, mmap: bool) -> Result<FaissIndex> {
        self.index.get_aggregate_index(mmap)
    }

    /// 在索引中搜索多组描述符，返回 Vec<Vec<(分数, 图片路径)>>
    ///
    /// # Arguments
    ///
    /// * `index` - 索引
    /// * `descriptors` - 多组描述符
    /// * `knn` - KNN 搜索的数量
    /// * `max_distance` - 最大距离
    /// * `max_result` - 最大结果数量
    /// * `params` - 搜索参数
    pub async fn search(
        &self,
        index: Arc<FaissIndex>,
        descriptors: &[Mat],
        knn: usize,
        max_distance: u32,
        max_result: usize,
        params: FaissSearchParams,
    ) -> Result<Vec<Vec<(f32, String)>>> {
        let mut mat = Mat::default();
        for des in descriptors {
            mat.push_back(des).unwrap();
        }

        info!(
            "对 {} 组 {} 条向量搜索 {} 个最近邻, {:?}",
            descriptors.len(),
            mat.rows(),
            knn,
            params
        );
        if mat.rows() == 0 {
            return Ok(vec![vec![]; descriptors.len()]);
        }

        let mut instant = Instant::now();

        let neighbors = spawn_blocking(move || index.search(&mat, knn, params)).await?;
        debug!("搜索耗时    ：{}ms", instant.elapsed().as_millis());
        instant = Instant::now();

        let mut result = vec![];
        let mut res = &*neighbors;
        let mut cur;
        for item in descriptors {
            (cur, res) = res.split_at(item.rows() as usize);
            result.push(self.process_neighbor_group(cur, max_distance as i32, max_result).await?);
        }

        debug!("处理结果耗时：{:.2}ms", instant.elapsed().as_millis());

        Ok(result)
    }

    /// 处理一个搜索结果分组
    async fn process_neighbor_group(
        &self,
        neighbors: &[Vec<Neighbor>],
        max_distance: i32,
        max_result: usize,
    ) -> Result<Vec<(f32, String)>> {
        let counter = Mutex::new(HashMap::new());

        // 遍历所有结果，并统计每个图片 ID 的出现次数
        stream::iter(neighbors.iter().flatten())
            .filter(|neighbor| future::ready(neighbor.distance <= max_distance))
            .for_each(|neighbor| async {
                if let Ok(id) = self.find_image_id(neighbor.index).await {
                    let mut counter = counter.lock().await;
                    counter
                        .entry(id)
                        .or_insert_with(Vec::new)
                        .push(1. - neighbor.distance as f32 / 256.);
                }
            })
            .await;

        // 计算得分，并取前 10 个结果
        let counter = counter.into_inner();
        let mut result = match self.score_type {
            ScoreType::Wilson => counter
                .into_iter()
                .map(|(id, scores)| (100. * utils::wilson_score(&scores), id))
                .collect::<Vec<_>>(),
            ScoreType::Count => counter
                .into_iter()
                .map(|(id, scores)| (scores.len() as f32, id))
                .collect::<Vec<_>>(),
        };
        result.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        result.truncate(max_result);

        // 查询实际的图片路径
        let futures = result
            .into_iter()
            .map(|(score, id)| async move {
                crud::get_image_path(&self.db, id).await.map(|path| (score, path))
            })
            .collect::<Vec<_>>();
        let result = futures::future::try_join_all(futures).await?;
        Ok(result)
    }

    /// 根据向量 ID 查找图片 ID
    async fn find_image_id(&self, id: i64) -> Result<i64> {
        if !self.cache {
            Ok(crud::get_image_id_by_vector_id(&self.db, id).await?)
        } else {
            let lock = self.total_vector_count.read().unwrap();
            let index = lock.partition_point(|&x| x < id) + 1;
            Ok(index as i64)
        }
    }

    pub async fn load_total_vector_count(&self) -> Result<()> {
        if self.cache {
            info!("正在加载图片 ID 缓存……");
            let vec = crud::get_all_total_vector_count(&self.db).await?;
            let mut lock = self.total_vector_count.write().unwrap();
            *lock = vec;
        }
        Ok(())
    }
}

impl IMDB {
    /// 分段构建索引再进行合并
    pub async fn build_index(&self, options: BuildOptions) -> Result<()> {
        info!("正在计算未索引的图片数量……");
        let image_unindexed = crud::count_image_unindexed(&self.db).await?;

        let pb = ProgressBar::new(image_unindexed as u64).with_style(pb_style());
        pb.set_message("正在构建索引...");

        let mut processed = 0;
        while let Ok(chunk) = crud::get_vectors_unindexed(&self.db, options.batch_size, 0).await {
            if chunk.is_empty() {
                break;
            }
            let mut index = self.index.get_template_index()?;
            index.set_ef_search(options.ef_search);
            assert!(index.is_trained(), "该索引未训练！");

            let mut images = vec![];
            let mut ids = vec![];
            let mut features = Array2::zeros((0, 32));
            features.reserve_rows(chunk.len() * 500)?;

            for record in chunk {
                for (i, feature) in record.vector.chunks(32).enumerate() {
                    features.push_row(ArrayView::from(feature))?;
                    // total_vector_count 记录了截止到这张图片的特征点数量累加和
                    // 因此使用它减去特征点本身的序号，就可以得到一个唯一的特征点 ID
                    ids.push(record.total_vector_count - i as i64);
                }
                images.push(record.id);
            }

            block_in_place(|| {
                index.add_with_ids(&features, &ids)?;
                index.write_file(self.conf_dir.next_sub_index())
            })?;

            crud::set_indexed_batch(&self.db, &images).await?;

            processed += images.len();
            pb.set_position(processed as u64);
        }

        pb.finish_with_message("索引构建完成");

        if options.no_merge {
            Ok(())
        } else if options.on_disk {
            block_in_place(|| self.index.merge_index_on_disk())
        } else {
            block_in_place(|| self.index.merge_index_on_memory())
        }
    }
}
