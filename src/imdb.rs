use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use anyhow::{Result, anyhow};
use futures::prelude::*;
use indicatif::ProgressBar;
use log::{debug, info, warn};
use rayon::prelude::*;
use tokio::sync::Mutex;
use tokio::task::{block_in_place, spawn_blocking};

use crate::config::{ConfDir, ScoreType};
use crate::db::*;
use crate::faiss::{FaissIndex, FaissSearchParams, Neighbor};
use crate::hnsw::HNSW;
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
pub struct IMDBBuilder<const N: usize> {
    conf_dir: ConfDir,
    wal: bool,
    cache: bool,
    score_type: ScoreType,
    hash: Option<ImageHash>,
}

impl<const N: usize> IMDBBuilder<N> {
    pub fn new(conf_dir: ConfDir) -> Self {
        Self { conf_dir, wal: true, cache: false, score_type: ScoreType::Wilson, hash: None }
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

    pub fn hash(mut self, hash: ImageHash) -> Self {
        self.hash = Some(hash);
        self
    }

    pub async fn open(self) -> Result<IMDB<N>> {
        if !self.conf_dir.path().exists() {
            std::fs::create_dir_all(self.conf_dir.path())?;
        }
        let db = init_db(self.conf_dir.database(), self.wal).await?;

        if let Ok((image_count, vector_count)) = crud::get_count(&db).await {
            info!("图片数量  : {image_count}");
            info!("特征点数量：{vector_count}");
        }

        if let Ok(old_hash) = crud::guess_hash(&db).await {
            if self.hash.is_some() && self.hash.unwrap() != old_hash {
                return Err(anyhow!("哈希算法不一致"));
            }
        }

        let pindex = if self.hash == Some(ImageHash::Dhash) {
            let mut index = if self.conf_dir.path().join("phash.hnsw.graph").exists() {
                HNSW::load(self.conf_dir.path())?
            } else {
                HNSW::new()
            };

            if let Ok((count, _)) = crud::get_count(&db).await {
                if count != index.ntotal() as i64 {
                    warn!(
                        "phash 索引大小不一致（{} != {}），正在重新构建……",
                        count,
                        index.ntotal()
                    );
                    index = HNSW::new();
                    let hashes = crud::get_all_hash(&db).await?;
                    debug!("正在添加 {} 条向量到 phash 索引……", hashes.len());
                    hashes.into_par_iter().for_each(|(id, hash)| {
                        index.add(&hash, id as usize);
                    });
                    debug!("phash 索引添加完成，大小：{}", index.ntotal());
                }
            }
            Some(index)
        } else {
            None
        };

        let imdb = IMDB {
            db,
            conf_dir: self.conf_dir.clone(),
            total_vector_count: RwLock::new(vec![]),
            cache: self.cache,
            index: IndexManager::new(self.conf_dir),
            score_type: self.score_type,
            pindex: pindex.map(Arc::new),
        };

        imdb.load_total_vector_count().await?;
        Ok(imdb)
    }
}

pub struct IMDB<const N: usize> {
    conf_dir: ConfDir,
    db: Database,
    /// 是否使用缓存来加速 id 查询，会导致第一次查询速度变慢
    cache: bool,
    /// 每张图片特征点 ID 的累加数量，用于加速计算
    total_vector_count: RwLock<Vec<i64>>,
    /// 特征点索引
    index: IndexManager<N>,
    /// phash 索引
    pindex: Option<Arc<HNSW>>,
    /// 评分方式
    score_type: ScoreType,
}

impl<const N: usize> IMDB<N> {
    /// 添加图片到数据库
    pub async fn add_image(
        &self,
        filename: impl AsRef<str>,
        hash: &[u8],
        descriptors: &[[u8; N]],
    ) -> Result<i64> {
        let mut tx = self.db.begin_with("BEGIN IMMEDIATE").await?;
        let id = crud::add_image(&mut *tx, hash, filename.as_ref()).await?;
        crud::add_vector(&mut *tx, id, descriptors.as_flattened()).await?;
        crud::add_vector_stats(&mut *tx, id, descriptors.len() as i64).await?;
        // 尽快提交事务，避免锁住数据库
        tx.commit().await?;
        if let Some(index) = self.pindex.clone() {
            let hash = hash.to_vec();
            spawn_blocking(move || index.add(&hash, id as usize)).await?;
        }
        Ok(id)
    }

    /// 检查图片是否存在
    pub async fn check_hash(&self, hash: &[u8], distance: u32) -> Result<Option<i64>> {
        if let Some(id) = crud::check_image_hash(&self.db, hash).await? {
            return Ok(Some(id));
        }
        // 由于 phash 检查较慢，因此放到后面检查
        if let Some(index) = self.pindex.clone() {
            if distance > 0 {
                let hash = hash.to_vec();
                let result = spawn_blocking(move || index.search(&hash, 1)).await?;
                if !result.is_empty() && result[0].distance <= distance as f32 {
                    return Ok(Some(result[0].d_id as i64));
                }
            }
        }
        Ok(None)
    }

    /// 更新图片路径
    pub async fn update_image_path(&self, id: i64, path: &str) -> Result<()> {
        Ok(crud::update_image_path(&self.db, id, path).await?)
    }

    /// 追加图片路径
    pub async fn append_image_path(&self, id: i64, path: &str) -> Result<bool> {
        Ok(crud::append_image_path(&self.db, id, path).await?)
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
    pub async fn export(&self, count: Option<usize>) -> Result<Vec<[u8; N]>> {
        let count = count.unwrap_or(usize::MAX);
        let mut arr = vec![];
        let mut i = 0;
        let records = crud::get_vectors(&self.db, count, 0).await?;
        for record in records {
            let (vectors, remainder) = record.vector.as_chunks::<N>();
            assert!(remainder.is_empty());
            arr.extend_from_slice(vectors);
            i += 1;
            if i >= count {
                break;
            }
        }

        Ok(arr)
    }

    /// 获取用于搜索的索引
    pub fn get_index(&self, mmap: bool) -> Result<FaissIndex<N>> {
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
        index: Arc<FaissIndex<N>>,
        descriptors: &[Vec<[u8; N]>],
        knn: usize,
        max_distance: u32,
        max_result: usize,
        params: FaissSearchParams,
    ) -> Result<Vec<Vec<(f32, String)>>> {
        let mat = descriptors.concat();

        info!(
            "对 {} 组 {} 条向量搜索 {} 个最近邻, {:?}",
            descriptors.len(),
            mat.len(),
            knn,
            params
        );
        if mat.is_empty() {
            return Ok(vec![vec![]; descriptors.len()]);
        }

        let mut instant = Instant::now();

        let neighbors = spawn_blocking(move || index.search(&mat, knn, Some(params))).await?;
        debug!("搜索耗时    ：{}ms", instant.elapsed().as_millis());
        instant = Instant::now();

        let mut result = vec![];
        let mut res = &*neighbors;
        let mut cur;
        for item in descriptors {
            (cur, res) = res.split_at(item.len());
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

impl<const N: usize> IMDB<N> {
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

            let mut images = Vec::with_capacity(chunk.len());
            let mut ids = Vec::with_capacity(chunk.len() * N);
            let mut features = Vec::with_capacity(chunk.len() * N);

            for record in chunk {
                for (i, feature) in record.vector.chunks_exact(N).enumerate() {
                    features.push(feature.try_into().unwrap());
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

    pub fn save_phash_index(&self) -> Result<()> {
        if let Some(index) = &self.pindex {
            debug!("正在保存 phash 索引，大小：{}……", index.ntotal());
            index.write(self.conf_dir.path())?;
        }
        Ok(())
    }
}
