use std::collections::HashMap;
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
        let db = init_db(conf_dir.database(), true).await?;
        if let Ok((image_count, vector_count)) = crud::get_count(&db).await {
            info!("图片数量  : {}", image_count);
            info!("特征点数量：{}", vector_count);
        }
        Ok(Self { db, conf_dir })
    }

    pub async fn new_without_wal(conf_dir: ConfDir) -> Result<Self> {
        let db = init_db(conf_dir.database(), false).await?;
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
    /// * `mmap` - 合并阶段是否使用 mmap
    pub async fn build_index(&self, chunk_size: usize, mmap: bool) -> Result<()> {
        let stream = crud::get_vectors(&self.db).await?;
        let mut stream = stream.chunks(chunk_size);

        while let Some(chunk) = stream.next().await {
            let mut index = self.get_index_template();
            if !index.is_trained() {
                panic!("该索引未训练！");
            }

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

            info!("构建索引: {}", features.len());

            block_in_place(|| {
                index.add_with_ids(features.features(), features.ids());
                index.write_file(self.conf_dir.index_tmp());
            });

            crud::set_indexed_batch(&self.db, &images).await?;
            std::fs::rename(self.conf_dir.index_tmp(), self.conf_dir.index_sub())?;
        }

        info!("合并索引中……");
        let mut index = self.get_index(mmap);
        let mut files = vec![];
        for i in 1.. {
            let index_file = self.conf_dir.index_sub_with(i);
            if !index_file.exists() {
                break;
            }
            let sub_index = FaissIndex::from_file(index_file.to_str().unwrap(), false);
            info!("合并索引: {} + {}", index.ntotal(), sub_index.ntotal());
            block_in_place(|| index.merge_from(&sub_index, 0));
            files.push(index_file);
        }

        block_in_place(|| index.write_file(self.conf_dir.index()));

        for file in files {
            std::fs::remove_file(file)?;
        }

        Ok(())
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

    /// 获取模板索引
    fn get_index_template(&self) -> FaissIndex {
        let index_file = self.conf_dir.index_template();
        if index_file.exists() {
            let index = FaissIndex::from_file(index_file.to_str().unwrap(), false);
            index
        } else {
            panic!("模板索引不存在，请先训练索引，并保存为 {}", index_file.display());
        }
    }

    /// 获取索引
    ///
    /// # Arguments
    ///
    /// * `mmap` - 是否使用 mmap 模式加载索引
    /// * `strategy` - 搜索策略
    pub fn get_index(&self, mmap: bool) -> FaissIndex {
        let index_file = self.conf_dir.index();
        let index = if index_file.exists() {
            if !mmap {
                info!("正在加载索引 {}", index_file.display());
            }
            let index = FaissIndex::from_file(index_file.to_str().unwrap(), mmap);
            info!("faiss 版本   : {}", index.faiss_version());
            info!("已添加特征点 : {}", index.ntotal());
            info!("倒排列表数量 : {}", index.nlist());
            info!("不平衡度     : {}", index.imbalance_factor());
            index.print_stats();
            index
        } else {
            self.get_index_template()
        };

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
        info!("对 {} 条向量搜索 {} 个最近邻, {:?}", descriptors.height(), knn, params);
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
            .map(|(path, scores)| (100. * utils::wilson_score(&scores), path))
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
