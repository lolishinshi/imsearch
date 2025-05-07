use std::collections::HashMap;
use std::env::set_current_dir;
use std::sync::RwLock;
use std::time::Instant;

use anyhow::Result;
use futures::prelude::*;
use indicatif::{ProgressBar, ProgressIterator};
use log::{Level, debug, info, log_enabled, warn};
use ndarray::prelude::*;
use opencv::core::{Mat, MatTraitConstManual};
use opencv::prelude::*;
use tokio::sync::Mutex;
use tokio::task::block_in_place;

use crate::config::ConfDir;
use crate::db::*;
use crate::faiss::{FaissIndex, FaissOnDiskInvLists, FaissSearchParams, Neighbor};
use crate::utils::{self, pb_style};

#[derive(Debug, Clone)]
pub struct IMDBBuilder {
    conf_dir: ConfDir,
    wal: bool,
    mmap: bool,
    cache: bool,
    ondisk: bool,
}

impl IMDBBuilder {
    pub fn new(conf_dir: ConfDir) -> Self {
        Self { conf_dir, wal: true, mmap: true, cache: false, ondisk: false }
    }

    /// 数据库是否开启 WAL，开启会影响删除
    pub fn wal(mut self, wal: bool) -> Self {
        self.wal = wal;
        self
    }

    /// 是否使用 mmap 模式加载索引
    pub fn mmap(mut self, mmap: bool) -> Self {
        self.mmap = mmap;
        self
    }

    /// 是否使用缓存来加速 id 查询，会导致第一次查询速度变慢
    pub fn cache(mut self, cache: bool) -> Self {
        self.cache = cache;
        self
    }

    pub fn ondisk(mut self, ondisk: bool) -> Self {
        self.ondisk = ondisk;
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
            conf_dir: self.conf_dir,
            total_vector_count: RwLock::new(vec![]),
            mmap: RwLock::new(self.mmap),
            cache: self.cache,
            ondisk: RwLock::new(self.ondisk),
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
    /// 是否使用 mmap 模式加载索引
    mmap: RwLock<bool>,
    /// 是否使用 OnDiskInvertedLists 格式合并索引
    ondisk: RwLock<bool>,
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

    /// 直接在内存中构建索引
    pub async fn build_index_without_split(&self, chunk_size: usize) -> Result<()> {
        info!("正在计算未索引的图片数量……");
        let image_unindexed = crud::count_image_unindexed(&self.db).await?;

        info!("正在加载索引……");
        let mut index = self.get_index();
        if !index.is_trained() {
            return Err(anyhow::anyhow!("索引未训练"));
        }

        let pb = ProgressBar::new(image_unindexed as u64).with_style(pb_style());
        pb.set_message("正在构建索引……");

        let mut processed = 0;

        while let Ok(chunk) = crud::get_vectors_unindexed(&self.db, chunk_size, 0).await {
            if chunk.is_empty() {
                break;
            }

            let mut images = vec![];
            let mut ids = vec![];
            let mut features = Array2::zeros((0, 32));
            features.reserve_rows(chunk.len() * 500)?;

            for record in &chunk {
                for (i, feature) in record.vector.chunks(32).enumerate() {
                    features.push_row(ArrayView::from(feature))?;
                    ids.push(record.total_vector_count - i as i64);
                }
                images.push(record.id);
            }

            pb.set_message("正在构建索引……");
            block_in_place(|| index.add_with_ids(&features, &ids));
            pb.set_message("正在保存已构建部分……");
            block_in_place(|| index.write_file(self.conf_dir.index_tmp()));

            crud::set_indexed_batch(&self.db, &images).await?;
            std::fs::rename(self.conf_dir.index_tmp(), self.conf_dir.index())?;

            processed += images.len();
            pb.set_position(processed as u64);
        }

        pb.finish_with_message("索引构建完成");

        Ok(())
    }

    /// 构建索引
    pub async fn build_index(&self, chunk_size: usize, no_split: bool) -> Result<()> {
        if no_split {
            return self.build_index_without_split(chunk_size).await;
        }

        info!("正在计算未索引的图片数量……");
        let image_unindexed = crud::count_image_unindexed(&self.db).await?;

        let pb = ProgressBar::new(image_unindexed as u64).with_style(pb_style());
        pb.set_message("正在构建索引...");

        let mut processed = 0;
        while let Ok(chunk) = crud::get_vectors_unindexed(&self.db, chunk_size, 0).await {
            if chunk.is_empty() {
                break;
            }
            let mut index = self.get_index_template();
            if !index.is_trained() {
                panic!("该索引未训练！");
            }

            let mut images = vec![];
            let mut ids = vec![];
            let mut features = Array2::zeros((0, 32));
            features.reserve_rows(chunk.len() * 500)?;

            for record in chunk {
                for (i, feature) in record.vector.chunks(32).enumerate() {
                    features.push_row(ArrayView::from(feature))?;
                    ids.push(record.total_vector_count - i as i64);
                }
                images.push(record.id);
            }

            block_in_place(|| {
                index.add_with_ids(&features, &ids);
                index.write_file(self.conf_dir.index_tmp());
            });

            crud::set_indexed_batch(&self.db, &images).await?;
            std::fs::rename(self.conf_dir.index_tmp(), self.conf_dir.index_sub())?;

            processed += images.len();
            pb.set_position(processed as u64);
        }

        pb.finish_with_message("索引构建完成");

        if *self.ondisk.read().unwrap() {
            if self.conf_dir.ondisk_ivf().exists() {
                warn!("OnDisk 索引已存在，忽略 --on-disk 选项");
                self.merge_index_on_memory()?;
            } else {
                self.merge_index_on_disk()?;
            }
        } else {
            self.merge_index_on_memory()?;
        }

        Ok(())
    }

    /// 合并索引
    fn merge_index_on_memory(&self) -> Result<()> {
        info!("在内存中合并所有索引……");
        let mut index = self.get_index();
        let sub_index_files = self.conf_dir.index_sub_all();
        let mut files = vec![];
        for index_file in sub_index_files.iter().progress_with_style(pb_style()) {
            if !index_file.exists() {
                break;
            }
            let sub_index = FaissIndex::from_file(index_file.to_str().unwrap(), false);
            block_in_place(|| index.merge_from(&sub_index, 0));
            files.push(index_file);
        }

        info!("保存索引……");
        block_in_place(|| index.write_file(self.conf_dir.index()));

        for file in files {
            std::fs::remove_file(file)?;
        }

        Ok(())
    }

    /// 使用 OnDiskInvertedLists 合并索引
    ///
    /// 注意这种模式下，必须合并到一个空索引中
    fn merge_index_on_disk(&self) -> Result<()> {
        info!("在磁盘上合并所有索引……");

        let mut invfs = vec![];
        let mut files = vec![];
        for index_file in self.conf_dir.index_sub_all() {
            if !index_file.exists() {
                break;
            }
            let mut sub_index = FaissIndex::from_file(index_file.to_str().unwrap(), true);
            invfs.push(sub_index.invlists());
            sub_index.set_own_invlists(false);
            files.push(index_file);
        }

        let mut index = self.get_index_template();

        let mut invlists = FaissOnDiskInvLists::new(
            index.nlist(),
            index.code_size() as usize,
            self.conf_dir.ondisk_ivf().to_str().unwrap(),
        );

        info!("合并倒排列表……");
        let ntotal = invlists.merge_from_multiple(&invfs, false, true);

        index.replace_invlists(invlists, true);
        index.set_ntotal(ntotal as i64);

        info!("保存索引……");
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

    /// 获取模板索引
    pub fn get_index_template(&self) -> FaissIndex {
        let index_file = self.conf_dir.index_template();
        if index_file.exists() {
            let index = FaissIndex::from_file(index_file.to_str().unwrap(), false);
            index
        } else {
            panic!("模板索引不存在，请先训练索引，并保存为 {}", index_file.display());
        }
    }

    /// 获取索引
    pub fn get_index(&self) -> FaissIndex {
        let index_file = self.conf_dir.index();
        let ivf_file = self.conf_dir.ondisk_ivf();

        let index = if index_file.exists() {
            let mut mmap = *self.mmap.read().unwrap();
            if ivf_file.exists() {
                if mmap {
                    warn!("OnDisk 倒排无法使用 mmap 模式加载");
                }
                // NOTE: 此处切换路径的原因是，faiss 会根据「当前路径」而不是 index 所在路径来加载倒排列表
                set_current_dir(self.conf_dir.path()).unwrap();
                mmap = false;
            }
            info!("正在加载索引 {}, mmap: {}", index_file.display(), mmap);
            let index = FaissIndex::from_file(index_file.to_str().unwrap(), mmap);
            info!("faiss 版本   : {}", index.faiss_version());
            info!("已添加特征点 : {}", index.ntotal());
            info!("倒排列表数量 : {}", index.nlist());
            info!("不平衡度     : {}", index.imbalance_factor());
            if log_enabled!(Level::Debug) {
                index.print_stats();
            }
            index
        } else {
            self.get_index_template()
        };

        index
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
        index: &FaissIndex,
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

        // TODO: 这里应该用 spawn_blocking 还是 block_in_place 呢？
        let neighbors = block_in_place(|| index.search(&mat, knn, params));
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
        let mut result = counter
            .into_iter()
            .map(|(id, scores)| (100. * utils::wilson_score(&scores), id))
            .collect::<Vec<_>>();
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

    pub fn set_mmap(&self, mmap: bool) {
        *self.mmap.write().unwrap() = mmap;
    }

    pub fn set_ondisk(&self, ondisk: bool) {
        *self.ondisk.write().unwrap() = ondisk;
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
