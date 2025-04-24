use crate::matrix::Matrix;
use faiss_sys::*;
use itertools::Itertools;
use log::{debug, info};
use std::ffi::CString;
use std::mem::MaybeUninit;
use std::ptr;
use std::time::Instant;

/// Faiss 搜索结果
pub struct Neighbor {
    /// 向量在索引中的 ID
    pub index: usize,
    /// 向量与查询向量的距离
    pub distance: u32,
}

/// Faiss 索引
pub struct FaissIndex {
    index: *mut FaissIndexBinary,
    /// 向量维数
    d: i32,
}

impl FaissIndex {
    /// 创建一个新的 Faiss 索引
    ///
    /// # Arguments
    ///
    /// * `d` - 向量维数，对于二进制向量来说，是总 bit 数
    /// * `description` - 索引的描述字符串
    pub fn new(d: i32, description: &str) -> Self {
        let index = std::ptr::null_mut();
        let description = std::ffi::CString::new(description).unwrap();
        unsafe {
            faiss_index_binary_factory(&index as *const _ as *mut _, d, description.as_ptr());
        }
        Self { index, d }
    }

    /// 从文件加载索引
    ///
    /// # Arguments
    ///
    /// * `d` - 文件路径
    /// * `mmap` - 是否使用 mmap 模式加载
    pub fn from_file(path: &str, mmap: bool) -> Self {
        let index = std::ptr::null_mut();
        let path = CString::new(path).unwrap();
        let io_flags = match mmap {
            true => 0x2 | 0x8 | 0x646f0000,
            _ => 0,
        };
        unsafe {
            faiss_read_index_binary_fname(path.as_ptr(), io_flags, &index as *const _ as *mut _);
        }
        let d = unsafe { faiss_IndexBinary_d(index) };
        Self { index, d }
    }

    /// 该索引中的向量数量
    pub fn ntotal(&self) -> i64 {
        unsafe { faiss_IndexBinary_ntotal(self.index) }
    }

    /// 该索引是否已经训练
    pub fn is_trained(&self) -> bool {
        unsafe { faiss_IndexBinary_is_trained(self.index) != 0 }
    }

    /// 将索引写入到文件
    pub fn write_file(&self, path: &str) {
        let path = CString::new(path).unwrap();
        unsafe {
            faiss_write_index_binary_fname(self.index, path.as_ptr());
        }
    }

    /// 使用自定义 ID 添加向量到索引中
    ///
    /// # Arguments
    ///
    /// * `v` - 向量，大小为 (n, d)
    /// * `ids` - 向量 id 列表，长度为 n
    pub fn add_with_ids<M>(&mut self, v: &M, ids: &[i64])
    where
        M: Matrix,
    {
        assert_eq!(v.width() * 8, self.d as usize);
        assert_eq!(v.height(), ids.len());
        unsafe {
            faiss_IndexBinary_add_with_ids(self.index, v.height() as i64, v.as_ptr(), ids.as_ptr());
        }
    }

    /// 批量搜索 points 中的向量，对每个向量，返回 knn 个最近邻
    ///
    /// # Arguments
    ///
    /// * `points` - 需要搜索的向量数组，大小为 (n, d)，其中 n 为向量数量，d 为向量维度
    /// * `knn` - 每个向量需要返回的最近邻数量
    /// * `params` - 搜索参数
    pub fn search<M>(&self, points: &M, knn: usize, params: FaissSearchParams) -> Vec<Vec<Neighbor>>
    where
        M: Matrix,
    {
        assert_eq!(points.width() * 8, self.d as usize);
        let mut dists = vec![0i32; points.height() * knn];
        let mut indices = vec![0i64; points.height() * knn];

        let start = Instant::now();

        // 初始化参数
        let mut raw_params = MaybeUninit::<*mut FaissSearchParametersIVF>::uninit();
        unsafe {
            faiss_SearchParametersIVF_new_with(
                raw_params.as_mut_ptr(),
                ptr::null_mut(),
                params.nprobe,
                params.max_codes,
            )
        };

        // 搜索
        unsafe {
            let params = raw_params.assume_init();
            faiss_IndexBinary_search_with_params(
                self.index,
                points.height() as i64,
                points.as_ptr(),
                knn as i64,
                params,
                dists.as_mut_ptr(),
                indices.as_mut_ptr(),
            );
            faiss_SearchParametersIVF_free(params);
        }

        // 打印搜索统计信息，并重置
        // NOTE: 这里的统计不是针对单次搜索的，由于统计变量是全局的，多线程搜索会累加
        let mut stats = unsafe { *faiss_get_indexIVF_stats() };

        debug!("knn search time  : {}ms", start.elapsed().as_millis());
        debug!("ndis             : {}", stats.nq);
        debug!("nprobe           : {}", stats.nlist);
        debug!("nheap_updates    : {}", stats.nheap_updates);
        debug!("quantization_time: {:.2}ms", stats.quantization_time);
        debug!("search_time      : {:.2}ms", stats.search_time);

        unsafe {
            faiss_IndexIVFStats_reset(&mut stats);
        }

        // 整理结果
        indices
            .into_iter()
            .zip(dists.into_iter())
            .map(|(index, distance)| Neighbor {
                index: index as usize,
                distance: distance as u32,
            })
            .chunks(knn)
            .into_iter()
            .map(|chunk| chunk.collect())
            .collect()
    }

    /// knn 搜索时使用堆排序，默认开启。关闭时使用计数排序。
    ///
    /// 具体来讲，对于每个查询向量，依次扫描它需要检索的所有倒排列表。
    /// 适用于查询数量少，但倒排列表较大的场景。
    pub fn set_use_heap(&mut self, use_heap: bool) {
        unsafe {
            faiss_IndexBinaryIVF_set_use_heap(self.index, use_heap as i32);
        }
    }

    /// 设置搜索策略为「倒排列表优先」，默认关闭，优先级高于 `set_use_heap`
    ///
    /// 具体来讲，对于每个需要查询的倒排列表，把匹配的所有向量都一起处理。
    /// 适用于大批量查询场景。
    pub fn set_per_invlit_search(&mut self, use_per_invlit_search: bool) {
        unsafe {
            faiss_IndexBinaryIVF_set_per_invlist_search(self.index, use_per_invlit_search as i32);
        }
    }

    /// 索引的不平衡度
    /// 1 表示完全平衡，越大表示越不平衡
    pub fn imbalance_factor(&self) -> f64 {
        let imbalance = unsafe { faiss_IndexBinaryIVF_imbalance_factor(self.index) };
        imbalance
    }

    /// 打印倒排列表信息
    pub fn print_stats(&self) {
        unsafe {
            faiss_IndexBinaryIVF_print_stats(self.index);
        }
    }

    /// 索引倒排列表数量
    pub fn nlist(&self) -> usize {
        unsafe { faiss_IndexBinaryIVF_nlist(self.index) }
    }

    /// 合并索引
    ///
    /// # Arguments
    ///
    /// * `other` - 需要合并的索引
    /// * `add_id` - 合并是在原 ID 基础上增加的 ID
    pub fn merge_from(&mut self, other: &Self, add_id: i64) {
        unsafe {
            faiss_IndexBinaryIVF_merge_from(self.index, other.index, add_id);
        }
    }
}

impl Drop for FaissIndex {
    fn drop(&mut self) {
        unsafe {
            faiss_IndexBinary_free(self.index);
        }
    }
}

unsafe impl Sync for FaissIndex {}
unsafe impl Send for FaissIndex {}

/// Faiss 搜索参数
pub struct FaissSearchParams {
    /// 需要搜索的倒排列表数量，默认为 1
    pub nprobe: usize,
    /// 搜索时最多检查多少个向量，默认为 0，表示不限制
    pub max_codes: usize,
}

impl Default for FaissSearchParams {
    fn default() -> Self {
        Self {
            nprobe: 1,
            max_codes: 0,
        }
    }
}
