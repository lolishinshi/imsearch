use faiss_sys::*;
use log::debug;
use opencv::prelude::*;
use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr::null_mut;

/// Faiss 搜索结果
pub struct Neighbor {
    /// 向量在索引中的 ID
    pub index: i64,
    /// 向量与查询向量的距离
    pub distance: i32,
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
        let index = null_mut();
        let description = CString::new(description).unwrap();
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
        let index = null_mut();
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
    pub fn write_file(&self, path: impl AsRef<Path>) {
        let path = path.as_ref();
        let path = path.to_str().unwrap();
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
    pub fn add_with_ids(&mut self, v: &Mat, ids: &[i64]) {
        assert_eq!(v.cols() * 8, self.d);
        assert_eq!(v.rows() as usize, ids.len());
        unsafe {
            faiss_IndexBinary_add_with_ids(self.index, v.rows() as i64, v.data(), ids.as_ptr());
        }
    }

    /// 批量搜索 points 中的向量，对每个向量，返回 knn 个最近邻
    ///
    /// # Arguments
    ///
    /// * `points` - 需要搜索的向量数组，大小为 (n, d)，其中 n 为向量数量，d 为向量维度
    /// * `knn` - 每个向量需要返回的最近邻数量
    /// * `params` - 搜索参数
    pub fn search(
        &self,
        points: &Mat,
        knn: usize,
        params: FaissSearchParams,
    ) -> Vec<Vec<Neighbor>> {
        assert_eq!(points.cols() * 8, self.d);
        let mut distances = vec![0i32; points.rows() as usize * knn];
        let mut labels = vec![0i64; points.rows() as usize * knn];

        // 初始化参数
        let mut raw_params = MaybeUninit::<*mut FaissSearchParametersIVF>::uninit();
        unsafe {
            faiss_SearchParametersIVF_new_with(
                raw_params.as_mut_ptr(),
                null_mut(),
                params.nprobe,
                params.max_codes,
            )
        };

        // 搜索
        unsafe {
            let params = raw_params.assume_init();
            faiss_IndexBinary_search_with_params(
                self.index,
                points.rows() as i64,
                points.data(),
                knn as i64,
                params,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            );
            faiss_SearchParametersIVF_free(params);
        }

        // 打印搜索统计信息，并重置
        // NOTE: 这里的统计不是针对单次搜索的，由于统计变量是全局的，多线程搜索会累加
        let (stats, raw_stats) = unsafe {
            let stats = faiss_get_indexIVF_stats();
            (*stats, stats)
        };

        debug!("ndis             : {}", stats.nq);
        debug!("nprobe           : {}", stats.nlist);
        debug!("nheap_updates    : {}", stats.nheap_updates);
        debug!("quantization_time: {:.2}ms", stats.quantization_time);
        debug!("search_time      : {:.2}ms", stats.search_time);

        unsafe {
            faiss_IndexIVFStats_reset(raw_stats);
        }

        // 整理结果
        let mut result = vec![];
        for (labels, distances) in labels.chunks(knn).zip(distances.chunks(knn)) {
            let neighbors = labels
                .iter()
                .zip(distances)
                .map(|(index, distance)| Neighbor { index: *index, distance: *distance })
                .collect::<Vec<_>>();
            result.push(neighbors);
        }

        result
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
        unsafe { faiss_IndexBinaryIVF_imbalance_factor(self.index) }
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

    /// 获取 faiss 版本
    pub fn faiss_version(&self) -> String {
        let version = unsafe { faiss_get_version() };
        let version = unsafe { CStr::from_ptr(version) };
        version.to_string_lossy().to_string()
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
#[derive(Debug, Clone)]
pub struct FaissSearchParams {
    /// 需要搜索的倒排列表数量，默认为 1
    pub nprobe: usize,
    /// 搜索时最多检查多少个向量，默认为 0，表示不限制
    pub max_codes: usize,
}

impl Default for FaissSearchParams {
    fn default() -> Self {
        Self { nprobe: 1, max_codes: 0 }
    }
}
