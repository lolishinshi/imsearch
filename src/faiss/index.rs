use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::null_mut;

use anyhow::Result;
use faiss_sys::*;
use log::debug;
use ndarray::Array2;
use opencv::prelude::*;

use super::FaissInvLists;
use super::types::*;
use crate::faiss::faiss_try;

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
    pub fn new(d: i32, description: &str) -> Result<Self> {
        let index = null_mut();
        let description = CString::new(description).unwrap();
        unsafe {
            faiss_try(faiss_index_binary_factory(
                &index as *const _ as *mut _,
                d,
                description.as_ptr(),
            ))?;
        }
        Ok(Self { index, d })
    }

    /// 从文件加载索引
    ///
    /// # Arguments
    ///
    /// * `d` - 文件路径
    /// * `mmap` - 是否使用 mmap 模式加载
    pub fn from_file<P: AsRef<Path>>(path: P, mmap: bool) -> Result<Self> {
        let index = null_mut();
        let path = path.as_ref().to_str().unwrap();
        let path = CString::new(path).unwrap();
        let io_flags = match mmap {
            true => 0x8 | 0x646f0000,
            _ => 0,
        };
        unsafe {
            faiss_try(faiss_read_index_binary_fname(
                path.as_ptr(),
                io_flags,
                &index as *const _ as *mut _,
            ))?;
        }
        let d = unsafe { faiss_IndexBinary_d(index) };
        Ok(Self { index, d })
    }

    /// 该索引中的向量数量
    pub fn ntotal(&self) -> i64 {
        unsafe { faiss_IndexBinary_ntotal(self.index) }
    }

    /// 设置索引中的向量数量
    pub fn set_ntotal(&mut self, ntotal: i64) {
        unsafe { faiss_IndexBinary_set_ntotal(self.index, ntotal) }
    }

    /// 该索引是否已经训练
    pub fn is_trained(&self) -> bool {
        unsafe { faiss_IndexBinary_is_trained(self.index) != 0 }
    }

    /// 将索引写入到文件，考虑到中途打断的情况，使用临时文件写入再重命名
    pub fn write_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let tmp_path = path.with_extension("tmp");
        unsafe {
            let tmp_path = tmp_path.to_str().unwrap();
            let tmp_path = CString::new(tmp_path).unwrap();
            faiss_try(faiss_write_index_binary_fname(self.index, tmp_path.as_ptr()))?;
        }
        std::fs::rename(tmp_path, path)?;
        Ok(())
    }

    /// 添加若干条向量到索引中
    ///
    /// # Arguments
    ///
    /// * `v` - 向量，大小为 (n, d)
    pub fn add(&mut self, v: &Array2<u8>) {
        assert_eq!(v.dim().1 * 8, self.d as usize);
        unsafe {
            faiss_IndexBinary_add(self.index, v.dim().0 as i64, v.as_ptr());
        }
    }

    /// 使用自定义 ID 添加向量到索引中
    ///
    /// # Arguments
    ///
    /// * `v` - 向量，大小为 (n, d)
    /// * `ids` - 向量 id 列表，长度为 n
    pub fn add_with_ids(&mut self, v: &Array2<u8>, ids: &[i64]) -> Result<()> {
        assert_eq!(v.dim().1 * 8, self.d as usize);
        assert_eq!(v.dim().0, ids.len());
        unsafe {
            faiss_try(faiss_IndexBinary_add_with_ids(
                self.index,
                v.dim().0 as i64,
                v.as_ptr(),
                ids.as_ptr(),
            ))?;
        }
        Ok(())
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
        let raw_params = params.into_raw();
        // 搜索
        unsafe {
            faiss_try(faiss_IndexBinary_search_with_params(
                self.index,
                points.rows() as i64,
                points.data(),
                knn as i64,
                *raw_params,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            ))
            .unwrap();
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

    pub fn code_size(&self) -> i32 {
        unsafe { faiss_IndexBinary_code_size(self.index) }
    }

    pub fn set_own_invlists(&mut self, own: bool) {
        unsafe { faiss_IndexBinaryIVF_set_own_invlists(self.index, own as i32) }
    }

    /// 合并索引
    ///
    /// # Arguments
    ///
    /// * `other` - 需要合并的索引
    /// * `add_id` - 合并是在原 ID 基础上增加的 ID
    pub fn merge_from(&mut self, other: &Self, add_id: i64) -> Result<()> {
        unsafe {
            faiss_try(faiss_IndexBinaryIVF_merge_from(self.index, other.index, add_id))?;
        }
        Ok(())
    }

    pub fn invlists(&self) -> FaissInvLists {
        unsafe { FaissInvLists(faiss_IndexBinaryIVF_invlists(self.index)) }
    }

    pub fn replace_invlists<T: Into<*mut FaissInvertedLists_H>>(
        &mut self,
        invlists: T,
        own: bool,
    ) -> Result<()> {
        unsafe {
            faiss_try(faiss_IndexBinaryIVF_replace_invlists(
                self.index,
                invlists.into(),
                own as i32,
            ))?;
        }
        Ok(())
    }

    pub fn list_size(&self, list_no: usize) -> usize {
        unsafe { faiss_IndexBinaryIVF_get_list_size(self.index, list_no) }
    }

    pub fn ids(&self, list_no: usize) -> Vec<i64> {
        let mut ids = vec![0i64; self.list_size(list_no)];
        unsafe {
            faiss_IndexBinaryIVF_invlists_get_ids(self.index, list_no, ids.as_mut_ptr());
        }
        ids
    }

    pub fn codes(&self, list_no: usize) -> Vec<u8> {
        let mut codes = vec![0u8; self.list_size(list_no) * self.code_size() as usize];
        unsafe {
            faiss_IndexBinaryIVF_invlists_get_codes(self.index, list_no, codes.as_mut_ptr());
        }
        codes
    }
}

impl Drop for FaissIndex {
    fn drop(&mut self) {
        debug!("释放 faiss 索引");
        unsafe {
            faiss_IndexBinaryIVF_free(self.index);
        }
    }
}

unsafe impl Sync for FaissIndex {}
unsafe impl Send for FaissIndex {}

/// 获取 faiss 版本
pub fn faiss_version() -> String {
    let version = unsafe { faiss_get_version() };
    let version = unsafe { CStr::from_ptr(version) };
    version.to_string_lossy().to_string()
}
