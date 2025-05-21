use std::mem::MaybeUninit;
use std::ops::Deref;
use std::ptr::null_mut;

use faiss_sys::*;

/// Faiss 搜索参数
// NOTE: 不要为它实现 Clone
pub struct RawFaissSearchParams {
    params: *mut FaissSearchParameters,
    quantizer_params: *mut FaissSearchParametersHNSW,
}

impl Deref for RawFaissSearchParams {
    type Target = *mut FaissSearchParameters;

    fn deref(&self) -> &Self::Target {
        &self.params
    }
}

impl Drop for RawFaissSearchParams {
    fn drop(&mut self) {
        unsafe {
            faiss_SearchParameters_free(self.params);
            faiss_SearchParametersHNSW_free(self.quantizer_params);
        }
    }
}

#[derive(Debug, Clone)]
pub struct FaissSearchParams {
    /// 需要搜索的倒排列表数量，默认为 1
    pub nprobe: usize,
    /// HNSW 的 efSearch 参数
    pub ef_search: usize,
}

impl FaissSearchParams {
    pub fn into_raw(self) -> RawFaissSearchParams {
        let mut params = MaybeUninit::<*mut FaissSearchParametersIVF>::zeroed();
        let mut quantizer_params = MaybeUninit::<*mut FaissSearchParametersHNSW>::zeroed();
        unsafe {
            faiss_SearchParametersHNSW_new_with(
                quantizer_params.as_mut_ptr(),
                null_mut(),
                self.ef_search as i32,
            );
            faiss_SearchParametersIVF_new_with(
                params.as_mut_ptr(),
                null_mut(),
                self.nprobe,
                0,
                quantizer_params.assume_init(),
            );
            RawFaissSearchParams {
                params: params.assume_init(),
                quantizer_params: quantizer_params.assume_init(),
            }
        }
    }
}

impl Default for FaissSearchParams {
    fn default() -> Self {
        Self { nprobe: 1, ef_search: 16 }
    }
}

/// Faiss 搜索结果
#[derive(Debug, Clone)]
pub struct Neighbor {
    /// 向量在索引中的 ID
    pub index: i64,
    /// 向量与查询向量的距离
    pub distance: i32,
}
