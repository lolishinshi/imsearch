mod index;
mod invlists;
mod types;

use std::ffi::CStr;

pub use index::*;
pub use invlists::*;
pub use types::*;

pub fn get_faiss_stats() -> faiss_sys::FaissIndexIVFStats {
    unsafe {
        let stats = faiss_sys::faiss_get_indexIVF_stats();
        *stats
    }
}

pub fn reset_faiss_stats() {
    unsafe {
        faiss_sys::faiss_IndexIVFStats_reset(faiss_sys::faiss_get_indexIVF_stats());
    }
}

fn faiss_try(v: i32) -> anyhow::Result<i32> {
    if v >= 0 {
        Ok(v)
    } else {
        let last_error = unsafe { faiss_sys::faiss_get_last_error() };
        if last_error.is_null() {
            Err(anyhow::anyhow!("未知错误"))
        } else {
            let last_error = unsafe { CStr::from_ptr(last_error) };
            Err(anyhow::anyhow!("{}", last_error.to_string_lossy()))
        }
    }
}
