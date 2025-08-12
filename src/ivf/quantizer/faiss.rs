use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;

use anyhow::Result;
use faiss_sys::*;

use crate::ivf::Quantizer;

#[derive(Debug)]
pub struct FaissHNSWQuantizer<const N: usize> {
    index: *mut FaissIndexBinary,
}

unsafe impl<const N: usize> Send for FaissHNSWQuantizer<N> {}
unsafe impl<const N: usize> Sync for FaissHNSWQuantizer<N> {}

impl<const N: usize> Quantizer<N> for FaissHNSWQuantizer<N> {
    fn open<P: AsRef<Path>>(path: P) -> Result<Self>
    where
        Self: Sized,
    {
        let mut index = ptr::null_mut();
        let path = path.as_ref();
        let path = CString::new(path.to_str().unwrap())?;
        unsafe {
            faiss_try(faiss_read_index_binary_fname(path.as_ptr(), 0, &mut index))?;
            faiss_try(faiss_IndexBinaryHNSW_set_efSearch(index.cast(), 16))?;
        }
        Ok(Self { index })
    }

    fn init(x: &[[u8; N]]) -> Result<Self>
    where
        Self: Sized,
    {
        let mut index = ptr::null_mut();
        unsafe {
            faiss_try(faiss_IndexBinaryHNSW_new(&mut index, (N * 8) as i32, 32))?;
            // faiss 默认值为 40, 16
            faiss_try(faiss_IndexBinaryHNSW_set_efConstruction(index, 128))?;
            faiss_try(faiss_IndexBinaryHNSW_set_efSearch(index, 16))?;
        }
        let index = index.cast();
        let xf = x.as_flattened();
        unsafe {
            faiss_try(faiss_IndexBinary_add(index, x.len() as i64, xf.as_ptr()))?;
        }
        Ok(Self { index })
    }

    fn search(&self, x: &[[u8; N]], k: usize) -> Result<Vec<i64>> {
        let xf = x.as_flattened();
        let mut distances = vec![0; x.len() * k];
        let mut labels = vec![0; x.len() * k];
        unsafe {
            faiss_try(faiss_IndexBinary_search(
                self.index,
                x.len() as i64,
                xf.as_ptr(),
                k as i64,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            ))?;
        }
        Ok(labels)
    }

    fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let path = CString::new(path.to_str().unwrap())?;
        unsafe {
            faiss_try(faiss_write_index_binary_fname(self.index, path.as_ptr()))?;
        }
        Ok(())
    }

    fn nlist(&self) -> usize {
        unsafe { faiss_IndexBinary_ntotal(self.index) as usize }
    }

    fn centroids(&self) -> Result<&[[u8; N]]> {
        let centroids = unsafe {
            let mut xb = ptr::null_mut();
            let storage = faiss_IndexBinaryHNSW_storage(self.index.cast());
            faiss_try(faiss_IndexBinaryFlat_xb(storage.cast(), &mut xb))?;
            std::slice::from_raw_parts(xb, self.nlist() * N)
        };
        let (centroids, _) = centroids.as_chunks::<N>();
        Ok(centroids)
    }
}

fn faiss_try(code: std::os::raw::c_int) -> Result<()> {
    if code != 0 {
        unsafe {
            let err = faiss_get_last_error();
            assert!(!err.is_null());
            let cstr = CStr::from_ptr(err);
            anyhow::bail!("faiss error {}: {}", code, cstr.to_string_lossy());
        }
    }
    Ok(())
}
