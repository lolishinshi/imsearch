use std::ffi::CString;
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
            faiss_try(faiss_IndexBinaryHNSW_set_efSearch(index, 32))?;
        }
        let index = index.cast();
        let xf = x.as_flattened();
        unsafe {
            faiss_try(faiss_IndexBinary_add(index, x.len() as i64, xf.as_ptr()))?;
        }
        Ok(Self { index })
    }

    fn search(&self, x: &[[u8; N]], k: usize) -> Result<Vec<Vec<usize>>> {
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
        // TODO: 能不能允许 search 返回 Iterator<Iterator<usize>>?
        Ok(labels.chunks_exact(k).map(|c| c.iter().map(|l| *l as usize).collect()).collect())
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
}

fn faiss_try(code: std::os::raw::c_int) -> Result<()> {
    if code != 0 {
        anyhow::bail!("faiss error: {}", code);
    }
    Ok(())
}
