use crate::matrix::Matrix;
use faiss_sys::*;
use itertools::Itertools;
use log::debug;
use std::ffi::CString;
use std::path::Path;
use std::time::Instant;

pub struct Neighbor {
    pub index: usize,
    pub distance: u32,
}

pub struct MultiFaissIndex {
    index: Vec<FaissIndex>,
}

impl MultiFaissIndex {
    pub fn from_file<I, P>(path: I, mmap: bool) -> Self
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let index = path
            .into_iter()
            .map(|path| FaissIndex::from_file(&*path.as_ref().to_string_lossy(), mmap))
            .collect();
        Self { index }
    }

    pub fn search<M>(&self, points: &M, knn: usize) -> Vec<Vec<Neighbor>>
    where
        M: Matrix,
    {
        let mut v = vec![];
        for index in self.index.iter() {
            v.extend(index.search(points, knn))
        }
        v
    }

    pub fn set_nprobe(&mut self, nprobe: usize) {
        for index in self.index.iter_mut() {
            index.set_nprobe(nprobe)
        }
    }
}

pub struct FaissIndex {
    index: *mut FaissIndexBinary,
    d: i32,
}

impl FaissIndex {
    pub fn new(d: i32, description: &str) -> Self {
        let index = std::ptr::null_mut();
        let description = std::ffi::CString::new(description).unwrap();
        unsafe {
            faiss_index_binary_factory(&index as *const _ as *mut _, d, description.as_ptr());
        }
        Self { index, d }
    }

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

    pub fn ntotal(&self) -> i64 {
        unsafe { faiss_IndexBinary_ntotal(self.index) }
    }

    pub fn is_trained(&self) -> bool {
        unsafe { faiss_IndexBinary_is_trained(self.index) != 0 }
    }

    pub fn write_file(&self, path: &str) {
        let path = CString::new(path).unwrap();
        unsafe {
            faiss_write_index_binary_fname(self.index, path.as_ptr());
        }
    }

    pub fn train<M>(&mut self, v: &M)
    where
        M: Matrix,
    {
        assert_eq!(v.width() * 8, self.d as usize);
        unsafe {
            faiss_IndexBinary_train(self.index, v.height() as i64, v.as_ptr());
        }
    }

    pub fn add<M>(&mut self, v: &M)
    where
        M: Matrix,
    {
        assert_eq!(v.width() * 8, self.d as usize);
        unsafe {
            faiss_IndexBinary_add(self.index, v.height() as i64, v.as_ptr());
        }
    }

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

    pub fn search<M>(&self, points: &M, knn: usize) -> Vec<Vec<Neighbor>>
    where
        M: Matrix,
    {
        assert_eq!(points.width() * 8, self.d as usize);
        let mut dists = vec![0i32; points.height() * knn];
        let mut indices = vec![0i64; points.height() * knn];

        let start = Instant::now();
        unsafe {
            faiss_IndexBinary_search(
                self.index,
                points.height() as i64,
                points.as_ptr(),
                knn as i64,
                dists.as_mut_ptr(),
                indices.as_mut_ptr(),
            );
        }

        let stats = unsafe { *faiss_get_indexIVF_stats() };

        debug!("knn search time  : {}ms", start.elapsed().as_millis());
        debug!("ndis             : {}", stats.nq);
        debug!("nprobe           : {}", stats.nlist);
        debug!("nheap_updates    : {}", stats.nheap_updates);
        debug!("quantization_time: {:.2}ms", stats.quantization_time);
        debug!("search_time      : {:.2}ms", stats.search_time);

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

    pub fn set_nprobe(&mut self, nprobe: usize) {
        unsafe {
            faiss_IndexBinaryIVF_set_nprobe(self.index, nprobe);
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
