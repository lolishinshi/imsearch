use crate::matrix::Matrix;
use itertools::Itertools;
use std::ffi::CString;
use std::os::raw::c_char;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct FaissIndexBinary {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct FaissIndexBinaryIVF {
    _unused: [u8; 0],
}

extern "C" {
    fn faiss_index_binary_factory(
        index: *mut *mut FaissIndexBinary,
        d: i32,
        description: *const c_char,
    );

    fn faiss_IndexBinary_d(index: *const FaissIndexBinary) -> i32;

    fn faiss_IndexBinary_ntotal(index: *const FaissIndexBinary) -> i64;

    fn faiss_IndexBinary_is_trained(index: *const FaissIndexBinary) -> bool;

    fn faiss_IndexBinary_train(index: *mut FaissIndexBinary, n: i64, x: *const u8);

    fn faiss_IndexBinary_add(index: *mut FaissIndexBinary, n: i64, x: *const u8);

    fn faiss_IndexBinary_add_with_ids(
        index: *mut FaissIndexBinary,
        n: i64,
        x: *const u8,
        xids: *const i64,
    );

    fn faiss_IndexBinary_search(
        index: *const FaissIndexBinary,
        n: i64,
        x: *const u8,
        k: i64,
        distances: *mut i32,
        labels: *mut i64,
    );

    fn faiss_IndexBinary_free(index: *mut FaissIndexBinary);

    fn faiss_write_index_binary_fname(index: *const FaissIndexBinary, f: *const c_char);

    fn faiss_read_index_binary_fname(
        f: *const c_char,
        io_flags: i32,
        index: *mut *mut FaissIndexBinary,
    );

    fn faiss_IndexBinaryIVF_set_nprobe(index: *mut FaissIndexBinaryIVF, nprobe: usize);

    fn faiss_IndexBinaryIVF_nlist(index: *const FaissIndexBinaryIVF) -> usize;
}

pub struct Neighbor {
    pub index: usize,
    pub distance: u32,
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
        unsafe { faiss_IndexBinary_is_trained(self.index) }
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
            faiss_IndexBinaryIVF_set_nprobe(self.index as *mut FaissIndexBinaryIVF, nprobe);
        }
    }

    pub fn nlist(&self) -> usize {
        unsafe { faiss_IndexBinaryIVF_nlist(self.index as *const FaissIndexBinaryIVF) }
    }
}

impl Drop for FaissIndex {
    fn drop(&mut self) {
        unsafe {
            faiss_IndexBinary_free(self.index);
        }
    }
}
