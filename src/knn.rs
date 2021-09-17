use itertools::Itertools;
use opencv::prelude::*;
use std::ffi::{c_void, CString, CStr};
use std::os::raw::c_char;
use std::os::unix::prelude::AsRawFd;

extern "C" {
    fn knn_searcher_init(
        points: *const c_void,
        table_number: u32,
        key_size: u32,
        multi_probe_level: u32,
    ) -> *const c_void;
    fn knn_searcher_add(this: *const c_void, points: *const c_void);
    fn knn_searcher_build_index(this: *const c_void);
    fn knn_searcher_search(
        this: *const c_void,
        points: *const c_void,
        indices: *mut usize,
        dists: *mut u32,
        knn: usize,
        checks: i32,
    ) -> i32;
    fn knn_searcher_delete(this: *const c_void);

    fn faiss_indexBinary_factory(d: i32, description: *const c_char) -> *mut c_void;

    fn faiss_IndexBinary_train(index: *mut c_void, n: i64, x: *const u8);

    fn faiss_IndexBinary_add(index: *mut c_void, n: i64, x: *const u8);

    fn faiss_IndexBinary_add_with_ids(index: *mut c_void, n: i64, x: *const u8, xids: *const i64);

    fn faiss_IndexBinary_search(
        index: *mut c_void,
        n: i64,
        x: *const u8,
        k: i64,
        distances: *const i32,
        labels: *const i64,
    );

    fn faiss_IndexBinary_delete(index: *mut c_void);

    fn faiss_write_index_binary(index: *mut c_void, f: *const c_char);

    fn faiss_read_index_binary(f: *const c_char, io_flags: i32) -> *mut c_void;
}

pub struct FaissSearcher<'a> {
    index: *mut c_void,
    d: i32,
    _phantom: std::marker::PhantomData<&'a u8>,
}

impl<'a> FaissSearcher<'a> {
    pub fn new(d: i32, description: &str) -> Self {
        let description = std::ffi::CString::new(description).unwrap();
        let index = unsafe { faiss_indexBinary_factory(d, description.as_ptr()) };
        Self {
            index,
            d,
            _phantom: Default::default(),
        }
    }

    pub fn from_file(path: &str, d: i32) -> Self {
        let path = CString::new(path).unwrap();
        let index = unsafe { faiss_read_index_binary(path.as_ptr(), 0) };
        Self {
            index,
            d,
            _phantom: Default::default(),
        }
    }

    pub fn write_file(&self, path: &str) {
        let path = CString::new(path).unwrap();
        unsafe { faiss_write_index_binary(self.index, path.as_ptr()) }
    }

    // TODO: 替换掉 OpenCV 的 Mat
    pub fn train(&mut self, v: &Mat) {
        assert_eq!(v.cols() * 8, self.d);
        unsafe {
            faiss_IndexBinary_train(self.index, v.rows() as i64, v.data().unwrap() as *const u8)
        }
    }

    pub fn add(&mut self, v: &'a Mat) {
        assert_eq!(v.cols() * 8, self.d);
        unsafe {
            faiss_IndexBinary_add(self.index, v.rows() as i64, v.data().unwrap() as *const u8)
        }
    }

    pub fn search(&self, points: &Mat, knn: usize) -> Vec<Vec<Neighbor>> {
        assert_eq!(points.cols() * 8, self.d);
        let mut dists = vec![0i32; points.rows() as usize * knn];
        let mut indices = vec![0i64; points.rows() as usize * knn];
        unsafe {
            faiss_IndexBinary_search(
                self.index,
                points.rows() as i64,
                points.data().unwrap() as *const u8,
                knn as i64,
                dists.as_ptr(),
                indices.as_ptr(),
            )
        }

        indices
            .into_iter()
            .zip(dists.into_iter())
            .map(|(index, distance)| Neighbor { index: index as usize, distance: distance as u32 })
            .chunks(knn)
            .into_iter()
            .map(|chunk| chunk.collect())
            .collect()
    }

}

impl<'a> Drop for FaissSearcher<'a> {
    fn drop(&mut self) {
        unsafe { faiss_IndexBinary_delete(self.index) }
    }
}

pub struct Neighbor {
    pub index: usize,
    pub distance: u32,
}

pub struct KnnSearcher<'a> {
    index: *const c_void,
    checks: i32,
    _phantom: std::marker::PhantomData<&'a u8>,
}

impl<'a> KnnSearcher<'a> {
    pub fn new(
        points: &'a Mat,
        table_number: u32,
        key_size: u32,
        multi_probe_level: u32,
        checks: i32,
    ) -> Self {
        assert_eq!(points.cols(), 32);
        assert!(checks >= -2);
        let index = unsafe {
            knn_searcher_init(
                points.as_raw_Mat(),
                table_number,
                key_size,
                multi_probe_level,
            )
        };
        assert!(!index.is_null());
        Self {
            index,
            checks,
            _phantom: Default::default(),
        }
    }

    pub fn add(&mut self, points: &'a Mat) {
        assert_eq!(points.cols(), 32);
        unsafe { knn_searcher_add(self.index, points.as_raw_Mat()) }
    }

    pub fn knn_search(&mut self, points: &Mat, knn: usize) -> Vec<Vec<Neighbor>> {
        assert_eq!(points.cols(), 32);
        let mut indices = vec![0usize; points.rows() as usize * knn];
        let mut dists = vec![0u32; points.rows() as usize * knn];
        unsafe {
            knn_searcher_search(
                self.index,
                points.as_raw_Mat(),
                indices.as_mut_ptr(),
                dists.as_mut_ptr(),
                knn,
                self.checks,
            )
        };
        indices
            .into_iter()
            .zip(dists.into_iter())
            .map(|(index, distance)| Neighbor { index, distance })
            .chunks(knn)
            .into_iter()
            .map(|chunk| chunk.collect())
            .collect()
    }

    pub fn build(&mut self) {
        unsafe { knn_searcher_build_index(self.index) }
    }
}

impl<'a> Drop for KnnSearcher<'a> {
    fn drop(&mut self) {
        unsafe { knn_searcher_delete(self.index) }
    }
}
