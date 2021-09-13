use itertools::Itertools;
use opencv::prelude::*;
use std::ffi::c_void;

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
