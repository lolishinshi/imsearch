use std::ffi::CString;
use std::mem;
use std::path::Path;

use faiss_sys::*;

pub struct FaissInvLists(pub(super) *mut FaissInvertedLists_H);

// NOTE: 在这个仓库中，磁盘倒排列表一般所有权都是给 faiss 的，因此不需要实现 Drop
pub struct FaissOnDiskInvLists(pub(super) *mut FaissOnDiskInvertedLists);

pub struct FaissHStackInvLists(pub(super) *mut FaissHStackInvertedLists);

impl From<FaissOnDiskInvLists> for *mut FaissInvertedLists_H {
    fn from(ivf: FaissOnDiskInvLists) -> Self {
        ivf.0
    }
}

impl From<FaissHStackInvLists> for *mut FaissInvertedLists_H {
    fn from(ivf: FaissHStackInvLists) -> Self {
        ivf.0 as *mut _
    }
}

impl FaissOnDiskInvLists {
    pub fn new<P: AsRef<Path>>(nlist: usize, code_size: usize, filename: P) -> Self {
        let filename = filename.as_ref().to_str().unwrap();
        unsafe {
            let f = CString::new(filename).unwrap();
            let mut inner = mem::zeroed();
            faiss_OnDiskInvertedLists_new(&mut inner, nlist, code_size, f.as_ptr());
            FaissOnDiskInvLists(inner)
        }
    }

    pub fn merge_from_multiple(
        &mut self,
        ivfs: Vec<FaissInvLists>,
        shift_ids: bool,
        verbose: bool,
    ) -> usize {
        unsafe {
            let mut ivfs = ivfs.into_iter().map(|ivf| ivf.0 as *const _).collect::<Vec<_>>();
            faiss_OnDiskInvertedLists_merge_from_multiple(
                self.0,
                ivfs.as_mut_ptr(),
                ivfs.len() as i32,
                shift_ids as i32,
                verbose as i32,
            ) as usize
        }
    }

    pub fn set_filename<P: AsRef<Path>>(&mut self, filename: P) {
        let filename = filename.as_ref().to_str().unwrap();
        unsafe {
            let f = CString::new(filename).unwrap();
            faiss_OnDiskInvertedLists_set_filename(self.0, f.as_ptr());
        }
    }
}

impl FaissHStackInvLists {
    pub fn new(ivfs: Vec<FaissInvLists>) -> Self {
        unsafe {
            let mut inner = mem::zeroed();
            let mut ivfs = ivfs.into_iter().map(|ivf| ivf.0 as *const _).collect::<Vec<_>>();
            faiss_HStackInvertedLists_new(&mut inner, ivfs.len() as i32, ivfs.as_mut_ptr());
            FaissHStackInvLists(inner)
        }
    }
}
