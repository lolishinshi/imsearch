use std::ffi::CString;
use std::mem;

use faiss_sys::*;

pub struct FaissInvLists(pub(super) *mut FaissInvertedLists_H);

pub struct FaissOnDiskInvLists(pub(super) *mut FaissOnDiskInvertedLists);

impl FaissOnDiskInvLists {
    pub fn new(nlist: usize, code_size: usize, filename: &str) -> Self {
        unsafe {
            let f = CString::new(filename).unwrap();
            let mut inner = mem::zeroed();
            faiss_OnDiskInvertedLists_new(&mut inner, nlist, code_size, f.as_ptr());
            FaissOnDiskInvLists(inner)
        }
    }

    pub fn merge_from_multiple(
        &mut self,
        ivfs: &[FaissInvLists],
        shift_ids: bool,
        verbose: bool,
    ) -> usize {
        unsafe {
            let mut ivfs = ivfs.iter().map(|ivf| ivf.0 as *const _).collect::<Vec<_>>();
            faiss_OnDiskInvertedLists_merge_from_multiple(
                self.0,
                ivfs.as_mut_ptr(),
                ivfs.len() as i32,
                shift_ids as i32,
                verbose as i32,
            ) as usize
        }
    }
}
