use opencv::sys::Result;
use std::ffi::c_void;

unsafe extern "C" {
    pub fn slam3_ORB_create(
        nfeatures: i32,
        scale_factor: f32,
        nlevels: i32,
        ini_th_fast: i32,
        min_th_fast: i32,
        interpolation: i32,
        angle: bool,
        ocvrs_return: *mut Result<*mut c_void>,
    );
    pub fn slam3_ORB_delete(instance: *mut c_void);
    pub fn slam3_ORB_detect_and_compute(
        orb: *const c_void,
        image: *const c_void,
        mask: *const c_void,
        keypoints: *const c_void,
        descriptors: *const c_void,
        v_lapping_area: *const c_void,
        ocvrs_return: *mut Result<()>,
    );
}
