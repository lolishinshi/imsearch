use std::ffi::c_void;

use opencv::Result;
use opencv::core::*;
use opencv::imgproc::InterpolationFlags;
use orb_slam3_sys::*;

// OpenCV 的辅助宏
macro_rules! return_send {
    (via $name: ident) => {
        let mut $name = ::std::mem::MaybeUninit::uninit();
    };
}

macro_rules! return_receive {
    ($name_via: ident => $name: ident) => {
        let $name = unsafe { $name_via.assume_init() };
    };
}

macro_rules! input_array_arg {
    ($name: ident) => {
        let $name = $name.input_array()?;
    };
}

macro_rules! output_array_arg {
    ($name: ident) => {
        let $name = $name.output_array()?;
    };
}

pub struct Slam3ORB {
    raw: *mut c_void,
}

impl Slam3ORB {
    pub fn create(
        nfeatures: i32,
        scale_factor: f32,
        nlevels: i32,
        ini_th_fast: i32,
        min_th_fast: i32,
        interpolation: InterpolationFlags,
        angle: bool,
    ) -> Result<Self> {
        return_send!(via ocvrs_return);
        unsafe {
            slam3_ORB_create(
                nfeatures,
                scale_factor,
                nlevels,
                ini_th_fast,
                min_th_fast,
                interpolation as i32,
                angle,
                ocvrs_return.as_mut_ptr(),
            )
        }
        return_receive!(ocvrs_return => ret);
        let ret = ret.into_result()?;
        Ok(Self { raw: ret })
    }

    pub fn detect_and_compute(
        &mut self,
        image: &dyn ToInputArray,
        mask: &dyn ToInputArray,
        keypoints: &mut Vector<KeyPoint>,
        descriptors: &mut dyn ToOutputArray,
        v_lapping_area: &Vector<i32>,
    ) -> Result<()> {
        input_array_arg!(image);
        input_array_arg!(mask);
        output_array_arg!(descriptors);
        return_send!(via ocvrs_return);
        unsafe {
            slam3_ORB_detect_and_compute(
                self.raw,
                image.as_raw__InputArray(),
                mask.as_raw__InputArray(),
                keypoints.as_raw_mut_VectorOfKeyPoint(),
                descriptors.as_raw__OutputArray(),
                v_lapping_area.as_raw_VectorOfi32(),
                ocvrs_return.as_mut_ptr(),
            )
        }
        return_receive!(ocvrs_return => ret);
        let ret = ret.into_result()?;
        Ok(ret)
    }
}

impl Drop for Slam3ORB {
    fn drop(&mut self) {
        unsafe {
            slam3_ORB_delete(self.raw);
        }
    }
}

unsafe impl Sync for Slam3ORB {}
unsafe impl Send for Slam3ORB {}
