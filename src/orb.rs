use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::OnceLock;

use opencv::Result;
use opencv::core::*;
use opencv::imgproc::InterpolationFlags;
use orb_slam3_sys::*;

use crate::config::OrbOptions;
use crate::utils;

// 注意：ORB_OPTIONS 必须在 ORB 之前初始化
pub static ORB_OPTIONS: OnceLock<OrbOptions> = OnceLock::new();

thread_local! {
    pub static ORB: RefCell<ORBDetector> = RefCell::new(ORBDetector::create(ORB_OPTIONS.get().unwrap().clone()));
}

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
        image: &impl ToInputArray,
        mask: &impl ToInputArray,
        keypoints: &mut Vector<KeyPoint>,
        descriptors: &mut impl ToOutputArray,
    ) -> Result<()> {
        let v_lapping_area = Vector::<i32>::from(vec![0, 0]);
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
        ret.into_result()?;
        Ok(())
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

pub struct ORBDetector {
    orb: HashMap<i32, Slam3ORB>,
    opts: OrbOptions,
}

impl ORBDetector {
    pub fn create(options: OrbOptions) -> Self {
        Self { orb: HashMap::new(), opts: options }
    }

    fn get_nfeatures(&self, image: &Mat) -> i32 {
        // 如果长宽都小于最大尺寸，则使用默认的特征点数量
        if image.cols() < self.opts.max_size.0 && image.rows() < self.opts.max_size.1 {
            return self.opts.orb_nfeatures as i32;
        }
        // 否则按 aspect_ratio 等比增加，最小幅度 100
        let min = image.cols().min(image.rows());
        let max = image.cols().max(image.rows());
        let aspect_ratio = max as f32 / min as f32;
        if aspect_ratio > self.opts.max_aspect_ratio {
            let ratio = aspect_ratio / self.opts.max_aspect_ratio - 1.;
            // 计算需要增加的特征点数量
            let extra_nfeatures =
                (self.opts.orb_nfeatures as f32 * ratio / 100.).round() as i32 * 100;
            let nfeatures = self.opts.orb_nfeatures as i32 + extra_nfeatures;
            return nfeatures.min(self.opts.max_features as i32);
        }
        self.opts.orb_nfeatures as i32
    }

    fn get_orb(&mut self, image: &Mat) -> &mut Slam3ORB {
        let nfeatures = self.get_nfeatures(image);
        self.orb.entry(nfeatures).or_insert_with(|| {
            Slam3ORB::create(
                nfeatures,
                self.opts.orb_scale_factor,
                self.opts.orb_nlevels as i32,
                self.opts.orb_ini_th_fast as i32,
                self.opts.orb_min_th_fast as i32,
                self.opts.orb_interpolation,
                !self.opts.orb_not_oriented,
            )
            .unwrap()
        })
    }

    pub fn detect_file(&mut self, path: &str) -> Result<(Mat, Vec<KeyPoint>, Vec<[u8; 32]>)> {
        let image = utils::imread(path, self.opts.max_size)?;
        let orb = self.get_orb(&image);
        let (keypoints, descriptors) = utils::detect_and_compute(orb, &image)?;
        Ok((image, keypoints, descriptors))
    }

    pub fn detect_bytes(&mut self, bytes: &[u8]) -> Result<(Vec<KeyPoint>, Vec<[u8; 32]>)> {
        let image = utils::imdecode(bytes, self.opts.max_size)?;
        let orb = self.get_orb(&image);
        let (keypoints, descriptors) = utils::detect_and_compute(orb, &image)?;
        Ok((keypoints, descriptors))
    }

    pub fn detect_image(&mut self, image: Mat) -> Result<(Vec<KeyPoint>, Vec<[u8; 32]>)> {
        let image = utils::adjust_image_size(image, self.opts.max_size)?;
        let orb = self.get_orb(&image);
        let (keypoints, descriptors) = utils::detect_and_compute(orb, &image)?;
        Ok((keypoints, descriptors))
    }
}
