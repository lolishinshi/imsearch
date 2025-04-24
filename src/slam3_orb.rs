use std::ffi::{c_int, c_void};

use anyhow::{Result, anyhow, bail};
use opencv::core::*;
use std::str::FromStr;

// C++与Rust交互的结果结构体
#[repr(C)]
pub struct RawResult<T> {
    pub error_code: i32,
    pub error_msg: *mut c_void,
    pub result: T,
}

impl<T> RawResult<T> {
    /// 将 RawResult<T> 转换为 Rust 的 Result<T, anyhow::Error>
    pub fn into_result(self) -> Result<T> {
        if self.error_code != 0 {
            let error_string = if !self.error_msg.is_null() {
                unsafe {
                    let msg = self.error_msg as *const String;
                    format!("错误码: {}, 错误信息: {}", self.error_code, *msg)
                }
            } else {
                format!("错误码: {}", self.error_code)
            };

            return Err(anyhow!(error_string));
        }

        Ok(self.result)
    }
}

#[repr(C)]
pub struct RawResultVoid {
    pub error_code: i32,
    pub error_msg: *mut c_void,
}

impl RawResultVoid {
    /// 将 RawResultVoid 转换为 Rust 的 Result<(), anyhow::Error>
    pub fn into_result(self) -> Result<()> {
        if self.error_code != 0 {
            let error_string = if !self.error_msg.is_null() {
                unsafe {
                    let msg = self.error_msg as *const String;
                    format!("错误码: {}, 错误信息: {}", self.error_code, *msg)
                }
            } else {
                format!("错误码: {}", self.error_code)
            };

            return Err(anyhow!(error_string));
        }

        Ok(())
    }
}

// TODO: 使用 OpenCV 自带的
#[derive(Debug, Copy, Clone)]
pub enum InterpolationFlags {
    Liner = 1,
    Cubic = 2,
    Area = 3,
    Lanczos4 = 4,
}

impl FromStr for InterpolationFlags {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::prelude::rust_2015::Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "liner" => Self::Liner,
            "cubic" => Self::Cubic,
            "area" => Self::Area,
            "lanczos4" => Self::Lanczos4,
            _ => bail!("Possible values: Liner, Lanczos4, Area, Lanczos4"),
        })
    }
}

pub struct Slam3ORB {
    raw: *const c_void,
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
        let mut return_value = std::mem::MaybeUninit::uninit();
        unsafe {
            slam3_ORB_create(
                nfeatures,
                scale_factor,
                nlevels,
                ini_th_fast,
                min_th_fast,
                interpolation as i32,
                angle,
                return_value.as_mut_ptr(),
            )
        }

        let raw_result = unsafe { return_value.assume_init() };
        let raw = raw_result.into_result()?;

        Ok(Self { raw })
    }

    pub fn detect_and_compute(
        &mut self,
        image: &dyn ToInputArray,
        mask: &dyn ToInputArray,
        keypoints: &mut Vector<KeyPoint>,
        descriptors: &mut dyn ToOutputArray,
        v_lapping_area: &Vector<i32>,
    ) -> Result<()> {
        let image = image.input_array()?;
        let mask = mask.input_array()?;
        let descriptors = descriptors.output_array()?;

        let mut return_value = std::mem::MaybeUninit::uninit();
        unsafe {
            slam3_ORB_detect_and_compute(
                self.raw,
                image.as_raw__InputArray(),
                mask.as_raw__InputArray(),
                keypoints.as_raw_mut_VectorOfKeyPoint(),
                descriptors.as_raw__OutputArray(),
                v_lapping_area.as_raw_VectorOfi32(),
                return_value.as_mut_ptr(),
            )
        }

        let raw_result = unsafe { return_value.assume_init() };
        raw_result.into_result()?;

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

impl Default for Slam3ORB {
    fn default() -> Self {
        Self::create(500, 1.2, 8, 20, 7, InterpolationFlags::Area, true).unwrap()
    }
}

unsafe impl Sync for Slam3ORB {}
unsafe impl Send for Slam3ORB {}

unsafe extern "C" {
    fn slam3_ORB_create(
        nfeatures: i32,
        scale_factor: f32,
        nlevels: i32,
        ini_th_fast: i32,
        min_th_fast: i32,
        interpolation: i32,
        angle: bool,
        ocvrs_return: *mut RawResult<*const c_void>,
    );
    fn slam3_ORB_delete(orb: *const c_void);
    fn slam3_ORB_detect_and_compute(
        orb: *const c_void,
        image: *const c_void,
        mask: *const c_void,
        keypoints: *const c_void,
        descriptors: *const c_void,
        v_lapping_area: *const c_void,
        ocvrs_return: *mut RawResult<c_int>,
    );
}

#[cfg(test)]
mod test {
    use super::Slam3ORB;
    use opencv::core::*;
    use opencv::features2d;
    use opencv::imgcodecs;

    #[test]
    fn detect_and_compute() {
        let img =
            imgcodecs::imread("./cache/box_in_scene.png", imgcodecs::IMREAD_GRAYSCALE).unwrap();
        let mask = Mat::default();
        let lap = Vector::<i32>::from(vec![0, 0]);
        let mut kps = Vector::<KeyPoint>::new();
        let mut des = Mat::default();
        let mut orb = Slam3ORB::default().unwrap();
        orb.detect_and_compute(&img, &mask, &mut kps, &mut des, &lap).unwrap();

        let mut output = Mat::default();
        features2d::draw_keypoints(
            &img,
            &kps,
            &mut output,
            Scalar::all(-1.0),
            features2d::DrawMatchesFlags::DEFAULT,
        )
        .unwrap();

        let flags =
            Vector::<i32>::from(vec![imgcodecs::ImwriteFlags::IMWRITE_PNG_COMPRESSION as i32, 9]);
        imgcodecs::imwrite("slam3_orb.png", &output, &flags).unwrap();
    }
}
