use std::ffi::c_void;

use anyhow::Result;
use opencv::prelude::*;
use opencv::{core, sys};
use std::str::FromStr;

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
            _ => anyhow::bail!("Possible values: Liner, Lanczos4, Area, Lanczos4"),
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
        let raw = unsafe {
            slam3_ORB_create(
                nfeatures,
                scale_factor,
                nlevels,
                ini_th_fast,
                min_th_fast,
                interpolation as i32,
                angle,
            )
            .into_result()?
        };
        Ok(Self { raw })
    }

    pub fn default() -> Result<Self> {
        Self::create(500, 1.2, 8, 20, 7, InterpolationFlags::Area, true)
    }

    pub fn detect_and_compute(
        &mut self,
        image: &dyn core::ToInputArray,
        mask: &dyn core::ToInputArray,
        keypoints: &mut core::Vector<core::KeyPoint>,
        descriptors: &mut dyn core::ToOutputArray,
        v_lapping_area: &core::Vector<i32>,
    ) -> Result<()> {
        let image = image.input_array()?;
        let mask = mask.input_array()?;
        let descriptors = descriptors.output_array()?;
        unsafe {
            slam3_ORB_detect_and_compute(
                self.raw,
                image.as_raw__InputArray(),
                mask.as_raw__InputArray(),
                keypoints.as_raw_mut_VectorOfKeyPoint(),
                descriptors.as_raw__OutputArray(),
                v_lapping_area.as_raw_VectorOfi32(),
            )
            .into_result()?
        }
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

extern "C" {
    fn slam3_ORB_create(
        nfeatures: i32,
        scale_factor: f32,
        nlevels: i32,
        ini_th_fast: i32,
        min_th_fast: i32,
        interpolation: i32,
        angle: bool,
    ) -> sys::Result<*const c_void>;
    fn slam3_ORB_delete(orb: *const c_void);
    fn slam3_ORB_detect_and_compute(
        orb: *const c_void,
        image: *const c_void,
        mask: *const c_void,
        keypoints: *const c_void,
        descriptors: *const c_void,
        v_lapping_area: *const c_void,
    ) -> sys::Result_void;
}

#[cfg(test)]
mod test {
    use super::Slam3ORB;
    use opencv::features2d;
    use opencv::imgcodecs;
    use opencv::prelude::*;

    #[test]
    fn detect_and_compute() {
        let img =
            imgcodecs::imread("./cache/box_in_scene.png", imgcodecs::IMREAD_GRAYSCALE).unwrap();
        let mask = Mat::default();
        let lap = opencv::types::VectorOfi32::from(vec![0, 0]);
        let mut kps = opencv::types::VectorOfKeyPoint::new();
        let mut des = Mat::default();
        let mut orb = Slam3ORB::default().unwrap();
        orb.detect_and_compute(&img, &mask, &mut kps, &mut des, &lap)
            .unwrap();

        let mut output = Mat::default();
        features2d::draw_keypoints(
            &img,
            &kps,
            &mut output,
            opencv::core::Scalar::all(-1.0),
            features2d::DrawMatchesFlags::DEFAULT,
        );

        let flags = opencv::types::VectorOfi32::from(vec![
            imgcodecs::ImwriteFlags::IMWRITE_PNG_COMPRESSION as i32,
            9,
        ]);
        imgcodecs::imwrite("slam3_orb.png", &output, &flags).unwrap();
    }
}
