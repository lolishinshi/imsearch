use std::ffi::c_void;

use opencv::prelude::*;
use opencv::Result;
use opencv::{core, sys};

pub struct Slam2ORB {
    raw: *const c_void,
}

impl Slam2ORB {
    pub fn create(
        nfeatures: i32,
        scale_factor: f32,
        nlevels: i32,
        ini_th_fast: i32,
        min_th_fast: i32,
    ) -> Result<Self> {
        let raw = unsafe {
            slam2_ORB_create(nfeatures, scale_factor, nlevels, ini_th_fast, min_th_fast)
                .into_result()?
        };
        Ok(Self { raw })
    }

    fn default() -> Result<Self> {
        Self::create(500, 1.2, 8, 31, 20)
    }

    pub fn detect_and_compute(
        &mut self,
        image: &dyn core::ToInputArray,
        mask: &dyn core::ToInputArray,
        keypoints: &mut core::Vector<core::KeyPoint>,
        descriptors: &mut dyn core::ToOutputArray,
    ) -> opencv::Result<()> {
        let image = image.input_array()?;
        let mask = mask.input_array()?;
        let descriptors = descriptors.output_array()?;
        unsafe {
            slam2_ORB_detect_and_compute(
                self.raw,
                image.as_raw__InputArray(),
                mask.as_raw__InputArray(),
                keypoints.as_raw_mut_VectorOfKeyPoint(),
                descriptors.as_raw__OutputArray(),
            )
            .into_result()?
        }
        Ok(())
    }
}

impl Drop for Slam2ORB {
    fn drop(&mut self) {
        unsafe {
            slam2_ORB_delete(self.raw);
        }
    }
}

extern "C" {
    fn slam2_ORB_create(
        nfeatures: i32,
        scale_factor: f32,
        nlevels: i32,
        ini_th_fast: i32,
        min_th_fast: i32,
    ) -> sys::Result<*const c_void>;
    fn slam2_ORB_delete(orb: *const c_void);
    fn slam2_ORB_detect_and_compute(
        orb: *const c_void,
        image: *const c_void,
        mask: *const c_void,
        keypoints: *const c_void,
        descriptors: *const c_void,
    ) -> sys::Result_void;
}

#[cfg(test)]
mod test {
    extern crate test;
    use super::Slam2ORB;
    use opencv::features2d;
    use opencv::imgcodecs;
    use opencv::prelude::*;
    use test::Bencher;

    #[test]
    fn detect_and_compute() {
        let img =
            imgcodecs::imread("./cache/box_in_scene.png", imgcodecs::IMREAD_GRAYSCALE).unwrap();
        let mask = Mat::default();
        let mut kps = opencv::types::VectorOfKeyPoint::new();
        let mut lap = opencv::types::VectorOfi32::from(vec![0, 0]);
        let mut des = Mat::default();
        let mut orb = Slam2ORB::default().unwrap();
        orb.detect_and_compute(&img, &mask, &mut kps, &mut des)
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
        imgcodecs::imwrite("slam2_orb.png", &output, &flags).unwrap();
    }

    #[bench]
    fn bench_detect_and_compute(b: &mut Bencher) {
        let img =
            imgcodecs::imread("./cache/box_in_scene.png", imgcodecs::IMREAD_GRAYSCALE).unwrap();
        let mask = Mat::default();
        let mut orb = Slam2ORB::default().unwrap();

        b.iter(|| {
            let mut kps = opencv::types::VectorOfKeyPoint::new();
            let mut des = Mat::default();
            orb.detect_and_compute(&img, &mask, &mut kps, &mut des)
                .unwrap();
        });
    }
}
