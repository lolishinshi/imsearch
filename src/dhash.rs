use anyhow::Result;
use opencv::core::{Size, ToInputArray};
use opencv::imgproc;
use opencv::prelude::*;

pub type DHash = [u8; 8];

pub fn d_hash(input_arr: &impl ToInputArray) -> Result<DHash> {
    let mut resize_img = Mat::default();
    imgproc::resize(
        input_arr,
        &mut resize_img,
        Size::new(9, 8),
        0.0,
        0.0,
        imgproc::INTER_LINEAR_EXACT,
    )?;

    let gray_img = if resize_img.channels() > 1 {
        let mut output = Mat::default();
        imgproc::cvt_color_def(&resize_img, &mut output, imgproc::COLOR_BGR2GRAY)?;
        output
    } else {
        resize_img
    };

    let mut hash = [0; 8];
    let data = gray_img.data_bytes()?;
    assert!(data.len() == 72);
    // TODO: 此处是否需要 SIMD？
    for (i, chunk) in data.chunks_exact(9).enumerate() {
        let mut b = 0;
        for j in 0..8 {
            b <<= 1;
            b |= if chunk[j] < chunk[j + 1] { 1 } else { 0 };
        }
        hash[i] = b;
    }

    Ok(hash)
}
