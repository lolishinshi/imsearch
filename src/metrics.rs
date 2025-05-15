use std::sync::LazyLock;

use prometheus::*;

static METRIC_SEARCH_IMAGE_COUNT: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        "im_search_image_count",
        "count of the image to search",
        &["size", "nprobe", "orb_scale_factor"]
    )
    .unwrap()
});

static METRIC_SEARCH_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "im_search_duration",
        "duration of the per-image search in seconds",
        &["size", "nprobe", "orb_scale_factor"]
    )
    .unwrap()
});

static METRIC_SEARCH_MAX_SCORE: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "im_search_max_score",
        "max score of the per-image search",
        &["size", "nprobe", "orb_scale_factor"],
        (5..=100).step_by(5).map(|x| x as f64).collect()
    )
    .unwrap()
});

/// 增加图像大小指标计数
pub fn inc_image_count(size: (u32, u32), nprobe: usize, orb_scale_factor: f32) {
    let size = to_fixed_size(size);

    METRIC_SEARCH_IMAGE_COUNT
        .with_label_values(&[size, &nprobe.to_string(), &orb_scale_factor.to_string()])
        .inc();
}

pub fn inc_search_duration(size: (u32, u32), nprobe: usize, orb_scale_factor: f32, duration: f32) {
    let size = to_fixed_size(size);

    METRIC_SEARCH_DURATION
        .with_label_values(&[size, &nprobe.to_string(), &orb_scale_factor.to_string()])
        .observe(duration as f64);
}

pub fn inc_search_max_score(size: (u32, u32), nprobe: usize, orb_scale_factor: f32, score: f32) {
    let size = to_fixed_size(size);

    METRIC_SEARCH_MAX_SCORE
        .with_label_values(&[size, &nprobe.to_string(), &orb_scale_factor.to_string()])
        .observe(score as f64);
}

/// 将图像面积范围调整到几个固定值
fn to_fixed_size((width, height): (u32, u32)) -> &'static str {
    let area = width * height;
    if area <= 128 * 128 {
        "128"
    } else if area <= 256 * 256 {
        "256"
    } else if area <= 512 * 512 {
        "512"
    } else if area <= 768 * 768 {
        "768"
    } else if area <= 1024 * 1024 {
        "1024"
    } else if area <= 1536 * 1536 {
        "1536"
    } else if area <= 2048 * 2048 {
        "2048"
    } else {
        "2048+"
    }
}
