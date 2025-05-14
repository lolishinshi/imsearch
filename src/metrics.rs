use std::sync::LazyLock;

use prometheus::*;

static METRIC_SEARCH_IMAGE_COUNT: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!("im_search_image_count", "count of the image to search", &["size"])
        .unwrap()
});

static METRIC_SEARCH_DURATION: LazyLock<Histogram> = LazyLock::new(|| {
    register_histogram!("im_search_duration", "duration of the per-image search in seconds")
        .unwrap()
});

static METRIC_SEARCH_MAX_SCORE: LazyLock<Histogram> = LazyLock::new(|| {
    register_histogram!(
        "im_search_max_score",
        "max score of the per-image search",
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    )
    .unwrap()
});

/// 增加图像大小指标计数
pub fn inc_image_count(size: (u32, u32)) {
    let (width, height) = size;

    let normalized_width = normalize_dimension(width);
    let normalized_height = normalize_dimension(height);

    let size_category = format!("{}x{}", normalized_width, normalized_height);

    METRIC_SEARCH_IMAGE_COUNT.with_label_values(&[&size_category]).inc();
}

pub fn inc_search_duration(duration: f64) {
    METRIC_SEARCH_DURATION.observe(duration);
}

pub fn inc_search_max_score(score: f64) {
    METRIC_SEARCH_MAX_SCORE.observe(score);
}

fn normalize_dimension(dimension: u32) -> u32 {
    if dimension <= 128 {
        128
    } else if dimension <= 256 {
        256
    } else if dimension <= 512 {
        512
    } else if dimension <= 768 {
        768
    } else if dimension <= 1024 {
        1024
    } else if dimension <= 1536 {
        1536
    } else {
        2048
    }
}
