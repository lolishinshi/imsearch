use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use imsearch::config::OrbOptions;
use imsearch::orb::ORBDetector;
use opencv::core::{KeyPoint, Vector};
use opencv::imgcodecs;
use opencv::imgproc::InterpolationFlags;
use opencv::prelude::*;

fn orb_detect(orb: &mut ORBDetector, img: &[u8]) -> (Mat, Vector<KeyPoint>, Mat) {
    orb.detect_bytes(img).unwrap()
}

fn imdecode(img: &[u8]) -> Mat {
    let mat = Mat::from_slice(img).unwrap();
    imgcodecs::imdecode(&mat, imgcodecs::IMREAD_GRAYSCALE).unwrap()
}

fn benchmark_image(c: &mut Criterion) {
    let mut orb = ORBDetector::create(OrbOptions {
        orb_nfeatures: 500,
        orb_scale_factor: 1.2,
        orb_nlevels: 8,
        orb_interpolation: InterpolationFlags::INTER_AREA,
        orb_ini_th_fast: 20,
        orb_min_th_fast: 7,
        orb_not_oriented: false,
        max_size: (1080, 768),
        max_aspect_ratio: 5.0,
        max_features: 1000,
    });

    let jpg = std::fs::read("benches/test.jpg").unwrap();
    let webp = std::fs::read("benches/test.webp").unwrap();

    let mut group = c.benchmark_group("图像处理");
    group.throughput(Throughput::Elements(1));
    group.bench_function("特征提取", |b| b.iter(|| orb_detect(&mut orb, black_box(&jpg))));
    group.bench_function("JPEG 解码", |b| b.iter(|| imdecode(black_box(&jpg))));
    group.bench_function("WebP 解码", |b| b.iter(|| imdecode(black_box(&webp))));
    group.finish();
}

criterion_group!(benches, benchmark_image);
criterion_main!(benches);
