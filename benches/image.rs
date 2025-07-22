use std::fs;
use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use imsearch::dhash::{DHash, d_hash};
use imsearch::orb::Slam3ORB;
use imsearch::utils;
use ndarray::Array2;
use opencv::core::KeyPoint;
use opencv::imgcodecs;
use opencv::imgproc::InterpolationFlags;
use opencv::prelude::*;

fn orb_detect(orb: &mut Slam3ORB, img: &Mat) -> (Vec<KeyPoint>, Array2<u8>) {
    utils::detect_and_compute(orb, img).unwrap()
}

fn imdecode(img: &[u8]) -> Mat {
    let mat = Mat::from_slice(img).unwrap();
    imgcodecs::imdecode(&mat, imgcodecs::IMREAD_GRAYSCALE).unwrap()
}

fn blake3_hash(img: &[u8]) -> Vec<u8> {
    blake3::hash(img).as_bytes().to_vec()
}

fn dhash_hash(img: &Mat) -> DHash {
    d_hash(img).unwrap()
}

fn benchmark_image(c: &mut Criterion) {
    let mut orb =
        Slam3ORB::create(500, 1.2, 8, 20, 7, InterpolationFlags::INTER_AREA, false).unwrap();

    let jpg = fs::read("benches/test.jpg").unwrap();
    let webp = fs::read("benches/test.webp").unwrap();

    let img = imgcodecs::imread("benches/test.jpg", imgcodecs::IMREAD_GRAYSCALE).unwrap();

    let mut group = c.benchmark_group("图像处理");
    group.throughput(Throughput::Elements(1));
    group.bench_function("特征提取", |b| b.iter(|| orb_detect(&mut orb, black_box(&img))));
    group.bench_function("JPEG 解码", |b| b.iter(|| imdecode(black_box(&jpg))));
    group.bench_function("WebP 解码", |b| b.iter(|| imdecode(black_box(&webp))));
    group.finish();
}

fn benchmark_hash(c: &mut Criterion) {
    let jpg = fs::read("benches/test.jpg").unwrap();
    let img = imgcodecs::imread("benches/test.jpg", imgcodecs::IMREAD_GRAYSCALE).unwrap();

    let mut group = c.benchmark_group("哈希计算");
    group.throughput(Throughput::Bytes(jpg.len() as u64));
    group.bench_function("BLAKE3", |b| b.iter(|| blake3_hash(black_box(&jpg))));
    group.bench_function("dHash", |b| b.iter(|| dhash_hash(black_box(&img))));
    group.finish();
}

criterion_group!(benches, benchmark_image, benchmark_hash);
criterion_main!(benches);
