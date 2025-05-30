#![feature(portable_simd)]
#![feature(likely_unlikely)]
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::convert::TryInto;
use std::hint::black_box;
use std::simd::num::SimdUint;
use std::simd::*;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use imsearch::hamming::{hamming, hamming_256, hamming_naive, knn_hamming};
use rand::prelude::*;

#[inline(always)]
fn hamming_256_simd(va: &[u8], vb: &[u8]) -> u32 {
    // 测试表明，此处使用 unsafe 转换并不会更快
    let va: &[u8; 32] = va.try_into().unwrap();
    let vb: &[u8; 32] = vb.try_into().unwrap();
    let va = u8x32::from_slice(va);
    let vb = u8x32::from_slice(vb);
    (va ^ vb).count_ones().reduce_sum() as u32
}

fn bench_hamming(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hamming");
    let mut rng = rand::rng();
    let mut src = vec![0u8; 32];
    let mut dst = vec![0u8; 8 << 20];
    rng.fill_bytes(&mut src);
    rng.fill_bytes(&mut dst);

    group.throughput(Throughput::Bytes(dst.len() as u64));
    group.bench_function("hamming_256_naive", |b| {
        // NOTE: 这里 32 去掉 black_box 反而更慢
        // LLVM 会进行极致的循环展开，一次性处理 32 * 256 / 8 = 1024 个字节
        // 循环体被塞爆了，导致指令缓存失效
        b.iter(|| {
            dst.chunks_exact(black_box(32))
                .map(|chunk| hamming_naive::<256>(&src, chunk))
                .sum::<u32>()
        });
    });
    group.bench_function("hamming_256_unrolled", |b| {
        b.iter(|| dst.chunks_exact(32).map(|chunk| hamming_256(&src, chunk)).sum::<u32>());
    });
    group.bench_function("hamming_256_simd", |b| {
        b.iter(|| dst.chunks_exact(32).map(|chunk| hamming_256_simd(&src, chunk)).sum::<u32>());
    });
    group.finish();
}

fn bench_hamming_knn(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hamming KNN");
    let mut rng = rand::rng();
    let mut src = vec![0u8; 32];
    let mut dst = vec![0u8; 8 << 20];
    let k = black_box(3);
    rng.fill_bytes(&mut src);
    rng.fill_bytes(&mut dst);

    group.throughput(Throughput::Bytes(dst.len() as u64));
    group.bench_function("BinaryHeap", |b| {
        b.iter(|| {
            let mut heap = BinaryHeap::new();
            for (i, chunk) in dst.chunks_exact(32).enumerate() {
                let d = hamming::<256>(&src, chunk);
                if heap.len() < k {
                    heap.push(Reverse((d as u64) << 32 | i as u64));
                } else {
                    let Reverse(peek) = heap.peek().unwrap();
                    if d < (*peek >> 32) as u32 {
                        heap.pop();
                        heap.push(Reverse((d as u64) << 32 | i as u64));
                    }
                }
            }
            let Reverse(a) = heap.pop().unwrap();
            let Reverse(b) = heap.pop().unwrap();
            let Reverse(c) = heap.pop().unwrap();
            (a >> 32, b >> 32, c >> 32)
        });
    });
    group.bench_function("Array", |b| {
        b.iter(|| knn_hamming::<256>(&src, &dst, k));
    });
    group.finish();
}

criterion_group!(benches, bench_hamming, bench_hamming_knn);
criterion_main!(benches);
