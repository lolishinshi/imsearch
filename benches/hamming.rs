#![feature(portable_simd)]
#![feature(likely_unlikely)]
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::convert::TryInto;
use std::hint::black_box;
use std::simd::num::SimdUint;
use std::simd::*;

use bytemuck::cast_slice;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;

#[inline(always)]
fn hamming_256_naive(va: &[u8], vb: &[u8]) -> u32 {
    let mut sum = 0;
    for i in 0..32 {
        sum += (va[i] ^ vb[i]).count_ones();
    }
    sum
}

#[inline(always)]
fn hamming_256_unrolled(va: &[u8], vb: &[u8]) -> u32 {
    // 测试表明，此处使用 unsafe 转换并不会更快
    //let va: &[u64] = unsafe { std::slice::from_raw_parts(va.as_ptr() as *const u64, 4) };
    //let vb: &[u64] = unsafe { std::slice::from_raw_parts(vb.as_ptr() as *const u64, 4) };
    let va: &[u64] = cast_slice(va);
    let vb: &[u64] = cast_slice(vb);
    (va[0] ^ vb[0]).count_ones()
        + (va[1] ^ vb[1]).count_ones()
        + (va[2] ^ vb[2]).count_ones()
        + (va[3] ^ vb[3]).count_ones()
}

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
        b.iter(|| {
            dst.chunks_exact(black_box(32)).map(|chunk| hamming_256_naive(&src, chunk)).sum::<u32>()
        });
    });
    group.bench_function("hamming_256_unrolled", |b| {
        b.iter(|| {
            dst.chunks_exact(black_box(32))
                .map(|chunk| hamming_256_unrolled(&src, chunk))
                .sum::<u32>()
        });
    });
    group.bench_function("hamming_256_simd", |b| {
        b.iter(|| {
            dst.chunks_exact(black_box(32)).map(|chunk| hamming_256_simd(&src, chunk)).sum::<u32>()
        });
    });
    group.finish();
}

fn bench_hamming_knn(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hamming KNN");
    let mut rng = rand::rng();
    let mut src = vec![0u8; 32];
    let mut dst = vec![0u8; 8 << 20];
    rng.fill_bytes(&mut src);
    rng.fill_bytes(&mut dst);

    group.throughput(Throughput::Bytes(dst.len() as u64));
    group.bench_function("BinaryHeap", |b| {
        b.iter(|| {
            let mut heap = BinaryHeap::new();
            for (i, chunk) in dst.chunks_exact(black_box(32)).enumerate() {
                let d = hamming_256_unrolled(&src, chunk);
                if heap.len() < black_box(3) {
                    heap.push(Reverse(d << 16 | i as u32));
                } else {
                    let Reverse(peek) = heap.peek().unwrap();
                    if d < (*peek) >> 16 {
                        heap.pop();
                        heap.push(Reverse(d << 16 | i as u32));
                    }
                }
            }
            let Reverse(a) = heap.pop().unwrap();
            let Reverse(b) = heap.pop().unwrap();
            let Reverse(c) = heap.pop().unwrap();
            (a >> 16, b >> 16, c >> 16)
        });
    });
    group.bench_function("Array", |b| {
        b.iter(|| {
            let mut arr = [u32::MAX; 10];
            let mut idx = [0; 10];
            let k = black_box(3);
            assert!(k <= arr.len());
            for (i, chunk) in dst.chunks_exact(black_box(32)).enumerate() {
                let d = hamming_256_unrolled(&src, chunk);
                if d > arr[0] {
                    continue;
                }
                for j in (0..k).rev() {
                    if d < arr[j] {
                        //arr[j..].rotate_right(1);
                        arr[j] = d;
                        //idx[j..].rotate_right(1);
                        idx[j] = i as u32;
                        break;
                    }
                }
            }
            (arr[0], arr[1], arr[2])
        });
    });
    group.finish();
}

criterion_group!(benches, bench_hamming, bench_hamming_knn);
criterion_main!(benches);
