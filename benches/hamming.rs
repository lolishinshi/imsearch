#![feature(portable_simd)]
use std::convert::TryInto;
use std::hint::black_box;
use std::simd::num::SimdUint;
use std::simd::*;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use imsearch::hamming::{hamming_32, hamming_naive, knn_hamming_array, knn_hamming_heap};
use rand::prelude::*;

#[inline(always)]
fn hamming_32_simd(va: &[u8], vb: &[u8]) -> u32 {
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
    group.bench_function("hamming_32_naive", |b| {
        // NOTE: 这里 32 去掉 black_box 反而更慢
        // LLVM 会进行极致的循环展开，一次性处理 32 * 256 / 8 = 1024 个字节
        // 循环体被塞爆了，导致指令缓存失效
        b.iter(|| {
            dst.chunks_exact(black_box(32))
                .map(|chunk| hamming_naive::<32>(&src, chunk))
                .sum::<u32>()
        });
    });
    group.bench_function("hamming_32_unrolled", |b| {
        b.iter(|| dst.chunks_exact(32).map(|chunk| hamming_32(&src, chunk)).sum::<u32>());
    });
    group.bench_function("hamming_32_simd", |b| {
        b.iter(|| dst.chunks_exact(32).map(|chunk| hamming_32_simd(&src, chunk)).sum::<u32>());
    });
    group.finish();
}

// https://users.rust-lang.org/t/can-i-conveniently-compile-bytes-into-a-rust-program-with-a-specific-alignment/24049/2
#[macro_use]
mod macros {
    #[repr(C)] // guarantee 'bytes' comes after '_align'
    pub struct AlignedAs<Align, Bytes: ?Sized> {
        pub _align: [Align; 0],
        pub bytes: Bytes,
    }

    macro_rules! include_bytes_align_as {
        ($align_ty:ty, $path:literal) => {{
            // const block expression to encapsulate the static
            use $crate::macros::AlignedAs;

            // this assignment is made possible by CoerceUnsized
            static ALIGNED: &AlignedAs<$align_ty, [u8]> =
                &AlignedAs { _align: [], bytes: *include_bytes!($path) };

            &ALIGNED.bytes
        }};
    }
}

fn bench_hamming_knn(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hamming KNN");
    let mut rng = rand::rng();
    let mut src = [0u8; 32];
    let dst: &[u8] = include_bytes_align_as!(u64, "./list.bin");
    let k = black_box(3);
    rng.fill_bytes(&mut src);

    group.throughput(Throughput::Bytes(dst.len() as u64));
    group.bench_function("BinaryHeap", |b| {
        let (dst, _) = dst.as_chunks::<32>();
        b.iter(|| knn_hamming_heap::<32>(&src, dst, k));
    });
    group.bench_function("Array", |b| {
        // NOTE: 这行代码如果移到 bench_function 外面会导致性能急剧下降
        // 原因暂不明确，在将 &[u8] 转为 &[[u8; N]] 之前没有观察到这个问题
        let (dst, _) = dst.as_chunks::<32>();
        b.iter(|| knn_hamming_array::<32>(&src, dst, k));
    });
    group.finish();
}

criterion_group!(benches, bench_hamming, bench_hamming_knn);
criterion_main!(benches);
