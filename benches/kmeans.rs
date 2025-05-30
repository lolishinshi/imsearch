use criterion::{Criterion, black_box, criterion_group, criterion_main};
use imsearch::kmeans::{binary_kmeans, binary_kmeans_2level};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// 生成有聚类模式的测试数据：256bit 向量
fn generate_clustered_data(n: usize, num_clusters: usize) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(42); // 使用固定种子确保结果可重现
    let d = 256 / 8; // 256bit = 32 bytes
    let mut data = vec![0u8; n * d];

    // 生成聚类中心模板
    let mut cluster_centers = vec![vec![0u8; d]; num_clusters];
    for center in &mut cluster_centers {
        rng.fill(&mut center[..]);
    }

    // 为每个向量分配到某个聚类，并在聚类中心附近生成数据
    for i in 0..n {
        let cluster_id = i % num_clusters;
        let base_center = &cluster_centers[cluster_id];

        // 在聚类中心附近生成数据（添加少量噪声）
        for j in 0..d {
            let noise_bits = rng.random::<u8>() & 0x0F; // 只改变低4位作为噪声
            data[i * d + j] = base_center[j] ^ noise_bits;
        }
    }

    data
}

fn bench_kmeans_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_256bit");

    // 定义测试参数组合
    let test_cases = black_box(vec![(7680, 256), (15360, 512), (30720, 1024)]);

    for (n, nc) in test_cases {
        // 生成有聚类模式的数据，聚类数量为目标聚类数量的一半
        let data = black_box(generate_clustered_data(n, nc / 2));

        group.bench_function(format!("binary_kmeans_{n}_{nc}"), |b| {
            b.iter(|| binary_kmeans::<256>(&data, n, nc, 50, false))
        });

        group.bench_function(format!("binary_kmeans_2level_{n}_{nc}"), |b| {
            b.iter(|| binary_kmeans_2level::<256>(&data, n, nc))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_kmeans_comparison);
criterion_main!(benches);
