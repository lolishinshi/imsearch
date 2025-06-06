use indicatif::{ProgressBar, ProgressIterator};
use kmeans::{EuclideanDistance, KMeans, KMeansConfig, KMeansState};
use log::info;

use crate::hamming::batch_knn_hamming;
use crate::utils::pb_style;

fn binary_to_real(x_in: &[u8]) -> Vec<f32> {
    let d = x_in.len() * 8;
    let mut v = vec![0.0; d];
    for i in 0..d {
        let bit_value = (x_in[i >> 3] >> (i & 7)) & 1;
        v[i] = (2 * bit_value as i32 - 1) as f32;
    }
    v
}

fn real_to_binary(x_in: &[f32]) -> Vec<u8> {
    assert!(x_in.len() % 8 == 0, "d must be a multiple of 8");
    let d = x_in.len();
    let mut v = vec![0; d / 8];
    for i in 0..d / 8 {
        let mut b = 0;
        for j in 0..8 {
            if x_in[i * 8 + j] > 0.0 {
                b |= 1 << j;
            }
        }
        v[i] = b;
    }
    v
}

pub fn imbalance_factor(hist: &[usize]) -> f32 {
    let (mut tot, mut uf) = (0.0, 0.0);
    for h in hist {
        let h = *h as f32;
        tot += h;
        uf += h.powf(2.0);
    }
    uf * hist.len() as f32 / tot.powf(2.0)
}

/// 使用 kmeans 聚类，返回聚类结果
///
/// 参数：
/// - x: 输入向量
/// - nc: 聚类中心数量
/// - max_iter: 最大迭代次数
/// - verbose: 是否打印详细信息
pub fn binary_kmeans<const N: usize>(
    x: &[[u8; N]],
    nc: usize,
    max_iter: usize,
    verbose: bool,
) -> Vec<[u8; N]> {
    let n = x.len();
    // KMeans 只能对浮点进行聚类，因此这里将二进制转成浮点向量
    // TODO: 这里距离函数可以自定义，理论上可以直接对二进制进行聚类
    let x = binary_to_real(x.as_flattened());
    let km: KMeans<_, 16, _> = KMeans::new(&x, n, N * 8, EuclideanDistance);
    let conf = if verbose {
        KMeansConfig::build()
            .init_done(&|_s: &KMeansState<f32>| info!("KMeans 初始化完成"))
            .iteration_done(&|s: &KMeansState<f32>, nr: usize, new_distsum: f32| {
                info!(
                    "第 {} 轮 - 不平衡度：{:.2} | 距离和变化：{:+.2}",
                    nr,
                    imbalance_factor(&s.centroid_frequency),
                    new_distsum - s.distsum
                );
            })
            .build()
    } else {
        KMeansConfig::default()
    };
    // NOTE: init_kmeanplusplus 会 panic，不知道为啥
    let result = km.kmeans_lloyd(nc, max_iter, KMeans::init_random_partition, &conf);
    let b = real_to_binary(&result.centroids.to_vec());
    b.chunks_exact(N).map(|x| x.try_into().unwrap()).collect()
}

/// 使用 kmeans 进行二级聚类，损失少许精度大幅提高速度
pub fn binary_kmeans_2level<const N: usize>(
    x: &[[u8; N]],
    nc: usize,
    max_iter: usize,
) -> Vec<[u8; N]> {
    let n = x.len();
    assert!(n >= 30 * nc, "向量数量必须大于 30 * {nc}");
    let nc1 = nc.isqrt();

    let n1 = (nc1 * 1024).min(n);
    info!("对 {n1} 组向量进行 1 级聚类，中心点数量 = {nc1}");
    let c1 = binary_kmeans::<N>(&x[..n1], nc1, max_iter, true);

    info!("根据 1 级聚类结果划分训练集");
    let r = batch_knn_hamming::<N>(x, &c1, 1);

    // 一级聚类中，每个聚类中心分配到的向量列表
    let mut xc = vec![vec![]; nc1];
    r.iter().enumerate().for_each(|(i, r)| {
        let n = r[0].0;
        xc[n].push(x[i]);
    });

    // 计算累加和，用于计算二级聚类中心点数量
    let bc_sum = xc
        .iter()
        .scan(0, |acc, x| {
            *acc += x.len();
            Some(*acc)
        })
        .collect::<Vec<_>>();

    // TODO: 测试加权分配 nc2 和使用固定值，哪个更好
    // 此处使用了累加和+错位相减来进行加权分配，这样可以保证 sum(nc2) = nc
    let mut nc2 = bc_sum.iter().map(|x| x * nc / bc_sum[bc_sum.len() - 1]).collect::<Vec<_>>();
    for i in (1..nc2.len()).rev() {
        nc2[i] -= nc2[i - 1];
    }
    assert_eq!(nc2.iter().sum::<usize>(), nc);

    let min_nc2 = nc2.iter().min().unwrap();
    let max_nc2 = nc2.iter().max().unwrap();
    info!("2 级聚类中心点数量：{min_nc2} ~ {max_nc2}");

    let mut c = vec![];
    let pb = ProgressBar::new(nc1 as u64).with_style(pb_style());
    for i in (0..nc1).progress_with(pb.clone()) {
        let x = &xc[i];
        pb.set_message(format!("对 {} 组向量进行二级聚类，中心点数量 = {}", x.len(), nc2[i]));
        if nc2[i] > 0 {
            let c2 = binary_kmeans::<N>(x, nc2[i], max_iter, false);
            c.extend(c2);
        }
    }
    pb.finish_with_message("二级聚类完成");

    assert_eq!(c.len(), nc);

    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_to_real() {
        let binary = vec![0b10110001];
        let result = binary_to_real(&binary);
        let expected = vec![1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_real_to_binary() {
        let real = vec![1.0, -1.0, 0.5, -0.1, 2.0, -2.0, 0.0, 3.0];
        let result = real_to_binary(&real);
        assert_eq!(result, vec![0b10010101]);
    }

    #[test]
    fn test_roundtrip_conversion() {
        let original = vec![0xAA, 0x55];
        let real = binary_to_real(&original);
        let converted = real_to_binary(&real);
        assert_eq!(original, converted);
    }

    #[test]
    #[should_panic(expected = "d must be a multiple of 8")]
    fn test_invalid_length() {
        // 测试长度不是 8 的倍数时的错误处理
        real_to_binary(&vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_binary_kmeans_basic() {
        // 创建一些简单的测试数据：两个明显不同的聚类
        let mut x = vec![[0u8; 4]; 16]; // 16个向量，每个4字节

        // 前8个向量设为模式1 (前半部分为0，后半部分为255)
        for i in 0..8 {
            x[i][2] = 0xFF;
            x[i][3] = 0xFF;
        }

        // 后8个向量设为模式2 (前半部分为255，后半部分为0)
        for i in 8..16 {
            x[i][0] = 0xFF;
            x[i][1] = 0xFF;
        }

        let centroids = binary_kmeans::<4>(&x, 2, 50, false);

        // 应该返回2个聚类中心
        assert_eq!(centroids.len(), 2);

        let c1 = &centroids[0];
        let c2 = &centroids[1];

        assert_ne!(c1, c2); // 两个中心应该不同
    }

    #[test]
    fn test_binary_kmeans_2level() {
        let nc = 4;
        let n = 32 * nc;

        // 创建有4个明显聚类的数据
        let mut x = vec![];

        for _ in 0..(n / 4) {
            x.push([0x00; 4]);
            x.push([0x77; 4]);
            x.push([0xAA; 4]);
            x.push([0xFF; 4]);
        }

        let mut centroids = binary_kmeans_2level::<4>(&x, nc, 50);
        centroids.sort_by_key(|x| x[0]);

        // 应该返回nc个聚类中心
        assert_eq!(centroids.len(), nc);

        // 验证返回的中心数量是正确的

        assert_eq!(&centroids[0], &[0x00; 4]);
        assert_eq!(&centroids[1], &[0x77; 4]);
        assert_eq!(&centroids[2], &[0xAA; 4]);
        assert_eq!(&centroids[3], &[0xFF; 4]);
    }

    #[test]
    fn test_binary_kmeans_single_cluster() {
        let x = vec![[66u8; 4]; 8];
        let centroids = binary_kmeans::<4>(&x, 1, 10, false);
        assert_eq!(centroids.len(), 1);
        assert_eq!(&centroids[0], &[66u8; 4]);
    }

    #[test]
    #[should_panic]
    fn test_binary_kmeans_invalid_length() {
        let x = vec![[0u8; 4]; 8];
        binary_kmeans_2level::<4>(&x, 1, 50);
    }
}
