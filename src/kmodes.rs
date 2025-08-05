use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use log::info;
use rand::prelude::*;
use rand::rng;
use rayon::prelude::*;

use crate::hamming::{batch_knn_hamming, hamming};
use crate::utils::pb_style;

pub fn kmodes_2level<const N: usize>(x: &[[u8; N]], nc: usize, max_iter: usize) -> KModeState<N> {
    let n = x.len();
    assert!(n >= 30 * nc, "向量数量必须大于 30 * {nc}");
    let nc1 = nc.isqrt();

    // 没有必要用全部向量进行一级聚类，这里取 nc1 的 1024 倍来训练，平衡精度和耗时
    let n1 = (nc1 * 1024).min(n);
    info!("对 {n1} 组向量进行 1 级聚类，中心点数量 = {nc1}");
    let ks = kmodes_binary::<N>(&x[..n1], nc1, max_iter);
    info!("1 级聚类完成，不平衡度：{:.2}", imbalance_factor(&ks.centroid_frequency));

    info!("根据 1 级聚类结果划分训练集");
    let (r, _) = update_assignments(x, &ks.centroids);

    // 一级聚类中，每个聚类中心分配到的向量列表
    let mut xc = vec![vec![]; nc1];
    r.iter().enumerate().for_each(|(i, r)| {
        xc[*r].push(x[i]);
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

    let mut fks = KModeState::default();
    let pb = ProgressBar::new(nc1 as u64).with_style(pb_style());
    for i in (0..nc1).progress_with(pb.clone()) {
        let x = &xc[i];
        if nc2[i] > 0 {
            let ks = kmodes_binary::<N>(x, nc2[i], max_iter);
            let factor = imbalance_factor(&ks.centroid_frequency);
            pb.set_message(format!(
                "对 {} 组向量进行二级聚类，中心点数量 = {}, 不平衡度 = {factor:.2}",
                x.len(),
                nc2[i]
            ));
            fks.distsum += ks.distsum;
            fks.centroids.extend(ks.centroids);
            fks.centroid_frequency.extend(ks.centroid_frequency);
        }
    }
    pb.finish_with_message("二级聚类完成");

    assert_eq!(fks.centroids.len(), nc);

    info!("总距离：{}，不平衡度：{:.2}", fks.distsum, imbalance_factor(&fks.centroid_frequency));

    fks
}

#[derive(Debug, Clone, Default)]
pub struct KModeState<const N: usize> {
    /// 聚类中心到所有向量的总距离
    pub distsum: u32,
    /// 聚类中心
    pub centroids: Vec<[u8; N]>,
    /// 每个聚类中心包含的向量数量
    pub centroid_frequency: Vec<usize>,
}

/// K-modes 聚类算法，用于二进制向量
/// 返回聚类后的二进制向量，和每个聚类中心的向量数量
pub fn kmodes_binary<const N: usize>(data: &[[u8; N]], k: usize, max_iter: usize) -> KModeState<N> {
    if data.is_empty() || k == 0 {
        return KModeState::default();
    }

    let mut rng = rng();

    // 随机初始化聚类中心
    let mut centroids: Vec<[u8; N]> = data.choose_multiple(&mut rng, k).cloned().collect();

    let mut assignments;
    let mut distance = u32::MAX;
    let mut centroid_frequency = vec![0; k];

    for _ in 0..max_iter {
        // 分配每个数据点到最近的聚类中心
        let (new_assignments, new_distance) = update_assignments(data, &centroids);

        // 如果距离没有变小，则算法收敛
        if new_distance >= distance {
            break;
        }
        assignments = new_assignments;
        distance = new_distance;

        // 更新聚类中心
        let (new_centroids, new_centroid_frequency): (Vec<[u8; N]>, Vec<usize>) = (0..k)
            .into_par_iter()
            .map(|cluster_id| update_centroid(data, &assignments, cluster_id))
            .unzip();
        centroids = new_centroids;
        centroid_frequency = new_centroid_frequency;
    }

    KModeState { distsum: distance, centroids, centroid_frequency }
}

/// 将每个点分配给最近的聚类中心，并返回聚类中心的序号和总距离
fn update_assignments<const N: usize>(
    data: &[[u8; N]],
    centroids: &[[u8; N]],
) -> (Vec<usize>, u32) {
    let (assignments, distances): (Vec<_>, Vec<_>) = data
        .par_iter()
        .map(|point| {
            let mut min_distance = u32::MAX;
            let mut best_cluster = 0;

            for (j, centroid) in centroids.iter().enumerate() {
                let distance = hamming::<N>(point, centroid);
                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = j;
                }
            }

            (best_cluster, min_distance)
        })
        .unzip();
    let distance = distances.iter().sum();
    (assignments, distance)
}

/// 更新聚类中心：计算分配给该聚类的所有点的众数
fn update_centroid<const N: usize>(
    data: &[[u8; N]],
    assignments: &[usize],
    cluster_id: usize,
) -> ([u8; N], usize) {
    // 获取分配给该聚类点的向量列表
    let cluster_points: Vec<&[u8; N]> = data
        .iter()
        .zip(assignments.iter())
        .filter_map(|(point, &assignment)| (assignment == cluster_id).then_some(point))
        .collect();

    if cluster_points.is_empty() {
        return ([0u8; N], 0);
    }

    let mut new_centroid = [0u8; N];

    // 对每个字节位置计算众数
    for byte_pos in 0..N {
        let mut bit_counts = [0u32; 8]; // 每个bit位的计数

        // 统计每个bit位的1的数量
        for point in &cluster_points {
            let byte_val = point[byte_pos];
            for bit_pos in 0..8 {
                if (byte_val >> bit_pos) & 1 == 1 {
                    bit_counts[bit_pos] += 1;
                }
            }
        }

        // 根据众数设置新的字节值
        let mut new_byte = 0u8;
        let half_count = cluster_points.len() as u32 / 2;

        for bit_pos in 0..8 {
            if bit_counts[bit_pos] > half_count {
                new_byte |= 1 << bit_pos;
            }
        }

        new_centroid[byte_pos] = new_byte;
    }

    (new_centroid, cluster_points.len())
}

/// 计算不平衡因子
fn imbalance_factor(hist: &[usize]) -> f32 {
    let (mut tot, mut uf) = (0.0, 0.0);
    for h in hist {
        let h = *h as f32;
        tot += h;
        uf += h.powf(2.0);
    }
    uf * hist.len() as f32 / tot.powf(2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 辅助函数：将字节数组转换为二进制字符串（用于测试和调试）
    fn bytes_to_binary_string<const N: usize>(bytes: &[u8; N]) -> String {
        bytes.iter().map(|&byte| format!("{:08b}", byte)).collect::<String>()
    }

    /// 生成样本数据
    fn generate_clustered_data<const N: usize>(
        n: usize,
        num_clusters: usize,
    ) -> (Vec<[u8; N]>, Vec<[u8; N]>) {
        let mut rng = StdRng::seed_from_u64(42); // 使用固定种子确保结果可重现
        let mut data = vec![[0u8; N]; n];

        // 生成聚类中心模板
        let mut cluster_centers = vec![[0u8; N]; num_clusters];
        for center in &mut cluster_centers {
            rng.fill(&mut center[..]);
        }

        // 为每个向量分配到某个聚类，并在聚类中心附近生成数据
        for i in 0..n {
            let cluster_id = i % num_clusters;
            let base_center = &cluster_centers[cluster_id];

            // 在聚类中心附近生成数据（添加少量噪声）
            for j in 0..N {
                let noise_bits = rng.random::<u8>() & 0x0F; // 只改变低4位作为噪声
                data[i][j] = base_center[j] ^ noise_bits;
            }
        }

        (data, cluster_centers)
    }

    #[test]
    fn test_kmodes_simple() {
        // 创建一些测试数据 (4个字节，32位)
        let data: Vec<[u8; 4]> = vec![
            [0b11110000, 0b11110000, 0b00001111, 0b00001111], // 类型1
            [0b11111111, 0b11110000, 0b00001111, 0b00000000], // 类型1
            [0b00001111, 0b00001111, 0b11110000, 0b11110000], // 类型2
            [0b00000000, 0b00001111, 0b11110000, 0b11111111], // 类型2
        ];

        let ks = kmodes_binary(&data, 2, 100);

        assert_eq!(ks.centroids.len(), 2);

        // 打印结果用于验证
        for (i, centroid) in ks.centroids.iter().enumerate() {
            println!("Centroid {}: {}", i, bytes_to_binary_string(centroid));
        }
    }

    #[test]
    fn test_kmodes_complete() {
        let (data, cluster_centers) = generate_clustered_data(30720, 1024);
        let ks = kmodes_binary::<32>(&data, 1024, 100);
        assert_eq!(ks.centroids.len(), 1024);

        // for (i, centroid) in centroids.iter().enumerate() {
        //     println!("Centroid {}: {}", i, bytes_to_binary_string(centroid));
        // }
    }
}
