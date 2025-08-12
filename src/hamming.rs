use std::cmp::Reverse;
use std::collections::BinaryHeap;

use bytemuck::cast_slice;

#[inline(always)]
pub fn hamming<const N: usize>(va: &[u8], vb: &[u8]) -> u32 {
    match N {
        32 => hamming_32(va, vb),
        _ => hamming_naive::<N>(va, vb),
    }
}

#[inline(always)]
pub fn hamming_naive<const N: usize>(va: &[u8], vb: &[u8]) -> u32 {
    let mut sum = 0;
    for i in 0..N {
        sum += (va[i] ^ vb[i]).count_ones();
    }
    sum
}

#[inline(always)]
pub fn hamming_32(va: &[u8], vb: &[u8]) -> u32 {
    let va: &[u64] = cast_slice(va);
    let vb: &[u64] = cast_slice(vb);
    // 测试表明，此处使用 unsafe 转换并不会更快
    //let va: &[u64] = unsafe { std::slice::from_raw_parts(va.as_ptr() as *const u64, 4) };
    //let vb: &[u64] = unsafe { std::slice::from_raw_parts(vb.as_ptr() as *const u64, 4) };
    (va[0] ^ vb[0]).count_ones()
        + (va[1] ^ vb[1]).count_ones()
        + (va[2] ^ vb[2]).count_ones()
        + (va[3] ^ vb[3]).count_ones()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct KNNResult {
    /// 注意此处 dis 排在前面，保证自动 derive 的 Ord 正确
    pub dis: u32,
    pub idx: usize,
}

pub fn knn_hamming<const N: usize>(va: &[u8; N], vb: &[[u8; N]], k: usize) -> Vec<(usize, u32)> {
    return knn_hamming_heap::<N>(va, vb, k);
}

/// 计算向量 va 和 vb 的汉明距离，并返回距离最小的 k 个索引和距离
pub fn knn_hamming_heap<const N: usize>(
    va: &[u8; N],
    vb: &[[u8; N]],
    k: usize,
) -> Vec<(usize, u32)> {
    let mut heap = BinaryHeap::new();
    for (i, chunk) in vb.iter().enumerate() {
        let d = hamming::<N>(va, chunk);
        if heap.len() < k {
            heap.push(Reverse(KNNResult { idx: i, dis: d }));
        } else {
            let Reverse(peek) = heap.peek().unwrap();
            if d < peek.dis {
                heap.pop();
                heap.push(Reverse(KNNResult { idx: i, dis: d }));
            }
        }
    }
    heap.into_iter().map(|Reverse(a)| (a.idx, a.dis)).collect()
}

pub fn knn_hamming_array<const N: usize>(
    va: &[u8; N],
    vb: &[[u8; N]],
    k: usize,
) -> Vec<(usize, u32)> {
    // 考虑到 k 通常很小，为了最大化性能，此处开辟一个栈上的固定数组来存储 KNN 结果
    assert!(k <= 8, "k must be less than 8");
    let mut dis = [u32::MAX; 8];
    let mut idx = [0; 8];
    for (i, chunk) in vb.iter().enumerate() {
        let d = hamming::<N>(va, chunk);
        if d >= dis[0] {
            continue;
        }
        // 维护一个长度为 K 的单调递减数组
        // 寻找插入点时，从后往前遍历
        // 插入时，将前面的元素向左移动，保证最大的元素在前面
        for j in (0..k).rev() {
            if d < dis[j] {
                dis[..=j].rotate_left(1);
                dis[j] = d;
                idx[..=j].rotate_left(1);
                idx[j] = i;
                break;
            }
        }
    }
    // 由于需要过滤掉未初始化元素，加上 reverse
    // 此处的拷贝不可避免，因此直接返回 tuple，省的后面高级 API 再拷贝一次
    idx.into_iter().zip(dis).filter(|(_, d)| *d != u32::MAX).rev().take(k).collect()
}

/// 批量计算 va 和 vb 的汉明距离，返回每个向量的 k 个最近邻居
pub fn batch_knn_hamming<const N: usize>(
    va: &[[u8; N]],
    vb: &[[u8; N]],
    k: usize,
) -> Vec<Vec<(usize, u32)>> {
    let mut r = Vec::with_capacity(va.len());
    for chunk in va.iter() {
        let t = knn_hamming::<N>(chunk, vb, k);
        r.push(t);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_identical() {
        let va = [0u8; 32];
        let vb = [0u8; 32];
        assert_eq!(hamming::<32>(&va, &vb), 0);
    }

    #[test]
    fn test_hamming_all_different() {
        let va = [0u8; 32];
        let vb = [255u8; 32];
        assert_eq!(hamming::<32>(&va, &vb), 256);
    }

    #[test]
    fn test_hamming_single_bit() {
        let va = [0u8; 1];
        let mut vb = [0u8; 1];
        vb[0] = 1; // 设置一个位为1
        assert_eq!(hamming::<1>(&va, &vb), 1);
    }

    #[test]
    fn test_knn_hamming_single_vector() {
        let va = [0u8; 32];
        let vb = [255u8; 32];
        let r = knn_hamming::<32>(&va, &[vb], 1);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 0); // 索引为0
        assert_eq!(r[0].1, 256); // 距离为256
    }

    #[test]
    fn test_knn_hamming_multiple_vectors() {
        let va = [0u8; 32];
        // 创建3个向量：距离分别为0, 2, 1
        let mut vb = [[0u8; 32]; 3];
        vb[1][0] = 3;
        vb[2][0] = 1;

        let r = knn_hamming::<32>(&va, &vb, 3);
        assert_eq!(r.len(), 3);

        // 结果应该按距离排序
        assert_eq!(r, &[(0, 0), (2, 1), (1, 2)]);
    }

    #[test]
    fn test_knn_hamming_k_limit() {
        let va = [0u8; 32];
        let vb = [[255u8; 32]; 2]; // 2个向量
        let r = knn_hamming::<32>(&va, &vb, 5); // 请求5个，但只有2个向量
        assert_eq!(r.len(), 2);
        assert_eq!(r, &[(0, 256), (1, 256)]);
    }

    #[test]
    #[should_panic(expected = "k must be less than 8")]
    fn test_knn_hamming_k_too_large() {
        let va = [0u8; 32];
        let vb = [0u8; 32];
        knn_hamming::<32>(&va, &[vb], 11); // 应该panic
    }
}
