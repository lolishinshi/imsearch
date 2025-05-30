use bytemuck::cast_slice;

#[inline(always)]
pub fn hamming<const N: usize>(va: &[u8], vb: &[u8]) -> u32 {
    match N {
        256 => hamming_256(va, vb),
        _ => hamming_naive::<N>(va, vb),
    }
}

#[inline(always)]
pub fn hamming_naive<const N: usize>(va: &[u8], vb: &[u8]) -> u32 {
    let mut sum = 0;
    for i in 0..N / 8 {
        sum += (va[i] ^ vb[i]).count_ones();
    }
    sum
}

#[inline(always)]
pub fn hamming_256(va: &[u8], vb: &[u8]) -> u32 {
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

// TODO: 使用 ndarray

/// 计算向量 va 和 vb 的汉明距离，并返回距离最小的 k 个索引和距离
///
/// 参数：
/// - va: N 位的向量 va
/// - vb: 若干组 N 位的向量 vb
/// - k: 返回的最近邻居数量
pub fn knn_hamming<const N: usize>(va: &[u8], vb: &[u8], k: usize) -> (Vec<usize>, Vec<u32>) {
    assert!(k <= 8, "k must be less than 8");
    let mut dis = [u32::MAX; 8];
    let mut idx = [0; 8];
    for (i, chunk) in vb.chunks_exact(N / 8).enumerate() {
        let d = hamming::<N>(va, chunk);
        if d > dis[0] {
            continue;
        }
        // 此处维护一个长度为 K 的单调递减数组
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
    idx.into_iter().zip(dis.into_iter()).filter(|(_, d)| *d != u32::MAX).rev().take(k).unzip()
}

/// 批量计算 va 和 vb 的汉明距离，返回每个向量的 k 个最近邻居
pub fn batch_knn_hamming<const N: usize>(
    va: &[u8],
    vb: &[u8],
    k: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<u32>>) {
    // TODO: 这里可以使用 Vec<[usize; N]> 或者 tinyvec 吗？
    let code_size = N / 8;
    let mut ids = Vec::with_capacity(va.len() / code_size);
    let mut dis = Vec::with_capacity(va.len() / code_size);
    for chunk in va.chunks_exact(code_size) {
        let (ids1, dis1) = knn_hamming::<N>(chunk, vb, k);
        ids.push(ids1);
        dis.push(dis1);
    }
    (ids, dis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_identical() {
        let va = [0u8; 32];
        let vb = [0u8; 32];
        assert_eq!(hamming::<256>(&va, &vb), 0);
    }

    #[test]
    fn test_hamming_all_different() {
        let va = [0u8; 32];
        let vb = [255u8; 32];
        assert_eq!(hamming::<256>(&va, &vb), 256);
    }

    #[test]
    fn test_hamming_single_bit() {
        let va = [0u8; 1];
        let mut vb = [0u8; 1];
        vb[0] = 1; // 设置一个位为1
        assert_eq!(hamming::<8>(&va, &vb), 1);
    }

    #[test]
    fn test_knn_hamming_single_vector() {
        let va = [0u8; 32];
        let vb = [255u8; 32];
        let (ids, dis) = knn_hamming::<256>(&va, &vb, 1);
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], 0); // 索引为0
        assert_eq!(dis[0], 256); // 距离为256
    }

    #[test]
    fn test_knn_hamming_multiple_vectors() {
        let va = [0u8; 32];
        // 创建3个向量：距离分别为0, 2, 1
        let mut vb = vec![0u8; 96];
        vb[32] = 3;
        vb[64] = 1;

        let (ids, dis) = knn_hamming::<256>(&va, &vb, 3);
        assert_eq!(ids.len(), 3);

        // 结果应该按距离排序
        assert_eq!(ids, &[0, 2, 1]);
        assert_eq!(dis, &[0, 1, 2]);
    }

    #[test]
    fn test_knn_hamming_k_limit() {
        let va = [0u8; 32];
        let vb = [255u8; 64]; // 2个向量
        let (ids, _) = knn_hamming::<256>(&va, &vb, 5); // 请求5个，但只有2个向量
        assert_eq!(ids.len(), 2);
        assert_eq!(ids, &[0, 1]);
    }

    #[test]
    #[should_panic(expected = "k must be less than 8")]
    fn test_knn_hamming_k_too_large() {
        let va = [0u8; 32];
        let vb = [0u8; 32];
        knn_hamming::<256>(&va, &vb, 11); // 应该panic
    }
}
