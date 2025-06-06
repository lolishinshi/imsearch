use std::fs;

use imsearch::ivf::invlists::{ArrayInvertedLists, InvertedLists, OnDiskInvlists, merge_invlists};
use rstest::*;
use tempfile::TempDir;

const N: usize = 4;

#[fixture]
fn temp_dir() -> TempDir {
    TempDir::new().unwrap()
}

#[fixture]
fn sample_data() -> (Vec<u64>, Vec<[u8; N]>) {
    let ids = vec![1, 2, 3, 4, 5];
    let codes = vec![
        [0x01, 0x02, 0x03, 0x04],
        [0x11, 0x12, 0x13, 0x14],
        [0x21, 0x22, 0x23, 0x24],
        [0x31, 0x32, 0x33, 0x34],
        [0x41, 0x42, 0x43, 0x44],
    ];
    (ids, codes)
}

#[fixture]
fn populated_array_invlists(sample_data: (Vec<u64>, Vec<[u8; N]>)) -> ArrayInvertedLists<N> {
    let (ids, codes) = sample_data;
    let mut invlist = ArrayInvertedLists::new(3);

    // 分配数据到不同的列表
    invlist.add_entries(0, &ids[0..2], &codes[0..2]).unwrap();
    invlist.add_entries(1, &ids[2..4], &codes[2..4]).unwrap();
    invlist.add_entries(2, &ids[4..5], &codes[4..5]).unwrap();

    invlist
}

// ArrayInvertedLists 基础功能测试
#[rstest]
fn test_array_invlists_creation() {
    let invlist = ArrayInvertedLists::<N>::new(5);
    assert_eq!(invlist.nlist(), 5);
    for i in 0..5 {
        assert_eq!(invlist.list_len(i), 0);
    }
}

#[rstest]
fn test_array_invlists_add_entries(sample_data: (Vec<u64>, Vec<[u8; N]>)) {
    let (ids, codes) = sample_data;
    let mut invlist = ArrayInvertedLists::new(2);

    let count = invlist.add_entries(0, &ids[0..3], &codes[0..3]).unwrap();
    assert_eq!(count, 3);
    assert_eq!(invlist.list_len(0), 3);
    assert_eq!(invlist.list_len(1), 0);

    let count = invlist.add_entries(1, &ids[3..5], &codes[3..5]).unwrap();
    assert_eq!(count, 2);
    assert_eq!(invlist.list_len(1), 2);
}

#[rstest]
fn test_array_invlists_get_list(populated_array_invlists: ArrayInvertedLists<N>) {
    let invlist = populated_array_invlists;

    let (ids, codes) = invlist.get_list(0).unwrap();
    assert_eq!(ids.len(), 2);
    assert_eq!(codes.len(), 2);
    assert_eq!(ids[0], 1);
    assert_eq!(ids[1], 2);
    assert_eq!(codes[0], [0x01, 0x02, 0x03, 0x04]);
    assert_eq!(codes[1], [0x11, 0x12, 0x13, 0x14]);

    let (ids, codes) = invlist.get_list(2).unwrap();
    assert_eq!(ids.len(), 1);
    assert_eq!(codes.len(), 1);
    assert_eq!(ids[0], 5);
    assert_eq!(codes[0], [0x41, 0x42, 0x43, 0x44]);
}

#[rstest]
fn test_array_invlists_imbalance(populated_array_invlists: ArrayInvertedLists<N>) {
    let invlist = populated_array_invlists;
    let imbalance = invlist.imbalance();

    // 验证不平衡度计算：列表长度为 [2, 2, 1]
    // imbalance = (2² + 2² + 1²) * 3 / (2 + 2 + 1)² = 9 * 3 / 25 = 1.08
    assert!((imbalance - 1.08).abs() < 0.01);
}

#[rstest]
#[should_panic(expected = "ids and codes length mismatch")]
fn test_array_invlists_add_entries_mismatch() {
    let mut invlist = ArrayInvertedLists::<N>::new(2);
    let ids = vec![1, 2, 3];
    let codes = vec![[0x01; N], [0x02; N]]; // 长度不匹配
    invlist.add_entries(0, &ids, &codes).unwrap();
}

#[rstest]
fn test_array_invlists_save_load(
    populated_array_invlists: ArrayInvertedLists<N>,
    temp_dir: TempDir,
) {
    let invlist = populated_array_invlists;
    let file_path = temp_dir.path().join("test_invlists.bin");

    // 保存
    invlist.save(&file_path).unwrap();
    assert!(file_path.exists());

    // 验证文件内容非空
    let metadata = fs::metadata(&file_path).unwrap();
    assert!(metadata.len() > 0);
}

// OnDiskInvlists 测试
#[rstest]
fn test_ondisk_invlists_load_nonexistent(temp_dir: TempDir) {
    let file_path = temp_dir.path().join("nonexistent.bin");
    let result = OnDiskInvlists::<N>::load(&file_path);
    assert!(result.is_err());
}

#[rstest]
fn test_ondisk_invlists_from_array(
    populated_array_invlists: ArrayInvertedLists<N>,
    temp_dir: TempDir,
) {
    let array_invlist = populated_array_invlists;
    let file_path = temp_dir.path().join("test_disk.bin");

    // 保存 ArrayInvertedLists
    array_invlist.save(&file_path).unwrap();

    // 加载为 OnDiskInvlists
    let disk_invlist = OnDiskInvlists::<N>::load(&file_path).unwrap();

    // 验证基本信息一致
    assert_eq!(disk_invlist.nlist(), array_invlist.nlist());
    for i in 0..disk_invlist.nlist() {
        assert_eq!(disk_invlist.list_len(i), array_invlist.list_len(i));
    }

    // 验证数据一致性
    for i in 0..disk_invlist.nlist() {
        let (array_ids, array_codes) = array_invlist.get_list(i).unwrap();
        let (disk_ids, disk_codes) = disk_invlist.get_list(i).unwrap();

        assert_eq!(array_ids.as_ref(), disk_ids.as_ref());
        assert_eq!(array_codes.as_ref(), disk_codes.as_ref());
    }
}

#[rstest]
fn test_ondisk_invlists_add_entries_unsupported(
    populated_array_invlists: ArrayInvertedLists<N>,
    temp_dir: TempDir,
) {
    let array_invlist = populated_array_invlists;
    let file_path = temp_dir.path().join("test_disk.bin");

    array_invlist.save(&file_path).unwrap();
    let mut disk_invlist = OnDiskInvlists::<N>::load(&file_path).unwrap();

    let ids = vec![100];
    let codes = vec![[0xFF; N]];

    // OnDiskInvlists 不支持更新操作
    let result = disk_invlist.add_entries(0, &ids, &codes);
    assert!(result.is_err());
}

// merge_invlists 函数测试
#[rstest]
fn test_merge_invlists_single(populated_array_invlists: ArrayInvertedLists<N>, temp_dir: TempDir) {
    let invlist = populated_array_invlists;
    let file_path = temp_dir.path().join("merged_single.bin");

    let nlist = invlist.nlist();
    let invlists = &[invlist];
    merge_invlists(invlists, nlist, &file_path).unwrap();

    // 验证合并结果
    let merged = OnDiskInvlists::<N>::load(&file_path).unwrap();
    assert_eq!(merged.nlist(), nlist);

    for i in 0..merged.nlist() {
        let (orig_ids, orig_codes) = invlists[0].get_list(i).unwrap();
        let (merged_ids, merged_codes) = merged.get_list(i).unwrap();

        assert_eq!(orig_ids.as_ref(), merged_ids.as_ref());
        assert_eq!(orig_codes.as_ref(), merged_codes.as_ref());
    }
}

#[rstest]
fn test_merge_invlists_multiple(sample_data: (Vec<u64>, Vec<[u8; N]>), temp_dir: TempDir) {
    let (ids, codes) = sample_data;

    // 创建两个不同的倒排列表
    let mut invlist1 = ArrayInvertedLists::new(2);
    invlist1.add_entries(0, &ids[0..2], &codes[0..2]).unwrap();
    invlist1.add_entries(1, &ids[2..3], &codes[2..3]).unwrap();

    let mut invlist2 = ArrayInvertedLists::new(2);
    invlist2.add_entries(0, &ids[3..4], &codes[3..4]).unwrap();
    invlist2.add_entries(1, &ids[4..5], &codes[4..5]).unwrap();

    let file_path = temp_dir.path().join("merged_multiple.bin");
    merge_invlists(&[invlist1, invlist2], 2, &file_path).unwrap();

    // 验证合并结果
    let merged = OnDiskInvlists::<N>::load(&file_path).unwrap();
    assert_eq!(merged.nlist(), 2);
    assert_eq!(merged.list_len(0), 3); // 2 + 1
    assert_eq!(merged.list_len(1), 2); // 1 + 1

    // 验证第一个列表包含来自两个源的数据
    let (merged_ids, merged_codes) = merged.get_list(0).unwrap();
    assert_eq!(merged_ids.len(), 3);
    assert!(merged_ids.contains(&1));
    assert!(merged_ids.contains(&2));
    assert!(merged_ids.contains(&4));
}

#[rstest]
fn test_merge_invlists_empty_lists(temp_dir: TempDir) {
    let invlist1 = ArrayInvertedLists::<N>::new(3);
    let invlist2 = ArrayInvertedLists::<N>::new(3);

    let file_path = temp_dir.path().join("merged_empty.bin");
    merge_invlists(&[invlist1, invlist2], 3, &file_path).unwrap();

    let merged = OnDiskInvlists::<N>::load(&file_path).unwrap();
    assert_eq!(merged.nlist(), 3);
    for i in 0..3 {
        assert_eq!(merged.list_len(i), 0);
    }
}

// 边界条件和特殊情况测试
#[rstest]
fn test_empty_invlists() {
    let invlist = ArrayInvertedLists::<N>::new(1);
    assert_eq!(invlist.nlist(), 1);
    assert_eq!(invlist.list_len(0), 0);

    let (ids, codes) = invlist.get_list(0).unwrap();
    assert!(ids.is_empty());
    assert!(codes.is_empty());

    // 空列表的不平衡度应该为 NaN
    assert!(invlist.imbalance().is_nan());
}

#[rstest]
fn test_single_entry_operations(sample_data: (Vec<u64>, Vec<[u8; N]>)) {
    let (ids, codes) = sample_data;
    let mut invlist = ArrayInvertedLists::new(1);

    // 添加单个条目
    invlist.add_entries(0, &ids[0..1], &codes[0..1]).unwrap();
    assert_eq!(invlist.list_len(0), 1);

    let (retrieved_ids, retrieved_codes) = invlist.get_list(0).unwrap();
    assert_eq!(retrieved_ids[0], ids[0]);
    assert_eq!(retrieved_codes[0], codes[0]);
}

#[rstest]
fn test_large_nlist() {
    let invlist = ArrayInvertedLists::<N>::new(10000);
    assert_eq!(invlist.nlist(), 10000);

    // 验证所有列表都是空的
    for i in 0..10000 {
        assert_eq!(invlist.list_len(i), 0);
    }
}

#[rstest]
fn test_imbalance_calculation_edge_cases() {
    // 完全平衡的情况
    let mut invlist = ArrayInvertedLists::<N>::new(3);
    let ids = vec![1, 2, 3, 4, 5, 6];
    let codes = vec![[0x01; N]; 6];

    invlist.add_entries(0, &ids[0..2], &codes[0..2]).unwrap();
    invlist.add_entries(1, &ids[2..4], &codes[2..4]).unwrap();
    invlist.add_entries(2, &ids[4..6], &codes[4..6]).unwrap();

    let imbalance = invlist.imbalance();
    assert_eq!(imbalance, 1.0); // 完全平衡

    // 完全不平衡的情况
    let mut invlist2 = ArrayInvertedLists::<N>::new(2);
    invlist2.add_entries(0, &ids, &codes).unwrap();
    // 第二个列表保持空

    let imbalance2 = invlist2.imbalance();
    assert_eq!(imbalance2, 2.0); // 完全不平衡：一个列表包含所有项，另一个为空
}

#[rstest]
fn test_different_code_sizes() {
    // 测试不同的代码大小
    let invlist_1 = ArrayInvertedLists::<1>::new(2);
    assert_eq!(invlist_1.nlist(), 2);

    let invlist_8 = ArrayInvertedLists::<8>::new(2);
    assert_eq!(invlist_8.nlist(), 2);

    let invlist_32 = ArrayInvertedLists::<32>::new(2);
    assert_eq!(invlist_32.nlist(), 2);
}

// 性能相关测试
#[rstest]
fn test_large_batch_operations() {
    let mut invlist = ArrayInvertedLists::<N>::new(1);

    // 创建大量数据
    let large_ids: Vec<u64> = (0..10000).collect();
    let large_codes: Vec<[u8; N]> = (0..10000).map(|i| [(i % 256) as u8; N]).collect();

    let count = invlist.add_entries(0, &large_ids, &large_codes).unwrap();
    assert_eq!(count, 10000);
    assert_eq!(invlist.list_len(0), 10000);

    let (retrieved_ids, retrieved_codes) = invlist.get_list(0).unwrap();
    assert_eq!(retrieved_ids.len(), 10000);
    assert_eq!(retrieved_codes.len(), 10000);

    // 验证数据正确性
    for i in 0..10000 {
        assert_eq!(retrieved_ids[i], i as u64);
        assert_eq!(retrieved_codes[i], [(i % 256) as u8; N]);
    }
}

#[rstest]
fn test_cow_behavior(populated_array_invlists: ArrayInvertedLists<N>) {
    let invlist = populated_array_invlists;

    // 测试 Cow 的借用行为
    let (ids1, codes1) = invlist.get_list(0).unwrap();
    let (ids2, codes2) = invlist.get_list(0).unwrap();

    // 应该能够获取多次而不发生移动
    assert_eq!(ids1.as_ref(), ids2.as_ref());
    assert_eq!(codes1.as_ref(), codes2.as_ref());
}
