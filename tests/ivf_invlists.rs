use std::fs;

use imsearch::ivf::invlists::{ArrayInvertedLists, InvertedLists, OnDiskInvlists};
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
