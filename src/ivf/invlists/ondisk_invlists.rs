use std::fs::File;
use std::path::Path;

use anyhow::Result;
use bytemuck::cast_slice;
use memmap2::{Advice, MmapMut};

use crate::ivf::InvertedLists;
use crate::ivf::invlists::read_metadata;

pub struct OnDiskInvlists<const N: usize> {
    /// 倒排列表数量
    nlist: usize,
    /// 内存映射文件
    mmap: MmapMut,
    /// 倒排列表大小
    list_len: Vec<usize>,
    /// 倒排列表大小累加和，用于辅助偏移计算
    list_len_acc: Vec<usize>,
}

impl<const N: usize> OnDiskInvlists<N> {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::options().read(true).write(true).open(path)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        // TODO: 是否有必要使用 MAP_POPULATE ？
        mmap.advise(Advice::Random)?;
        let (nlist, code_size, list_len) = read_metadata(&mmap)?;
        assert_eq!(code_size, N, "code_size mismatch");
        let list_len_acc = list_len
            .iter()
            .scan(0, |acc, x| {
                let prev = *acc;
                *acc += x;
                Some(prev)
            })
            .collect();
        Ok(Self { nlist, mmap, list_len, list_len_acc })
    }

    /// 获取指定倒排列表的偏移量，其中前 len * u64 是 ids，后面 len * N * u8 是 codes
    fn list_offset(&self, list_no: usize) -> usize {
        size_of::<u64>() * 2
            + size_of::<u64>() * self.nlist
            + (size_of::<u64>() + N * size_of::<u8>()) * self.list_len_acc[list_no]
    }
}

impl<const N: usize> InvertedLists<N> for OnDiskInvlists<N> {
    fn nlist(&self) -> usize {
        self.nlist
    }

    fn list_len(&self, list_no: usize) -> usize {
        self.list_len[list_no]
    }

    fn get_list(&self, list_no: usize) -> Result<(&[u64], &[[u8; N]])> {
        let len = self.list_len(list_no);
        let offset = self.list_offset(list_no);
        let data = &self.mmap[offset..][..len * (size_of::<u64>() + N * size_of::<u8>())];
        let (ids, codes) = data.split_at(len * size_of::<u64>());
        let ids = cast_slice(ids);
        let (codes, _) = codes.as_chunks();
        Ok((ids, codes))
    }

    fn add_entries(&mut self, _list_no: usize, _ids: &[u64], _codes: &[[u8; N]]) -> Result<u64> {
        unimplemented!("OnDiskInvlists 不支持更新操作")
    }
}
