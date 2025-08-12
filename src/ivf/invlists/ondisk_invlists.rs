use std::borrow::Cow;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;

use anyhow::Result;
use binrw::BinRead;
use bytemuck::cast_slice_mut;
use memmap2::{Advice, MmapMut};
use zstd::bulk::decompress_to_buffer;

use crate::ivf::{InvertedLists, OnDiskIvfMetadata};

/// 磁盘倒排列表
pub struct OnDiskInvlists<const N: usize> {
    /// 元数据
    metadata: OnDiskIvfMetadata,
    /// 内存映射文件
    mmap: MmapMut,
}

impl<const N: usize> OnDiskInvlists<N> {
    /// 加载磁盘倒排列表
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::options().read(true).write(true).open(path)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        // TODO: 是否有必要使用 MAP_POPULATE ？
        mmap.advise(Advice::Random)?;
        let metadata = OnDiskIvfMetadata::read(&mut Cursor::new(&mmap))?;
        assert_eq!(metadata.code_size, N as u64, "code_size mismatch");
        Ok(Self { metadata, mmap })
    }

    // 加载一个倒排列表的长度，偏移量、大小和分割点
    fn list_info(&self, list_no: usize) -> (usize, usize, usize, usize) {
        let len = self.metadata.list_len[list_no] as usize;
        let offset = self.metadata.list_offset[list_no] as usize;
        let size = self.metadata.list_size[list_no] as usize;
        let split = self.metadata.list_split[list_no] as usize;
        (len, offset, size, split)
    }
}

impl<const N: usize> InvertedLists<N> for OnDiskInvlists<N> {
    #[inline(always)]
    fn nlist(&self) -> usize {
        self.metadata.nlist as usize
    }

    #[inline(always)]
    fn list_len(&self, list_no: usize) -> usize {
        self.metadata.list_len[list_no] as usize
    }

    #[inline(always)]
    fn get_list(&self, list_no: usize) -> Result<(Cow<'_, [u64]>, Cow<'_, [[u8; N]]>)> {
        let (len, offset, size, split) = self.list_info(list_no);
        let (ids, codes) = self.mmap[offset..][..size].split_at(split);
        // 为了避免复杂的类型转换，这里先建立目标类型的缓冲区，然后作为 &mut [u8] 传入
        // TODO: 需要使用 MaybeUninit 来避免初始化开销吗？
        let mut ids_buf = vec![0u64; len];
        let mut codes_buf = vec![[0u8; N]; len];
        // TODO: 是否需要延迟解压？
        decompress_to_buffer(ids, cast_slice_mut(&mut ids_buf))?;
        decompress_to_buffer(codes, codes_buf.as_flattened_mut())?;
        Ok((Cow::Owned(ids_buf), Cow::Owned(codes_buf)))
    }

    fn add_entry(&mut self, _list_no: usize, _id: u64, _code: &[u8; N]) -> Result<()> {
        unimplemented!("OnDiskInvlists 不支持更新操作")
    }
}
