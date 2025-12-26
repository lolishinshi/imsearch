use std::borrow::Cow;
use std::cell::RefCell;
use std::fs::File;
use std::io::Cursor;
use std::os::unix::fs::FileExt;
use std::path::Path;

use anyhow::Result;
use binrw::BinRead;
use bytemuck::cast_slice_mut;
use memmap2::Mmap;
use zstd::bulk::decompress_to_buffer;

use crate::ivf::{InvertedLists, OnDiskIvfMetadata};

thread_local! {
    static READ_BUFFER: RefCell<Vec<u8>> = RefCell::new(vec![0u8; 1024]);
}

/// 磁盘倒排列表
pub struct OnDiskInvlists<const N: usize> {
    /// 元数据
    metadata: OnDiskIvfMetadata,
    /// 文件句柄
    file: File,
}

impl<const N: usize> OnDiskInvlists<N> {
    /// 加载磁盘倒排列表
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::options().read(true).write(true).open(path)?;

        // 临时使用 mmap 读取元数据
        let mmap = unsafe { Mmap::map(&file)? };
        let metadata = OnDiskIvfMetadata::read(&mut Cursor::new(&mmap))?;

        assert_eq!(metadata.code_size, N as u64, "code_size mismatch");
        Ok(Self { metadata, file })
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

        // 使用线程局部缓冲区来避免频繁的内存分配
        READ_BUFFER.with(|buf| {
            let mut buf = buf.borrow_mut();
            unsafe { reserve_and_set_len(&mut buf, size) };

            // 考虑到大部分读取为一次性的随机读取，此处使用 pread 而不是 mmap，来避免大量缺页中断
            self.file.read_exact_at(&mut buf, offset as u64)?;

            let (ids, codes) = buf.split_at(split);

            // 为了避免复杂的类型转换，这里先建立目标类型的缓冲区，然后作为 &mut [u8] 传入
            let mut ids_buf: Vec<u64> = Vec::with_capacity(len);
            let mut codes_buf: Vec<[u8; N]> = Vec::with_capacity(len);
            unsafe { ids_buf.set_len(len) };
            unsafe { codes_buf.set_len(len) };

            // TODO: 是否需要延迟解压？
            decompress_to_buffer(ids, cast_slice_mut(&mut ids_buf))?;
            decompress_to_buffer(codes, codes_buf.as_flattened_mut())?;
            Ok((Cow::Owned(ids_buf), Cow::Owned(codes_buf)))
        })
    }

    fn add_entry(&mut self, _list_no: usize, _id: u64, _code: &[u8; N]) -> Result<()> {
        unimplemented!("OnDiskInvlists 不支持更新操作")
    }
}

unsafe fn reserve_and_set_len<T>(vec: &mut Vec<T>, size: usize) {
    vec.clear();
    vec.reserve(size);
    unsafe { vec.set_len(size) };
}
