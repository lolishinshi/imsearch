use std::borrow::Cow;
use std::cell::RefCell;
use std::fs::File;
use std::io::Cursor;
use std::mem::size_of;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::slice;

use anyhow::Result;
use binrw::BinRead;
use memmap2::Mmap;
use zstd::bulk::Decompressor;
use zstd::zstd_safe::WriteBuf;

use crate::ivf::{InvertedLists, OnDiskIvfMetadata};

thread_local! {
    static READ_BUFFER: RefCell<Vec<u8>> = RefCell::new(vec![0u8; 1024]);
    static DECOMPRESSOR: RefCell<Decompressor<'static>> =
        RefCell::new(Decompressor::new().unwrap());
}

// 使用一个包装类型并实现 WriteBuf 来避免 Vec<u8> 的初始化开销
// 这里需要注意 T: Copy 是个过于宽泛的约束，但在实际使用中，T 只会是 u64 或 [u8; N]，因此不会有问题。
struct TypedWriteBuf<T: Copy>(Vec<T>);

unsafe impl<T: Copy> WriteBuf for TypedWriteBuf<T> {
    fn as_slice(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(self.0.as_ptr().cast::<u8>(), self.0.len() * size_of::<T>())
        }
    }

    fn capacity(&self) -> usize {
        self.0.capacity() * size_of::<T>()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.0.as_mut_ptr().cast()
    }

    unsafe fn filled_until(&mut self, n: usize) {
        unsafe { self.0.set_len(n / size_of::<T>().max(1)) };
    }
}

fn decompress_vec<T: Copy>(
    decompressor: &mut Decompressor<'_>,
    compressed: &[u8],
    len: usize,
) -> Result<Vec<T>> {
    let mut output = TypedWriteBuf(Vec::with_capacity(len));
    decompressor.decompress_to_buffer(compressed, &mut output)?;
    Ok(output.0)
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
            buf.resize(size, 0);

            // 考虑到大部分读取为一次性的随机读取，此处使用 pread 而不是 mmap，来避免大量缺页中断
            self.file.read_exact_at(&mut buf, offset as u64)?;

            let (ids, codes) = buf.split_at(split);

            // TODO: 是否需要延迟解压？
            DECOMPRESSOR.with(|decompressor| {
                let mut decompressor = decompressor.borrow_mut();
                let ids_buf = decompress_vec(&mut decompressor, ids, len)?;
                let codes_buf = decompress_vec::<[u8; N]>(&mut decompressor, codes, len)?;
                Ok((Cow::Owned(ids_buf), Cow::Owned(codes_buf)))
            })
        })
    }

    fn add_entry(&mut self, _list_no: usize, _id: u64, _code: &[u8; N]) -> Result<()> {
        unimplemented!("OnDiskInvlists 不支持更新操作")
    }
}
