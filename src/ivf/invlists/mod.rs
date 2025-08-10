mod array_invlists;
mod ondisk_invlists;
mod vstack_invlists;

use std::borrow::Cow;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::Result;
pub use array_invlists::*;
use binrw::{BinWrite, binrw};
use bytemuck::cast_slice;
use itertools::izip;
pub use ondisk_invlists::*;
use rayon::prelude::*;
pub use vstack_invlists::*;
use zstd::bulk::compress;

use crate::kmodes::imbalance_factor;

pub trait InvertedLists<const N: usize> {
    /// 返回倒排表的列表数量
    fn nlist(&self) -> usize;

    /// 返回指定倒排表的元素数量
    fn list_len(&self, list_no: usize) -> usize;

    /// 返回指定倒排表中向量的 ID 列表和数据
    fn get_list(&self, list_no: usize) -> Result<(Cow<'_, [u64]>, Cow<'_, [[u8; N]]>)>;

    /// 往指定倒排表中添加一个元素
    fn add_entry(&mut self, list_no: usize, id: u64, code: &[u8; N]) -> Result<()>;

    /// 往指定倒排表中批量添加元素
    fn add_entries(&mut self, list_no: usize, ids: &[u64], codes: &[[u8; N]]) -> Result<()> {
        for (id, code) in izip!(ids, codes) {
            self.add_entry(list_no, *id, &code)?;
        }
        Ok(())
    }

    /// 计算不平衡度
    fn imbalance(&self) -> f32 {
        let mut hist = Vec::with_capacity(self.nlist());
        for i in 0..self.nlist() {
            hist.push(self.list_len(i));
        }
        imbalance_factor(&hist)
    }
}

/// 保存到文件
pub fn save_invlists<const N: usize, P, T>(invlists: &T, path: P) -> Result<()>
where
    P: AsRef<Path>,
    T: InvertedLists<N> + Sync,
{
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // 提前写入 metadata 占位，后续再来覆盖
    let mut metadata = OnDiskIvfMetadata::new(invlists.nlist(), N);
    metadata.write(&mut writer)?;

    // 注意此处 offset 为刨去 metadata 后的偏移量
    let mut offset = writer.stream_position()?;

    // TODO: 增加写入进度条
    rayon::scope(|s| {
        let (tx, rx) = crossbeam_channel::bounded(num_cpus::get());
        s.spawn(move |_| {
            (0..invlists.nlist()).into_par_iter().for_each(|i| {
                let (ids, codes) = invlists.get_list(i).unwrap();
                let list_len = ids.len();
                let ids = compress(cast_slice(&ids), 0).unwrap();
                let codes = compress(codes.as_flattened(), 0).unwrap();
                tx.send((i, list_len, ids, codes)).unwrap();
            });
        });
        while let Ok((i, list_len, ids, codes)) = rx.recv() {
            write_one_list(&mut writer, &mut metadata, i, list_len, &ids, &codes, &mut offset)
                .unwrap();
        }
    });

    writer.seek(SeekFrom::Start(0))?;
    metadata.write(&mut writer)?;
    Ok(())
}

#[binrw]
#[brw(little)]
pub struct OnDiskIvfMetadata {
    /// 倒排列表数量
    pub nlist: u64,
    /// 向量字节数
    pub code_size: u64,
    /// 每个倒排列表的元素数量
    #[br(count = nlist)]
    pub list_len: Vec<u64>,
    /// 倒排列表在整个文件中的偏移量
    #[br(count = nlist)]
    pub list_offset: Vec<u64>,
    /// 倒排列表的总大小
    #[br(count = nlist)]
    pub list_size: Vec<u64>,
    /// 单个倒排列表中 id 和 code 部分的分割点
    #[br(count = nlist)]
    pub list_split: Vec<u64>,
}

impl OnDiskIvfMetadata {
    pub fn new(nlist: usize, code_size: usize) -> Self {
        Self {
            nlist: nlist as u64,
            code_size: code_size as u64,
            list_len: vec![0; nlist],
            list_offset: vec![0; nlist],
            list_size: vec![0; nlist],
            list_split: vec![0; nlist],
        }
    }
}

fn write_one_list<W: Write>(
    writer: &mut W,
    metadata: &mut OnDiskIvfMetadata,
    list_no: usize,
    list_len: usize,
    ids: &[u8],
    codes: &[u8],
    offset: &mut u64,
) -> Result<()> {
    let size = (ids.len() + codes.len()) as u64;

    metadata.list_len[list_no] = list_len as u64;
    metadata.list_offset[list_no] = *offset;
    metadata.list_size[list_no] = size;
    metadata.list_split[list_no] = ids.len() as u64;

    *offset += size;

    writer.write_all(&ids)?;
    writer.write_all(&codes)?;
    Ok(())
}
