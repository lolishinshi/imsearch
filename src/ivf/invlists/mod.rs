mod array_invlists;
mod ondisk_invlists;

use std::fs::File;
use std::io::{BufWriter, Cursor, Read, Write};
use std::path::Path;

use anyhow::Result;
pub use array_invlists::*;
use bytemuck::{cast_slice, cast_slice_mut};
use byteorder::{NativeEndian, ReadBytesExt, WriteBytesExt};
pub use ondisk_invlists::*;

use crate::kmodes::imbalance_factor;

pub trait InvertedLists<const N: usize> {
    /// 返回倒排表的列表数量
    fn nlist(&self) -> usize;

    /// 返回指定倒排表的元素数量
    fn list_len(&self, list_no: usize) -> usize;

    /// 返回指定倒排表中向量的 ID 列表和数据
    fn get_list(&self, list_no: usize) -> Result<(&[u64], &[[u8; N]])>;

    /// 往指定倒排表中添加元素
    fn add_entries(&mut self, list_no: usize, ids: &[u64], codes: &[[u8; N]]) -> Result<u64>;

    /// 计算不平衡度
    fn imbalance(&self) -> f32 {
        let mut hist = Vec::with_capacity(self.nlist());
        for i in 0..self.nlist() {
            hist.push(self.list_len(i));
        }
        imbalance_factor(&hist)
    }
}

/// 合并多个倒排列表，并保存到指定文件
pub fn merge_invlists<const N: usize, T, P>(invlists: &[T], nlist: usize, path: P) -> Result<()>
where
    T: InvertedLists<N>,
    P: AsRef<Path>,
{
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let mut list_len = vec![0; nlist];
    for i in 0..nlist {
        for invlist in invlists {
            list_len[i] += invlist.list_len(i);
        }
    }
    writer.write_u64::<NativeEndian>(nlist as u64)?;
    writer.write_u64::<NativeEndian>(N as u64)?;
    writer.write_all(cast_slice(&list_len))?;

    for i in 0..nlist {
        let mut ids = Vec::new();
        let mut codes = Vec::new();
        for invlist in invlists {
            let (other_ids, other_codes) = invlist.get_list(i)?;
            ids.extend_from_slice(&other_ids);
            codes.extend_from_slice(&other_codes);
        }
        writer.write_all(cast_slice(&ids))?;
        writer.write_all(codes.as_flattened())?;
    }

    Ok(())
}

/// 从文件头读取 nlist, code_size, list_len
fn read_metadata(buf: &[u8]) -> Result<(usize, usize, Vec<usize>)> {
    let mut cursor = Cursor::new(buf);
    let nlist = cursor.read_u64::<NativeEndian>()?;
    let code_size = cursor.read_u64::<NativeEndian>()?;
    let mut list_len = vec![0usize; nlist as usize];
    cursor.read_exact(cast_slice_mut(&mut list_len))?;
    Ok((nlist as usize, code_size as usize, list_len))
}
