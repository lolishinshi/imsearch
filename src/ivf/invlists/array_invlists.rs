use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::Result;
use bytemuck::cast_slice;
use byteorder::{NativeEndian, WriteBytesExt};

use super::InvertedLists;

pub struct ArrayInvertedLists<const N: usize> {
    pub nlist: usize,
    pub codes: Vec<Vec<[u8; N]>>,
    pub ids: Vec<Vec<u64>>,
}

impl<const N: usize> ArrayInvertedLists<N> {
    pub fn new(nlist: usize) -> Self {
        Self { nlist, codes: vec![vec![]; nlist], ids: vec![vec![]; nlist] }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        // 写入 nlist 和 code_size
        writer.write_u64::<NativeEndian>(self.nlist() as u64)?;
        writer.write_u64::<NativeEndian>(N as u64)?;
        // 写入每个倒排列表大小
        let list_len = self.ids.iter().map(|n| n.len()).collect::<Vec<_>>();
        writer.write_all(cast_slice(&list_len))?;
        // 写入每个倒排列表
        for i in 0..self.nlist() {
            let (ids, codes) = self.get_list(i)?;
            writer.write_all(cast_slice(&ids))?;
            writer.write_all(codes.as_flattened())?;
        }
        writer.flush()?;
        Ok(())
    }
}

impl<const N: usize> InvertedLists<N> for ArrayInvertedLists<N> {
    fn nlist(&self) -> usize {
        self.nlist
    }

    fn list_len(&self, list_no: usize) -> usize {
        self.ids[list_no].len()
    }

    fn get_list(&self, list_no: usize) -> Result<(&[u64], &[[u8; N]])> {
        Ok((&self.ids[list_no], &self.codes[list_no]))
    }

    fn add_entries(&mut self, list_no: usize, ids: &[u64], codes: &[[u8; N]]) -> Result<u64> {
        assert_eq!(ids.len(), codes.len(), "ids and codes length mismatch");
        self.ids[list_no].extend_from_slice(ids);
        self.codes[list_no].extend_from_slice(codes);
        Ok(ids.len() as u64)
    }
}
