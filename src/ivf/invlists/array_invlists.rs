use std::borrow::Cow;

use anyhow::Result;

use super::InvertedLists;

/// 完全存储在内存中的倒排列表
pub struct ArrayInvertedLists<const N: usize> {
    pub nlist: usize,
    pub codes: Vec<Vec<[u8; N]>>,
    pub ids: Vec<Vec<u64>>,
}

impl<const N: usize> ArrayInvertedLists<N> {
    pub fn new(nlist: usize) -> Self {
        Self { nlist, codes: vec![vec![]; nlist], ids: vec![vec![]; nlist] }
    }
}

impl<const N: usize> InvertedLists<N> for ArrayInvertedLists<N> {
    fn nlist(&self) -> usize {
        self.nlist
    }

    fn list_len(&self, list_no: usize) -> usize {
        self.ids[list_no].len()
    }

    fn get_list(&self, list_no: usize) -> Result<(Cow<'_, [u64]>, Cow<'_, [[u8; N]]>)> {
        Ok((Cow::Borrowed(&self.ids[list_no]), Cow::Borrowed(&self.codes[list_no])))
    }

    fn add_entry(&mut self, list_no: usize, id: u64, code: &[u8; N]) -> Result<()> {
        self.ids[list_no].push(id);
        self.codes[list_no].push(*code);
        Ok(())
    }
}
