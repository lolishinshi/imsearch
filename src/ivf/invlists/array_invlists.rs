use std::borrow::Cow;
use std::mem;

use anyhow::Result;

use super::{InvertedLists, InvertedListsReader, InvertedListsWriter};

pub struct ArrayInvertedLists<const N: usize> {
    pub nlist: u32,
    pub codes: Vec<Vec<[u8; N]>>,
    pub ids: Vec<Vec<u64>>,
}

pub struct ArrayInvertedListsReader<'a, const N: usize>(&'a ArrayInvertedLists<N>);

pub struct ArrayInvertedListsWriter<'a, const N: usize>(&'a mut ArrayInvertedLists<N>);

impl<const N: usize> ArrayInvertedLists<N> {
    pub fn new(nlist: u32) -> Self {
        Self { nlist, codes: vec![vec![]; nlist as usize], ids: vec![vec![]; nlist as usize] }
    }
}

impl<const N: usize> InvertedLists<N> for ArrayInvertedLists<N> {
    type Reader<'a>
        = ArrayInvertedListsReader<'a, N>
    where
        Self: 'a;
    type Writer<'a>
        = ArrayInvertedListsWriter<'a, N>
    where
        Self: 'a;

    fn reader(&self) -> Result<Self::Reader<'_>> {
        Ok(ArrayInvertedListsReader(self))
    }

    fn writer(&mut self) -> Result<Self::Writer<'_>> {
        Ok(ArrayInvertedListsWriter(self))
    }
}

impl<const N: usize> InvertedListsReader<N> for ArrayInvertedListsReader<'_, N> {
    fn nlist(&self) -> u32 {
        self.0.nlist
    }

    fn list_len(&self, list_no: u32) -> usize {
        self.0.ids[list_no as usize].len()
    }

    fn get_list(&self, list_no: u32) -> Result<(Cow<[u64]>, Cow<[[u8; N]]>)> {
        let list_no = list_no as usize;
        Ok((Cow::Borrowed(&self.0.ids[list_no]), Cow::Borrowed(&self.0.codes[list_no])))
    }
}

impl<const N: usize> InvertedListsReader<N> for ArrayInvertedListsWriter<'_, N> {
    fn nlist(&self) -> u32 {
        ArrayInvertedListsReader(self.0).nlist()
    }

    fn list_len(&self, list_no: u32) -> usize {
        ArrayInvertedListsReader(self.0).list_len(list_no)
    }

    fn get_list(&self, list_no: u32) -> Result<(Cow<[u64]>, Cow<[[u8; N]]>)> {
        // 这个写法无法通过生命周期检查
        // ArrayInvertedListsReader(self.0).get_list(list_no)
        let list_no = list_no as usize;
        Ok((Cow::Borrowed(&self.0.ids[list_no]), Cow::Borrowed(&self.0.codes[list_no])))
    }
}

impl<const N: usize> InvertedListsWriter<N> for ArrayInvertedListsWriter<'_, N> {
    fn add_entries(&mut self, list_no: u32, ids: &[u64], codes: &[[u8; N]]) -> Result<u64> {
        assert_eq!(ids.len(), codes.len(), "ids and codes length mismatch");
        let list_no = list_no as usize;
        self.0.ids[list_no].extend_from_slice(ids);
        self.0.codes[list_no].extend_from_slice(codes);
        Ok(ids.len() as u64)
    }

    fn clear(&mut self, list_no: u32) -> Result<()> {
        let list_no = list_no as usize;
        // 代替 clear + shrink_to_fit
        // clear 不会释放内存，shrink_to_fit 可能会发生拷贝
        let _ = mem::replace(&mut self.0.ids[list_no], vec![]);
        let _ = mem::replace(&mut self.0.codes[list_no], vec![]);
        Ok(())
    }
}
