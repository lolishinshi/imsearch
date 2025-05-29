use std::borrow::Cow;

use anyhow::Result;

use super::{InvertedLists, InvertedListsReader, InvertedListsWriter};

pub struct ArrayInvertedLists {
    pub nlist: u32,
    pub code_size: u32,
    pub codes: Vec<Vec<u8>>,
    pub ids: Vec<Vec<u64>>,
}

pub struct ArrayInvertedListsReader<'a>(&'a ArrayInvertedLists);

pub struct ArrayInvertedListsWriter<'a>(&'a mut ArrayInvertedLists);

impl ArrayInvertedLists {
    pub fn new(nlist: u32, code_size: u32) -> Self {
        Self {
            nlist,
            code_size,
            codes: vec![vec![]; nlist as usize],
            ids: vec![vec![]; nlist as usize],
        }
    }
}

impl InvertedLists for ArrayInvertedLists {
    type Reader<'a>
        = ArrayInvertedListsReader<'a>
    where
        Self: 'a;
    type Writer<'a>
        = ArrayInvertedListsWriter<'a>
    where
        Self: 'a;

    fn reader(&self) -> Result<Self::Reader<'_>> {
        Ok(ArrayInvertedListsReader(self))
    }

    fn writer(&mut self) -> Result<Self::Writer<'_>> {
        Ok(ArrayInvertedListsWriter(self))
    }
}

impl InvertedListsReader for ArrayInvertedListsReader<'_> {
    fn nlist(&self) -> u32 {
        self.0.nlist
    }

    fn code_size(&self) -> u32 {
        self.0.code_size
    }

    fn list_len(&self, list_no: u32) -> usize {
        self.0.ids[list_no as usize].len()
    }

    fn get_list(&self, list_no: u32) -> (Cow<[u64]>, Cow<[u8]>) {
        let list_no = list_no as usize;
        (Cow::Borrowed(&self.0.ids[list_no]), Cow::Borrowed(&self.0.codes[list_no]))
    }
}

impl InvertedListsReader for ArrayInvertedListsWriter<'_> {
    fn nlist(&self) -> u32 {
        self.0.nlist
    }

    fn code_size(&self) -> u32 {
        self.0.code_size
    }

    fn list_len(&self, list_no: u32) -> usize {
        self.0.codes[list_no as usize].len() / self.0.code_size as usize
    }

    fn get_list(&self, list_no: u32) -> (Cow<[u64]>, Cow<[u8]>) {
        let list_no = list_no as usize;
        (Cow::Borrowed(&self.0.ids[list_no]), Cow::Borrowed(&self.0.codes[list_no]))
    }
}

impl InvertedListsWriter for ArrayInvertedListsWriter<'_> {
    fn add_entries(&mut self, list_no: u32, ids: &[u64], codes: &[u8]) -> u64 {
        let list_no = list_no as usize;
        self.0.ids[list_no].extend_from_slice(ids);
        self.0.codes[list_no].extend_from_slice(codes);
        ids.len() as u64
    }

    fn truncate(&mut self, list_no: u32, new_size: usize) {
        let list_no = list_no as usize;
        self.0.ids[list_no].truncate(new_size);
        self.0.codes[list_no].truncate(new_size * self.0.code_size as usize);
    }
}
