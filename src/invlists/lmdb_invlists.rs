use anyhow::Result;
use bytemuck::checked::cast_slice;
use byteorder::NativeEndian;
use heed::types::{Bytes, SerdeBincode, Str, U32};
use heed::{Database, Env, EnvOpenOptions, RoTxn, RwTxn, WithTls};
use serde::{Deserialize, Serialize};

use super::{InvertedLists, InvertedListsReader, InvertedListsWriter};

#[derive(Serialize, Deserialize)]
struct Meta {
    nlist: u32,
    code_size: u32,
    list_len: Vec<usize>,
}

pub struct LmdbInvertedLists {
    env: Env<WithTls>,
    meta: Meta,
    db_meta: Database<Str, SerdeBincode<Meta>>,
    db_list: Database<U32<NativeEndian>, Bytes>,
}

impl LmdbInvertedLists {
    pub fn new(path: &str, nlist: u32, code_size: u32) -> heed::Result<Self> {
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(1 << 40) // 1TiB
                .max_dbs(3)
                .open(path)?
        };
        let mut txn = env.write_txn()?;
        let db_meta = env.create_database::<Str, SerdeBincode<Meta>>(&mut txn, Some("meta"))?;
        let db_list = env.create_database::<U32<NativeEndian>, Bytes>(&mut txn, Some("list"))?;
        let meta = match db_meta.get(&mut txn, &"meta")? {
            Some(meta) => meta,
            None => Meta { nlist, code_size, list_len: vec![0; nlist as usize] },
        };
        assert_eq!(meta.nlist, nlist);
        assert_eq!(meta.code_size, code_size);
        txn.commit()?;
        Ok(Self { env, meta, db_meta, db_list })
    }
}

impl InvertedLists for LmdbInvertedLists {
    type Reader<'a>
        = LmdbInvertedListsReader<'a>
    where
        Self: 'a;
    type Writer<'a>
        = LmdbInvertedListsWriter<'a>
    where
        Self: 'a;

    fn reader(&self) -> Result<Self::Reader<'_>> {
        let txn = self.env.read_txn()?;
        Ok(LmdbInvertedListsReader { txn, meta: &self.meta, db_list: self.db_list })
    }

    fn writer(&mut self) -> Result<Self::Writer<'_>> {
        let txn = self.env.write_txn()?;
        Ok(LmdbInvertedListsWriter {
            txn,
            meta: &mut self.meta,
            db_meta: self.db_meta,
            db_list: self.db_list,
        })
    }
}

pub struct LmdbInvertedListsReader<'a> {
    txn: RoTxn<'a, WithTls>,
    meta: &'a Meta,
    db_list: Database<U32<NativeEndian>, Bytes>,
}

impl InvertedListsReader for LmdbInvertedListsReader<'_> {
    fn nlist(&self) -> u32 {
        self.meta.nlist
    }

    fn code_size(&self) -> u32 {
        self.meta.code_size
    }

    fn list_len(&self, list_no: u32) -> usize {
        let list_no = list_no as usize;
        self.meta.list_len[list_no]
    }

    fn get_list(&self, list_no: u32) -> (&[u64], &[u8]) {
        let list_len = self.list_len(list_no);
        let list_data = self.db_list.get(&self.txn, &list_no).unwrap().unwrap();
        let (ids, codes) = list_data.split_at(list_len * 8);
        let ids = unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const u64, list_len) };
        (ids, codes)
    }
}

pub struct LmdbInvertedListsWriter<'a> {
    txn: RwTxn<'a>,
    meta: &'a mut Meta,
    db_meta: Database<Str, SerdeBincode<Meta>>,
    db_list: Database<U32<NativeEndian>, Bytes>,
}

impl LmdbInvertedListsWriter<'_> {
    pub fn commit(mut self) -> Result<()> {
        self.db_meta.put(&mut self.txn, &"meta", &self.meta)?;
        self.txn.commit()?;
        Ok(())
    }
}

impl InvertedListsReader for LmdbInvertedListsWriter<'_> {
    fn nlist(&self) -> u32 {
        self.meta.nlist
    }

    fn code_size(&self) -> u32 {
        self.meta.code_size
    }

    fn list_len(&self, list_no: u32) -> usize {
        self.meta.list_len[list_no as usize]
    }

    fn get_list(&self, list_no: u32) -> (&[u64], &[u8]) {
        let list_len = self.list_len(list_no);
        let list_data = self.db_list.get(&self.txn, &list_no).unwrap().unwrap();
        let (ids, codes) = list_data.split_at(list_len * 8);
        let ids = cast_slice(ids);
        (ids, codes)
    }
}

impl InvertedListsWriter for LmdbInvertedListsWriter<'_> {
    fn add_entries(&mut self, list_no: u32, ids: &[u64], codes: &[u8]) -> u64 {
        let (ids_, codes_) = self.get_list(list_no);
        let mut buf = vec![];
        buf.extend_from_slice(cast_slice(ids_));
        buf.extend_from_slice(cast_slice(ids));
        buf.extend_from_slice(codes_);
        buf.extend_from_slice(codes);
        self.db_list.put(&mut self.txn, &list_no, &buf).unwrap();
        self.meta.list_len[list_no as usize] += ids.len();
        ids.len() as u64
    }

    fn truncate(&mut self, list_no: u32, new_size: usize) {
        let (ids, codes) = self.get_list(list_no);
        let mut buf = vec![];
        buf.extend_from_slice(cast_slice(&ids[..new_size]));
        buf.extend_from_slice(&codes[..new_size * self.code_size() as usize]);
        self.db_list.put(&mut self.txn, &list_no, &buf).unwrap();
        self.meta.list_len[list_no as usize] = new_size;
    }
}
