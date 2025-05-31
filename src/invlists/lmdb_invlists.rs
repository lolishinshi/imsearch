use std::borrow::Cow;
use std::path::Path;

use anyhow::Result;
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

pub struct LmdbInvertedLists<const N: usize> {
    env: Env<WithTls>,
    meta: Meta,
    db_meta: Database<Str, SerdeBincode<Meta>>,
    db_list: Database<U32<NativeEndian>, Bytes>,
}

impl<const N: usize> LmdbInvertedLists<N> {
    pub fn new<P>(path: P, nlist: u32) -> heed::Result<Self>
    where
        P: AsRef<Path>,
    {
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
            None => Meta { nlist, code_size: N as u32, list_len: vec![0; nlist as usize] },
        };
        assert_eq!(meta.nlist, nlist, "nlist mismatch");
        assert_eq!(meta.code_size, N as u32, "code_size mismatch");
        txn.commit()?;
        Ok(Self { env, meta, db_meta, db_list })
    }
}

impl<const N: usize> InvertedLists<N> for LmdbInvertedLists<N> {
    type Reader<'a>
        = LmdbInvertedListsReader<'a, N>
    where
        Self: 'a;
    type Writer<'a>
        = LmdbInvertedListsWriter<'a, N>
    where
        Self: 'a;

    fn reader(&self) -> Result<Self::Reader<'_>> {
        let txn = self.env.read_txn()?;
        Ok(LmdbInvertedListsReader { txn, meta: &self.meta, db_list: self.db_list })
    }

    fn writer(&mut self) -> Result<Self::Writer<'_>> {
        let txn = self.env.write_txn()?;
        Ok(LmdbInvertedListsWriter {
            txn: Some(txn),
            meta: &mut self.meta,
            db_meta: self.db_meta,
            db_list: self.db_list,
        })
    }
}

pub struct LmdbInvertedListsReader<'a, const N: usize> {
    txn: RoTxn<'a, WithTls>,
    meta: &'a Meta,
    db_list: Database<U32<NativeEndian>, Bytes>,
}

impl<const N: usize> InvertedListsReader<N> for LmdbInvertedListsReader<'_, N> {
    fn nlist(&self) -> u32 {
        self.meta.nlist
    }

    fn list_len(&self, list_no: u32) -> usize {
        let list_no = list_no as usize;
        self.meta.list_len[list_no]
    }

    fn get_list(&self, list_no: u32) -> Result<(Cow<[u64]>, Cow<[u8]>)> {
        let len = self.list_len(list_no);
        if len == 0 {
            return Ok((Cow::Borrowed(&[]), Cow::Borrowed(&[])));
        }
        let data = self.db_list.get(&self.txn, &list_no)?.unwrap();
        // NOTE: 由于 LMDB 不保证数据是对齐的，这里使用 bincode 来反序列化，而不是直接 cast_slice
        let (ids, codes) = bincode::deserialize(data)?;
        Ok((ids, codes))
    }
}

pub struct LmdbInvertedListsWriter<'a, const N: usize> {
    txn: Option<RwTxn<'a>>,
    meta: &'a mut Meta,
    db_meta: Database<Str, SerdeBincode<Meta>>,
    db_list: Database<U32<NativeEndian>, Bytes>,
}

impl<const N: usize> Drop for LmdbInvertedListsWriter<'_, N> {
    fn drop(&mut self) {
        // TODO: 由于在 drop 中清理，这里没办法处理错误
        let mut txn = self.txn.take().unwrap();
        self.db_meta.put(&mut txn, &"meta", &self.meta).unwrap();
        txn.commit().unwrap();
    }
}

impl<const N: usize> InvertedListsReader<N> for LmdbInvertedListsWriter<'_, N> {
    fn nlist(&self) -> u32 {
        self.meta.nlist
    }

    fn list_len(&self, list_no: u32) -> usize {
        self.meta.list_len[list_no as usize]
    }

    fn get_list(&self, list_no: u32) -> Result<(Cow<[u64]>, Cow<[u8]>)> {
        let len = self.list_len(list_no);
        if len == 0 {
            return Ok((Cow::Borrowed(&[]), Cow::Borrowed(&[])));
        }
        let txn = self.txn.as_ref().unwrap();
        let data = self.db_list.get(txn, &list_no)?.unwrap();
        let (ids, codes) = bincode::deserialize(data)?;
        Ok((ids, codes))
    }
}

impl<const N: usize> InvertedListsWriter<N> for LmdbInvertedListsWriter<'_, N> {
    fn add_entries(&mut self, list_no: u32, ids: &[u64], codes: &[u8]) -> Result<u64> {
        assert_eq!(ids.len(), codes.len() / N, "ids and codes length mismatch");
        let (oids, ocodes) = self.get_list(list_no)?;
        let added = ids.len();

        let ids = [&*oids, ids].concat();
        let codes = [&*ocodes, codes].concat();
        let data = bincode::serialize(&(&ids, &codes))?;

        let txn = self.txn.as_mut().unwrap();
        self.db_list.put(txn, &list_no, &data)?;
        self.meta.list_len[list_no as usize] += added;

        Ok(added as u64)
    }

    fn truncate(&mut self, list_no: u32, new_size: usize) -> Result<()> {
        let (ids, codes) = self.get_list(list_no)?;
        let data = bincode::serialize(&(&ids[..new_size], &codes[..new_size * N as usize]))?;
        let txn = self.txn.as_mut().unwrap();
        self.db_list.put(txn, &list_no, &data)?;
        self.meta.list_len[list_no as usize] = new_size;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    // 创建辅助函数，减少重复代码
    fn create_test_data(count: usize, code_size: usize) -> (Vec<u64>, Vec<u8>) {
        let ids: Vec<u64> = (1..=count as u64).collect();
        let codes = vec![42u8; count * code_size];
        (ids, codes)
    }

    #[test]
    fn test_basic_operations() {
        let temp_dir = tempdir().unwrap();
        let mut invlists = LmdbInvertedLists::<64>::new(temp_dir.path(), 3).unwrap();

        // 测试初始状态
        {
            let reader = invlists.reader().unwrap();
            assert_eq!(reader.nlist(), 3);
            for i in 0..3 {
                assert_eq!(reader.list_len(i), 0);
                let (ids, codes) = reader.get_list(i).unwrap();
                assert!(ids.is_empty() && codes.is_empty());
            }
        }

        // 测试添加条目和多次添加
        let (ids1, codes1) = create_test_data(2, 64);
        let (ids2, codes2) = create_test_data(3, 64);
        let ids2: Vec<u64> = ids2.into_iter().map(|x| x + 10).collect(); // 避免重复ID

        {
            let mut writer = invlists.writer().unwrap();
            let added = writer.add_entries(0, &ids1, &codes1).unwrap();
            assert_eq!(added, 2);
            writer.add_entries(0, &ids2, &codes2).unwrap(); // 测试多次添加到同一列表
        }

        // 验证合并后的数据
        {
            let reader = invlists.reader().unwrap();
            assert_eq!(reader.list_len(0), 5);
            let (retrieved_ids, retrieved_codes) = reader.get_list(0).unwrap();
            let expected_ids = [ids1.clone(), ids2.clone()].concat();
            let expected_codes = [codes1.clone(), codes2.clone()].concat();
            assert_eq!(retrieved_ids.as_ref(), &expected_ids);
            assert_eq!(retrieved_codes.as_ref(), &expected_codes);
        }
    }

    #[test]
    fn test_truncate_and_different_lists() {
        let temp_dir = tempdir().unwrap();
        let mut invlists = LmdbInvertedLists::<32>::new(temp_dir.path(), 3).unwrap();

        let (ids, codes) = create_test_data(5, 32);

        {
            let mut writer = invlists.writer().unwrap();
            // 在不同列表中添加数据
            writer.add_entries(0, &ids, &codes).unwrap();
            writer.add_entries(1, &ids[..2], &codes[..2 * 32]).unwrap();

            // 测试截断
            writer.truncate(0, 3).unwrap();
        }

        let reader = invlists.reader().unwrap();
        // 验证截断结果
        assert_eq!(reader.list_len(0), 3);
        let (retrieved_ids, retrieved_codes) = reader.get_list(0).unwrap();
        assert_eq!(retrieved_ids.as_ref(), &ids[..3]);
        assert_eq!(retrieved_codes.as_ref(), &codes[..3 * 32]);

        // 验证不同列表的独立性
        assert_eq!(reader.list_len(1), 2);
        assert_eq!(reader.list_len(2), 0);
    }

    #[test]
    fn test_persistence() {
        let temp_dir = tempdir().unwrap();
        let (ids, codes) = create_test_data(3, 64);

        // 创建并添加数据
        {
            let mut invlists = LmdbInvertedLists::<64>::new(temp_dir.path(), 2).unwrap();
            let mut writer = invlists.writer().unwrap();
            writer.add_entries(0, &ids, &codes).unwrap();
        }

        // 重新打开并验证数据持久化
        let invlists = LmdbInvertedLists::<64>::new(temp_dir.path(), 2).unwrap();
        let reader = invlists.reader().unwrap();
        assert_eq!(reader.nlist(), 2);
        assert_eq!(reader.list_len(0), 3);
        let (retrieved_ids, retrieved_codes) = reader.get_list(0).unwrap();
        assert_eq!(retrieved_ids.as_ref(), &ids);
        assert_eq!(retrieved_codes.as_ref(), &codes);
    }

    #[test]
    fn test_clear() {
        let temp_dir = tempdir().unwrap();
        let mut invlists = LmdbInvertedLists::<32>::new(temp_dir.path(), 3).unwrap();

        // 添加数据到多个列表
        {
            let mut writer = invlists.writer().unwrap();
            for i in 0..3 {
                let (ids, codes) = create_test_data(2, 32);
                writer.add_entries(i, &ids, &codes).unwrap();
            }
        }

        // 清空并验证
        {
            let mut writer = invlists.writer().unwrap();
            writer.clear().unwrap();
        }

        let reader = invlists.reader().unwrap();
        for i in 0..3 {
            assert_eq!(reader.list_len(i), 0);
            let (ids, codes) = reader.get_list(i).unwrap();
            assert!(ids.is_empty() && codes.is_empty());
        }
    }

    #[test]
    #[should_panic(expected = "nlist mismatch")]
    fn test_nlist_mismatch() {
        let temp_dir = tempdir().unwrap();
        {
            let mut invlists = LmdbInvertedLists::<64>::new(temp_dir.path(), 5).unwrap();
            invlists.writer().unwrap();
        }
        let _invlists = LmdbInvertedLists::<64>::new(temp_dir.path(), 10).unwrap();
    }

    #[test]
    #[should_panic(expected = "code_size mismatch")]
    fn test_code_size_mismatch() {
        let temp_dir = tempdir().unwrap();
        {
            let mut invlists = LmdbInvertedLists::<64>::new(temp_dir.path(), 5).unwrap();
            invlists.writer().unwrap();
        }
        let _invlists = LmdbInvertedLists::<32>::new(temp_dir.path(), 5).unwrap();
    }

    #[test]
    fn test_merge_from() {
        let temp_dir1 = tempdir().unwrap();
        let temp_dir2 = tempdir().unwrap();
        let mut invlists1 = LmdbInvertedLists::<64>::new(temp_dir1.path(), 3).unwrap();
        let mut invlists2 = LmdbInvertedLists::<64>::new(temp_dir2.path(), 3).unwrap();

        // 准备测试数据
        let (ids1, codes1) = create_test_data(2, 64);
        let (ids2, codes2) = create_test_data(2, 64);
        let ids2: Vec<u64> = ids2.into_iter().map(|x| x + 10).collect();

        // 在两个索引中添加数据
        {
            let mut writer1 = invlists1.writer().unwrap();
            writer1.add_entries(0, &ids1, &codes1).unwrap();

            let mut writer2 = invlists2.writer().unwrap();
            writer2.add_entries(0, &ids2, &codes2).unwrap();
            writer2.add_entries(2, &[100], &vec![99u8; 64]).unwrap();
        }

        // 执行合并
        {
            let mut writer1 = invlists1.writer().unwrap();
            let mut writer2 = invlists2.writer().unwrap();
            writer1.merge_from(&mut writer2, 1000).unwrap();
        }

        // 验证合并结果
        let reader1 = invlists1.reader().unwrap();
        assert_eq!(reader1.list_len(0), 4); // 原有2个 + 合并2个
        let (merged_ids, _) = reader1.get_list(0).unwrap();
        assert_eq!(merged_ids.as_ref(), &[1, 2, 1011, 1012]); // ids2 + 1000偏移

        assert_eq!(reader1.list_len(2), 1);
        let (ids_list2, _) = reader1.get_list(2).unwrap();
        assert_eq!(ids_list2.as_ref(), &[1100]); // 100 + 1000偏移

        // 验证源索引被清空
        let reader2 = invlists2.reader().unwrap();
        for i in 0..3 {
            assert_eq!(reader2.list_len(i), 0);
        }
    }
}
