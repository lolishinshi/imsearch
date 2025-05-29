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

pub struct LmdbInvertedLists {
    env: Env<WithTls>,
    meta: Meta,
    db_meta: Database<Str, SerdeBincode<Meta>>,
    db_list: Database<U32<NativeEndian>, Bytes>,
}

impl LmdbInvertedLists {
    pub fn new<P>(path: P, nlist: u32, code_size: u32) -> heed::Result<Self>
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
            None => Meta { nlist, code_size, list_len: vec![0; nlist as usize] },
        };
        assert_eq!(meta.nlist, nlist, "nlist mismatch");
        assert_eq!(meta.code_size, code_size, "code_size mismatch");
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

    fn get_list(&self, list_no: u32) -> (Cow<[u64]>, Cow<[u8]>) {
        let len = self.list_len(list_no);
        if len == 0 {
            return (Cow::Borrowed(&[]), Cow::Borrowed(&[]));
        }
        let data = self.db_list.get(&self.txn, &list_no).unwrap().unwrap();
        // NOTE: 由于 LMDB 不保证数据是对齐的，这里使用 bincode 来反序列化，而不是直接 cast_slice
        let (ids, codes) = bincode::deserialize(data).unwrap();
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

    fn get_list(&self, list_no: u32) -> (Cow<[u64]>, Cow<[u8]>) {
        let len = self.list_len(list_no);
        if len == 0 {
            return (Cow::Borrowed(&[]), Cow::Borrowed(&[]));
        }
        let data = self.db_list.get(&self.txn, &list_no).unwrap().unwrap();
        let (ids, codes) = bincode::deserialize(data).unwrap();
        (ids, codes)
    }
}

impl InvertedListsWriter for LmdbInvertedListsWriter<'_> {
    fn add_entries(&mut self, list_no: u32, ids: &[u64], codes: &[u8]) -> u64 {
        let (ids_, codes_) = self.get_list(list_no);
        let (mut ids_, mut codes_) = (ids_.to_vec(), codes_.to_vec());
        ids_.extend_from_slice(ids);
        codes_.extend_from_slice(codes);
        let data = bincode::serialize(&(ids_, codes_)).unwrap();
        self.db_list.put(&mut self.txn, &list_no, &data).unwrap();
        self.meta.list_len[list_no as usize] += ids.len();
        ids.len() as u64
    }

    fn truncate(&mut self, list_no: u32, new_size: usize) {
        let (ids, codes) = self.get_list(list_no);
        let data =
            bincode::serialize(&(&ids[..new_size], &codes[..new_size * self.code_size() as usize]))
                .unwrap();
        self.db_list.put(&mut self.txn, &list_no, &data).unwrap();
        self.meta.list_len[list_no as usize] = new_size;
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_new_lmdb_inverted_lists() {
        let temp_dir = tempdir().unwrap();
        let nlist = 10;
        let code_size = 32;

        let invlists = LmdbInvertedLists::new(temp_dir.path(), nlist, code_size).unwrap();

        let reader = invlists.reader().unwrap();
        assert_eq!(reader.nlist(), nlist);
        assert_eq!(reader.code_size(), code_size);

        // 测试初始状态下所有列表为空
        for i in 0..nlist {
            assert_eq!(reader.list_len(i), 0);
            let (ids, codes) = reader.get_list(i);
            assert!(ids.is_empty());
            assert!(codes.is_empty());
        }
    }

    #[test]
    fn test_add_entries() {
        let temp_dir = tempdir().unwrap();
        let nlist = 5;
        let code_size = 16;

        let mut invlists = LmdbInvertedLists::new(temp_dir.path(), nlist, code_size).unwrap();

        // 添加一些条目
        let ids = vec![1, 2, 3];
        let codes = vec![0u8; 3 * code_size as usize]; // 3个向量，每个向量16字节

        {
            let mut writer = invlists.writer().unwrap();
            let added = writer.add_entries(0, &ids, &codes);
            assert_eq!(added, 3);
            writer.commit().unwrap();
        }

        // 验证添加的数据
        let reader = invlists.reader().unwrap();
        assert_eq!(reader.list_len(0), 3);
        let (retrieved_ids, retrieved_codes) = reader.get_list(0);
        assert_eq!(retrieved_ids.as_ref(), &ids);
        assert_eq!(retrieved_codes.as_ref(), &codes);
    }

    #[test]
    fn test_multiple_add_entries() {
        let temp_dir = tempdir().unwrap();
        let nlist = 3;
        let code_size = 8;

        let mut invlists = LmdbInvertedLists::new(temp_dir.path(), nlist, code_size).unwrap();

        // 第一次添加
        let ids1 = vec![1, 2];
        let codes1 = vec![1u8; 2 * code_size as usize];

        // 第二次添加
        let ids2 = vec![3, 4];
        let codes2 = vec![2u8; 2 * code_size as usize];

        {
            let mut writer = invlists.writer().unwrap();
            writer.add_entries(0, &ids1, &codes1);
            writer.add_entries(0, &ids2, &codes2);
            writer.commit().unwrap();
        }

        // 验证合并后的数据
        let reader = invlists.reader().unwrap();
        assert_eq!(reader.list_len(0), 4);
        let (retrieved_ids, retrieved_codes) = reader.get_list(0);

        let expected_ids = [&ids1[..], &ids2[..]].concat();
        let expected_codes = [&codes1[..], &codes2[..]].concat();

        assert_eq!(retrieved_ids.as_ref(), &expected_ids);
        assert_eq!(retrieved_codes.as_ref(), &expected_codes);
    }

    #[test]
    fn test_truncate() {
        let temp_dir = tempdir().unwrap();
        let nlist = 2;
        let code_size = 4;

        let mut invlists = LmdbInvertedLists::new(temp_dir.path(), nlist, code_size).unwrap();

        // 添加5个条目
        let ids = vec![1, 2, 3, 4, 5];
        let codes = vec![0u8; 5 * code_size as usize];

        {
            let mut writer = invlists.writer().unwrap();
            writer.add_entries(0, &ids, &codes);

            // 截断为3个条目
            writer.truncate(0, 3);
            writer.commit().unwrap();
        }

        // 验证截断后的数据
        let reader = invlists.reader().unwrap();
        assert_eq!(reader.list_len(0), 3);
        let (retrieved_ids, retrieved_codes) = reader.get_list(0);

        assert_eq!(retrieved_ids.as_ref(), &ids[..3]);
        assert_eq!(retrieved_codes.as_ref(), &codes[..3 * code_size as usize]);
    }

    #[test]
    fn test_different_lists() {
        let temp_dir = tempdir().unwrap();
        let nlist = 3;
        let code_size = 8;

        let mut invlists = LmdbInvertedLists::new(temp_dir.path(), nlist, code_size).unwrap();

        // 在不同的列表中添加数据
        let ids0 = vec![1, 2];
        let codes0 = vec![0u8; 2 * code_size as usize];

        let ids1 = vec![3, 4, 5];
        let codes1 = vec![1u8; 3 * code_size as usize];

        {
            let mut writer = invlists.writer().unwrap();
            writer.add_entries(0, &ids0, &codes0);
            writer.add_entries(1, &ids1, &codes1);
            writer.commit().unwrap();
        }

        // 验证不同列表的数据
        let reader = invlists.reader().unwrap();

        assert_eq!(reader.list_len(0), 2);
        assert_eq!(reader.list_len(1), 3);
        assert_eq!(reader.list_len(2), 0);

        let (retrieved_ids0, retrieved_codes0) = reader.get_list(0);
        let (retrieved_ids1, retrieved_codes1) = reader.get_list(1);
        let (retrieved_ids2, retrieved_codes2) = reader.get_list(2);

        assert_eq!(retrieved_ids0.as_ref(), &ids0);
        assert_eq!(retrieved_codes0.as_ref(), &codes0);
        assert_eq!(retrieved_ids1.as_ref(), &ids1);
        assert_eq!(retrieved_codes1.as_ref(), &codes1);
        assert!(retrieved_ids2.is_empty());
        assert!(retrieved_codes2.is_empty());
    }

    #[test]
    fn test_persistence() {
        let temp_dir = tempdir().unwrap();
        let nlist = 2;
        let code_size = 16;

        let ids = vec![100, 200, 300];
        let codes = vec![42u8; 3 * code_size as usize];

        // 创建并添加数据
        {
            let mut invlists = LmdbInvertedLists::new(temp_dir.path(), nlist, code_size).unwrap();
            let mut writer = invlists.writer().unwrap();
            writer.add_entries(0, &ids, &codes);
            writer.commit().unwrap();
        }

        // 重新打开并验证数据持久化
        {
            let invlists = LmdbInvertedLists::new(temp_dir.path(), nlist, code_size).unwrap();
            let reader = invlists.reader().unwrap();

            assert_eq!(reader.nlist(), nlist);
            assert_eq!(reader.code_size(), code_size);
            assert_eq!(reader.list_len(0), 3);

            let (retrieved_ids, retrieved_codes) = reader.get_list(0);
            assert_eq!(retrieved_ids.as_ref(), &ids);
            assert_eq!(retrieved_codes.as_ref(), &codes);
        }
    }

    #[test]
    fn test_writer_clear() {
        let temp_dir = tempdir().unwrap();
        let nlist = 4;
        let code_size = 8;

        let mut invlists = LmdbInvertedLists::new(temp_dir.path(), nlist, code_size).unwrap();

        // 在多个列表中添加数据
        {
            let mut writer = invlists.writer().unwrap();
            for i in 0..nlist {
                let ids = vec![i as u64 + 1, i as u64 + 2];
                let codes = vec![i as u8; 2 * code_size as usize];
                writer.add_entries(i, &ids, &codes);
            }
            writer.commit().unwrap();
        }

        // 验证所有列表都有数据
        {
            let reader = invlists.reader().unwrap();
            for i in 0..nlist {
                assert_eq!(reader.list_len(i), 2);
            }
        }

        // 清空所有列表
        {
            let mut writer = invlists.writer().unwrap();
            writer.clear();
            writer.commit().unwrap();
        }

        // 验证所有列表都已清空
        {
            let reader = invlists.reader().unwrap();
            for i in 0..nlist {
                assert_eq!(reader.list_len(i), 0);
                let (ids, codes) = reader.get_list(i);
                assert!(ids.is_empty());
                assert!(codes.is_empty());
            }
        }
    }

    #[test]
    #[should_panic(expected = "nlist mismatch")]
    fn test_nlist_mismatch() {
        let temp_dir = tempdir().unwrap();

        // 创建 nlist=5 的索引，然后销毁
        {
            let mut invlists1 = LmdbInvertedLists::new(temp_dir.path(), 5, 32).unwrap();
            // 确保 meta 被写入数据库
            let writer = invlists1.writer().unwrap();
            writer.commit().unwrap();
        } // 这里 invlists1 被销毁

        // 尝试用不同的 nlist 打开同一个路径应该失败
        let _invlists2 = LmdbInvertedLists::new(temp_dir.path(), 10, 32).unwrap();
    }

    #[test]
    #[should_panic(expected = "code_size mismatch")]
    fn test_code_size_mismatch() {
        let temp_dir = tempdir().unwrap();

        // 创建 code_size=32 的索引，然后销毁
        {
            let mut invlists1 = LmdbInvertedLists::new(temp_dir.path(), 5, 32).unwrap();
            // 确保 meta 被写入数据库
            let writer = invlists1.writer().unwrap();
            writer.commit().unwrap();
        } // 这里 invlists1 被销毁

        // 尝试用不同的 code_size 打开同一个路径应该失败
        let _invlists2 = LmdbInvertedLists::new(temp_dir.path(), 5, 64).unwrap();
    }

    #[test]
    fn test_merge_from() {
        let temp_dir1 = tempdir().unwrap();
        let temp_dir2 = tempdir().unwrap();
        let nlist = 3;
        let code_size = 8;

        let mut invlists1 = LmdbInvertedLists::new(temp_dir1.path(), nlist, code_size).unwrap();
        let mut invlists2 = LmdbInvertedLists::new(temp_dir2.path(), nlist, code_size).unwrap();

        // 在第一个索引中添加数据
        {
            let mut writer1 = invlists1.writer().unwrap();
            writer1.add_entries(0, &[1, 2], &vec![1u8; 2 * code_size as usize]);
            writer1.add_entries(1, &[3], &vec![2u8; code_size as usize]);
            writer1.commit().unwrap();
        }

        // 在第二个索引中添加数据
        {
            let mut writer2 = invlists2.writer().unwrap();
            writer2.add_entries(0, &[4, 5], &vec![3u8; 2 * code_size as usize]);
            writer2.add_entries(2, &[6], &vec![4u8; code_size as usize]);
            writer2.commit().unwrap();
        }

        // 合并第二个索引到第一个
        {
            let mut writer1 = invlists1.writer().unwrap();
            let mut writer2 = invlists2.writer().unwrap();
            writer1.merge_from(&mut writer2, 100); // 添加 100 的偏移量
            writer1.commit().unwrap();
            writer2.commit().unwrap();
        }

        // 验证合并结果
        let reader1 = invlists1.reader().unwrap();

        // 列表 0: [1, 2] + [104, 105] (4+100, 5+100)
        assert_eq!(reader1.list_len(0), 4);
        let (ids0, codes0) = reader1.get_list(0);
        assert_eq!(ids0.as_ref(), &[1, 2, 104, 105]);
        let expected_codes0 =
            [&vec![1u8; 2 * code_size as usize][..], &vec![3u8; 2 * code_size as usize][..]]
                .concat();
        assert_eq!(codes0.as_ref(), &expected_codes0);

        // 列表 1: [3] (没有变化)
        assert_eq!(reader1.list_len(1), 1);
        let (ids1, codes1) = reader1.get_list(1);
        assert_eq!(ids1.as_ref(), &[3]);
        assert_eq!(codes1.as_ref(), &vec![2u8; code_size as usize]);

        // 列表 2: [106] (6+100)
        assert_eq!(reader1.list_len(2), 1);
        let (ids2, codes2) = reader1.get_list(2);
        assert_eq!(ids2.as_ref(), &[106]);
        assert_eq!(codes2.as_ref(), &vec![4u8; code_size as usize]);

        // 验证第二个索引已被清空
        let reader2 = invlists2.reader().unwrap();
        for i in 0..nlist {
            assert_eq!(reader2.list_len(i), 0);
        }
    }
}
