use std::borrow::Cow;
use std::path::Path;

use anyhow::Result;
use byteorder::NativeEndian;
use heed::types::{Bytes, SerdeBincode, Str, U32};
use heed::{Database, Env, EnvOpenOptions, RoTxn, RwTxn, WithTls};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use super::{InvertedLists, InvertedListsReader, InvertedListsWriter};

/// 存储在 lmdb 中的倒排列表元数据
#[derive(Serialize, Deserialize)]
struct Meta {
    /// 倒排列表数量
    nlist: u32,
    /// 每个向量的大小，用于校验
    code_size: u32,
    /// 每个倒排列表的元素数量，用于快速统计
    list_len: Vec<usize>,
}

/// 存储在 lmdb 中的倒排列表数据
/// 由于 lmdb 不保证读取到的数据是对齐的，因此想零拷贝序列化很麻烦，干脆直接使用 serde
#[serde_as]
#[derive(Serialize, Deserialize)]
struct Entry<const N: usize> {
    ids: Vec<u64>,
    #[serde_as(as = "Vec<[_; N]>")]
    codes: Vec<[u8; N]>,
}

pub struct LmdbInvertedLists<const N: usize> {
    /// lmdb env，此处使用了 Thread Local Storage 提升速度
    env: Env<WithTls>,
    /// 元数据
    meta: Meta,
    /// 元数据数据库
    db_meta: Database<Str, SerdeBincode<Meta>>,
    /// 倒排列表数据库
    db_list: Database<U32<NativeEndian>, Bytes>,
}

impl<const N: usize> LmdbInvertedLists<N> {
    /// 创建一个新的倒排列表
    pub fn new<P>(path: P, nlist: u32) -> heed::Result<Self>
    where
        P: AsRef<Path>,
    {
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(1 << 40) // 此处直接分配 1TiB 大小，后续可以考虑动态增长
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
        self.meta.list_len[list_no as usize]
    }

    fn get_list(&self, list_no: u32) -> Result<(Cow<[u64]>, Cow<[[u8; N]]>)> {
        let len = self.list_len(list_no);
        if len == 0 {
            return Ok((Cow::Borrowed(&[]), Cow::Borrowed(&[])));
        }
        let data = self.db_list.get(&self.txn, &list_no)?.unwrap();
        let entry = bincode::deserialize::<Entry<N>>(data)?;
        Ok((Cow::Owned(entry.ids), Cow::Owned(entry.codes)))
    }
}

pub struct LmdbInvertedListsWriter<'a, const N: usize> {
    // 由于 txn.commit 需要消耗所有权，这里用 Option 包裹，确保 drop 中能调用 commit
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

    fn get_list(&self, list_no: u32) -> Result<(Cow<[u64]>, Cow<[[u8; N]]>)> {
        let len = self.list_len(list_no);
        if len == 0 {
            return Ok((Cow::Borrowed(&[]), Cow::Borrowed(&[])));
        }
        let txn = self.txn.as_ref().unwrap();
        let data = self.db_list.get(txn, &list_no)?.unwrap();
        let entry = bincode::deserialize::<Entry<N>>(data)?;
        Ok((Cow::Owned(entry.ids), Cow::Owned(entry.codes)))
    }
}

impl<const N: usize> InvertedListsWriter<N> for LmdbInvertedListsWriter<'_, N> {
    fn add_entries(&mut self, list_no: u32, ids: &[u64], codes: &[[u8; N]]) -> Result<u64> {
        assert_eq!(ids.len(), codes.len(), "ids and codes length mismatch");
        let (oids, ocodes) = self.get_list(list_no)?;
        let added = ids.len();

        let data = bincode::serialize(&Entry {
            ids: [&*oids, ids].concat(),
            codes: [&*ocodes, codes].concat(),
        })?;

        let txn = self.txn.as_mut().unwrap();
        self.db_list.put(txn, &list_no, &data)?;
        self.meta.list_len[list_no as usize] += added;

        Ok(added as u64)
    }

    fn clear(&mut self, list_no: u32) -> Result<()> {
        let data = bincode::serialize(&Entry::<N> { ids: vec![], codes: vec![] })?;
        let txn = self.txn.as_mut().unwrap();
        self.db_list.put(txn, &list_no, &data)?;
        self.meta.list_len[list_no as usize] = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    // 创建辅助函数，减少重复代码
    fn create_test_data<const N: usize>(count: usize) -> (Vec<u64>, Vec<[u8; N]>) {
        let ids: Vec<u64> = (1..=count as u64).collect();
        let codes = vec![[42u8; N]; count];
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
        let (ids1, codes1) = create_test_data(2);
        let (ids2, codes2) = create_test_data(3);
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
    fn test_persistence() {
        let temp_dir = tempdir().unwrap();
        let (ids, codes) = create_test_data(3);

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
                let (ids, codes) = create_test_data(2);
                writer.add_entries(i, &ids, &codes).unwrap();
            }
        }

        // 清空并验证
        {
            let mut writer = invlists.writer().unwrap();
            for i in 0..3 {
                writer.clear(i).unwrap();
            }
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
        let (ids1, codes1) = create_test_data(2);
        let (ids2, codes2) = create_test_data(2);
        let ids2: Vec<u64> = ids2.into_iter().map(|x| x + 10).collect();

        // 在两个索引中添加数据
        {
            let mut writer1 = invlists1.writer().unwrap();
            writer1.add_entries(0, &ids1, &codes1).unwrap();

            let mut writer2 = invlists2.writer().unwrap();
            writer2.add_entries(0, &ids2, &codes2).unwrap();
            writer2.add_entries(2, &[100], &vec![[99u8; 64]; 1]).unwrap();
        }

        // 执行合并
        {
            let mut writer1 = invlists1.writer().unwrap();
            let mut writer2 = invlists2.writer().unwrap();
            writer1.merge_from(&mut writer2).unwrap();
        }

        // 验证合并结果
        let reader1 = invlists1.reader().unwrap();
        assert_eq!(reader1.list_len(0), 4); // 原有2个 + 合并2个
        let (merged_ids, _) = reader1.get_list(0).unwrap();
        assert_eq!(merged_ids.as_ref(), &[1, 2, 11, 12]);

        assert_eq!(reader1.list_len(2), 1);
        let (ids_list2, _) = reader1.get_list(2).unwrap();
        assert_eq!(ids_list2.as_ref(), &[100]);

        // 验证源索引被清空
        let reader2 = invlists2.reader().unwrap();
        for i in 0..3 {
            assert_eq!(reader2.list_len(i), 0);
        }
    }
}
