mod array_invlists;
mod lmdb_invlists;

use std::borrow::Cow;

use anyhow::Result;
pub use array_invlists::*;
pub use lmdb_invlists::*;

pub trait InvertedLists<const N: usize> {
    type Reader<'a>: InvertedListsReader<N>
    where
        Self: 'a;
    type Writer<'a>: InvertedListsWriter<N>
    where
        Self: 'a;

    fn reader(&self) -> Result<Self::Reader<'_>>;

    fn writer(&mut self) -> Result<Self::Writer<'_>>;
}

pub trait InvertedListsReader<const N: usize> {
    /// 返回倒排表的列表数量
    fn nlist(&self) -> u32;

    /// 返回指定倒排表的元素数量
    fn list_len(&self, list_no: u32) -> usize;

    /// 返回指定倒排表中向量的 ID 列表和数据
    fn get_list(&self, list_no: u32) -> Result<(Cow<[u64]>, Cow<[[u8; N]]>)>;
}

pub trait InvertedListsWriter<const N: usize>: InvertedListsReader<N> {
    /// 往指定倒排表中添加元素
    ///
    /// 返回添加的元素数量
    fn add_entries(&mut self, list_no: u32, ids: &[u64], codes: &[[u8; N]]) -> Result<u64>;

    /// 清空指定倒排表
    fn clear(&mut self, list_no: u32) -> Result<()>;

    /// 合并另一个倒排列表，被合并的倒排列表会被清空
    fn merge_from(&mut self, other: &mut impl InvertedListsWriter<N>) -> Result<()> {
        assert_eq!(self.nlist(), other.nlist(), "nlist mismatch");
        for i in 0..self.nlist() {
            let (ids, codes) = other.get_list(i)?;
            self.add_entries(i, &ids, &codes)?;
            // 一边合并一边清空，保证内存及时释放
            other.clear(i)?;
        }
        Ok(())
    }
}
