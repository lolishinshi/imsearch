use std::borrow::Cow;

use anyhow::Result;

use super::InvertedLists;

/// 垂直堆叠多个倒排列表，通常用于合并工作
pub struct VStackInvlists<const N: usize, T> {
    nlist: usize,
    invlists: Vec<T>,
}

impl<const N: usize, T> VStackInvlists<N, T>
where
    T: InvertedLists<N>,
{
    pub fn new(invlists: Vec<T>) -> Self {
        assert!(!invlists.is_empty(), "invlists is empty");
        let nlist = invlists[0].nlist();
        for invlist in &invlists {
            assert_eq!(invlist.nlist(), nlist, "nlist mismatch");
        }
        Self { nlist, invlists }
    }
}

impl<const N: usize, T> InvertedLists<N> for VStackInvlists<N, T>
where
    T: InvertedLists<N>,
{
    fn nlist(&self) -> usize {
        self.nlist
    }

    fn list_len(&self, list_no: usize) -> usize {
        let mut len = 0;
        for invlist in &self.invlists {
            len += invlist.list_len(list_no);
        }
        len
    }

    fn get_list(&self, list_no: usize) -> Result<(Cow<'_, [u64]>, Cow<'_, [[u8; N]]>)> {
        let mut ids: Vec<u64> = vec![];
        let mut codes: Vec<[u8; N]> = vec![];
        for invlist in &self.invlists {
            let (i, c) = invlist.get_list(list_no)?;
            ids.extend_from_slice(&*i);
            codes.extend_from_slice(&*c);
        }
        Ok((Cow::Owned(ids), Cow::Owned(codes)))
    }

    fn add_entry(&mut self, _list_no: usize, _ids: u64, _codes: &[u8; N]) -> Result<()> {
        unimplemented!("VStackInvlists 不支持更新操作")
    }
}
