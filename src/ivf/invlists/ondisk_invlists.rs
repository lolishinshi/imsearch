pub struct OnDiskInvlists<const N: usize> {
    pub nlist: u32,
    pub codes: Vec<Vec<[u8; N]>>,
    pub ids: Vec<Vec<u64>>,
}
