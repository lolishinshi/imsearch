use std::convert::TryInto;
use std::mem;

use crate::db::database::ImageColumnFamily;
use crate::db::database::MetaData;
use rocksdb::{Error, Options, DB};

pub fn init_column_family(db: &DB) -> Result<(), Error> {
    let opts = &Options::default();
    db.create_cf(ImageColumnFamily::NewFeature, &opts)?;
    db.create_cf(ImageColumnFamily::IdToFeature, &opts)?;
    db.create_cf(ImageColumnFamily::IdToImageId, &opts)?;
    db.create_cf(ImageColumnFamily::IdToImage, &opts)?;
    db.create_cf(ImageColumnFamily::ImageList, &opts)?;
    db.create_cf(ImageColumnFamily::MetaData, &opts)?;

    let meta_data = db.cf_handle(ImageColumnFamily::MetaData.as_ref()).unwrap();
    db.put_cf(&meta_data, MetaData::TotalImages, 0u64.to_le_bytes())?;
    db.put_cf(&meta_data, MetaData::TotalFeatures, 0u64.to_le_bytes())?;

    Ok(())
}

pub trait CompactBytes {
    fn to_cpt_bytes(&self) -> &[u8];
    fn from_cpt_bytes(b: &[u8]) -> Self;
}

macro_rules! impl_compact_bytes {
    ($($t:ty),*) => {
        $(impl CompactBytes for $t {
            #[inline]
            fn to_cpt_bytes(&self) -> &[u8] {
                unsafe {
                    let b = mem::transmute::<_, &[u8; (Self::BITS / 8) as usize]>(self);
                    // SAFETY: `index` is between 0..=8
                    let index = Self::BITS / 8 - self.leading_zeros() / 8;
                    b.get_unchecked(..index as usize)
                }
            }

            #[inline]
            fn from_cpt_bytes(bytes: &[u8]) -> Self {
                let mut buf = [0u8; (Self::BITS / 8) as usize];
                buf[..bytes.len()].copy_from_slice(bytes);
                Self::from_le_bytes(buf)
            }
        })*
    }
}

impl_compact_bytes!(i32, u64);

pub fn bytes_to_u64<T: AsRef<[u8]>>(bytes: T) -> u64 {
    u64::from_le_bytes(
        bytes
            .as_ref()
            .try_into()
            .expect("byte cannot convert to u64"),
    )
}

pub fn bytes_to_i32<T: AsRef<[u8]>>(bytes: T) -> i32 {
    i32::from_le_bytes(
        bytes
            .as_ref()
            .try_into()
            .expect("byte cannot convert to i32"),
    )
}

#[cfg(test)]
mod tests {
    use super::CompactBytes;

    #[test]
    fn to_bytes() {
        assert_eq!(1i32.to_cpt_bytes(), &[1]);
        assert_eq!(255i32.to_cpt_bytes(), &[255]);
        assert_eq!(256u64.to_cpt_bytes(), &[0, 1]);
        assert_eq!(65535u64.to_cpt_bytes(), &[255, 255]);
        assert_eq!(2147483647i32.to_cpt_bytes(), &[255, 255, 255, 127]);
    }

    #[test]
    fn from_bytes() {
        assert_eq!(u64::from_cpt_bytes(&[1]), 1);
        assert_eq!(i32::from_cpt_bytes(&[255, 255, 255, 127]), 2147483647);
    }

    #[test]
    fn convert_between_u64_bytes() {
        for i in 0..(1 << 16) {
            let b = i.to_cpt_bytes();
            assert_eq!(u64::from_cpt_bytes(b), i);
        }
    }
}
