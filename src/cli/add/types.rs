use either::Either;
use opencv::core::Mat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Duplicate {
    Overwrite,
    Append,
    Ignore,
}

pub struct ImageData {
    pub path: String,
    pub data: Vec<u8>,
}

pub struct HashedImageData {
    pub path: String,
    pub data: Either<Mat, Vec<u8>>,
    pub hash: Vec<u8>,
}

pub struct ProcessableImage {
    pub path: String,
    pub hash: Vec<u8>,
    pub descriptors: Vec<[u8; 32]>,
}
