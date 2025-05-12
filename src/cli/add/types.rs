use either::Either;
use opencv::core::Mat;

pub struct HashedImageData {
    pub path: String,
    pub data: Either<Mat, Vec<u8>>,
    pub hash: Vec<u8>,
}

pub struct ProcessableImage {
    pub path: String,
    pub hash: Vec<u8>,
    pub descriptors: Mat,
}
