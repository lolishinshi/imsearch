use opencv::prelude::*;

/// An abstraction of 2d u8 array
pub trait Matrix {
    /// Return matrix width
    fn width(&self) -> usize;
    /// Return matrix height
    fn height(&self) -> usize;
    /// Return raw pointer to data
    fn as_ptr(&self) -> *const u8;
    /// Get specific line
    fn line(&self, n: usize) -> &[u8];
    /// Iterate over lines
    fn iter_lines(&self) -> MatrixLineIterator
    where
        Self: Sized,
    {
        MatrixLineIterator {
            matrix: self,
            current_line: 0,
        }
    }
}

pub struct MatrixLineIterator<'a> {
    matrix: &'a dyn Matrix,
    current_line: usize,
}

impl<'a> Iterator for MatrixLineIterator<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_line < self.matrix.height() as usize {
            self.current_line += 1;
            Some(self.matrix.line(self.current_line - 1))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.matrix.height(), Some(self.matrix.height()))
    }
}

impl<'a> ExactSizeIterator for MatrixLineIterator<'a> {}

impl Matrix for Mat {
    fn width(&self) -> usize {
        self.cols() as usize
    }

    fn height(&self) -> usize {
        self.rows() as usize
    }

    fn as_ptr(&self) -> *const u8 {
        self.data().unwrap() as *const u8
    }

    fn line(&self, n: usize) -> &[u8] {
        let cols = self.cols() as usize;
        &self.data_typed::<u8>().unwrap()[n * cols..(n + 1) * cols]
    }
}

#[derive(Debug)]
pub struct Matrix2D {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

impl Matrix2D {
    pub fn new(width: usize) -> Self {
        Self {
            width,
            height: 0,
            data: vec![],
        }
    }

    pub fn push(&mut self, v: &[u8]) {
        assert_eq!(self.width, v.len());
        self.height += 1;
        self.data.extend_from_slice(v);
    }

    pub fn clear(&mut self) {
        self.height = 0;
        self.data.clear();
    }
}

impl Matrix for Matrix2D {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    fn line(&self, n: usize) -> &[u8] {
        &self.data[n * self.width..(n + 1) * self.width]
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;
    use opencv::prelude::*;

    #[test]
    fn mat_into_iter() {
        let mat = Mat::from_slice_2d(&[[1u8, 2], [3, 4]]).unwrap();
        let mut iter = mat.iter_lines();

        assert_eq!(iter.len(), 2);
        assert_eq!(iter.next(), Some(&[1u8, 2][..]));
        assert_eq!(iter.next(), Some(&[3u8, 4][..]));
        assert_eq!(iter.next(), None);
    }
}
