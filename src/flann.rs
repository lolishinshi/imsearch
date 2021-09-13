use anyhow::Result;
use itertools::IntoChunks;
use opencv::prelude::*;

pub struct Flann<'a>(flann::SliceIndex<'a, u8>);

impl<'a> Flann<'a> {
    pub fn new(
        points: &'a Mat,
        table_number: u32,
        key_size: u32,
        multi_probe_level: u32,
        checks: i32,
    ) -> Result<Self> {
        let checks = match checks {
            0 => flann::Checks::Autotuned,
            -1 => flann::Checks::Unlimited,
            _ => flann::Checks::Exact(checks),
        };
        let params = flann::Parameters {
            algorithm: flann::Algorithm::Lsh,
            table_number,
            key_size,
            multi_probe_level,
            checks,
            ..Default::default()
        };
        let point_len = points.cols() as usize;
        let points = points.data_typed::<u8>()?;
        Ok(Self(flann::SliceIndex::new(point_len, points, params)?))
    }

    pub fn add(&mut self, points: &'a Mat) -> Result<()> {
        let points = points.data_typed::<u8>()?;
        Ok(self.0.add_many_slices(points)?)
    }

    pub fn knn_search(
        &mut self,
        points: &Mat,
        k: usize,
    ) -> Result<IntoChunks<impl Iterator<Item = flann::Neighbor<f32>>>> {
        let points = points.data_typed::<u8>()?;
        Ok(self.0.find_many_nearest_neighbors_flat(k, points)?)
    }
}
