mod slam3_orb;
mod slam2_orb;

use std::path::Path;

use anyhow::Result;
use opencv::features2d;
use opencv::prelude::*;
use opencv::{flann, imgcodecs};
use rusqlite::{params, Connection};

pub struct ImageDb {
    conn: Connection,
    orb: opencv::types::PtrOfORB,
}

impl ImageDb {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path)?;
        let orb = <dyn features2d::ORB>::default()?;

        conn.execute_batch(
            "BEGIN;
             CREATE TABLE IF NOT EXISTS features (
               feature BLOB PRIMARY KEY,
               id      INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS images (
               id    INTEGER PRIMARY KEY AUTOINCREMENT,
               image TEXT UNIQUE NOT NULL
             );
             COMMIT;",
        )?;

        Ok(Self { conn, orb })
    }

    fn detect_and_compute(&mut self, img: &Mat) -> Result<Mat> {
        let mask = Mat::default();
        let mut kps = opencv::types::VectorOfKeyPoint::new();
        let mut des = Mat::default();

        self.orb
            .detect_and_compute(&img, &mask, &mut kps, &mut des, false);

        Ok(des)
    }

    pub fn add<S: AsRef<str>>(&mut self, path: S) -> Result<()> {
        let img = imgcodecs::imread(path.as_ref(), imgcodecs::IMREAD_GRAYSCALE)?;
        let des = self.detect_and_compute(&img)?;

        self.conn.execute(
            "INSERT OR IGNORE INTO images (image) VALUES (?1)",
            params![path.as_ref()],
        )?;
        let id = self.conn.query_row(
            "SELECT id FROM images WHERE image = ?1",
            params![path.as_ref()],
            |row| row.get::<usize, i32>(0),
        )?;

        let tx = self.conn.transaction()?;
        for i in 0..des.rows() {
            let row = des.row(i)?;
            let data = row.data_typed::<u8>()?;
            tx.execute(
                "INSERT OR IGNORE INTO features (feature, id) VALUES (?1, ?2)",
                params![data, id],
            )?;
        }
        tx.commit()?;

        Ok(())
    }

    pub fn search<S: AsRef<str>>(&mut self, path: S) -> Result<Vec<(usize, String)>> {
        let img = imgcodecs::imread(path.as_ref(), imgcodecs::IMREAD_GRAYSCALE)?;
        let des = self.detect_and_compute(&img)?;

        let index_params = opencv::core::Ptr::new(flann::IndexParams::from(
            flann::LshIndexParams::new(6, 12, 1).unwrap(),
        ));
        let search_params =
            opencv::core::Ptr::new(flann::SearchParams::new_1(32, 0.0, true).unwrap());
        let mut flann = features2d::FlannBasedMatcher::new(&index_params, &search_params).unwrap();

        let count = self
            .conn
            .query_row("SELECT COUNT(*) FROM features", [], |row| {
                row.get::<usize, usize>(0)
            })?;
        for i in (0..count).step_by(500 * 1000) {
            let mut stmt = self
                .conn
                .prepare("SELECT feature FROM features LIMIT 500 * 1000 OFFSET ?")?;
            let mut rows = stmt.query(params![i])?;
            let mut old_des = vec![];
            while let Some(row) = rows.next()? {
                let data = row.get::<usize, Vec<u8>>(0)?;
                old_des.push(data);
            }
            let old_des = Mat::from_slice_2d(&old_des)?;

            let mut matches = opencv::types::VectorOfVectorOfDMatch::new();
            let mask = Mat::default();
            flann.knn_train_match(&des, &old_des, &mut matches, 2, &mask, false)?;

            for match_ in matches.iter() {
                let m = match_.get(0).unwrap();
                let n = match_.get(1).unwrap();
                if m.distance < 0.7 * n.distance {
                    todo!()
                }
            }
            println!("{}", matches.len());
        }

        Ok(vec![(0, "-".to_owned())])
    }
}

/*
   let index_params = Ptr::new(flann::IndexParams::from(
       flann::LshIndexParams::new(6, 12, 1).unwrap(),
   ));
   let search_params = Ptr::new(flann::SearchParams::new_1(32, 0.0, true).unwrap());
   let mut flann = features2d::FlannBasedMatcher::new(&index_params, &search_params).unwrap();

   let mut matches = opencv::types::VectorOfVectorOfDMatch::new();
   flann
       .knn_train_match(&des1, &des2, &mut matches, 2, &mask, false)
       .unwrap();
*/
