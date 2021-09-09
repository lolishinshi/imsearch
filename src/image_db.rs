use std::path::Path;

use anyhow::Result;
use opencv::features2d;
use opencv::prelude::*;
use opencv::{flann, imgcodecs};
use rusqlite::{params, Connection};
use r2d2_sqlite::SqliteConnectionManager;
use crate::slam3_orb::Slam3ORB;
use crate::utils;

pub struct ImageDb {
    pool: r2d2::Pool<SqliteConnectionManager>,
    orb: Slam3ORB,
    flann: features2d::FlannBasedMatcher,
}

impl ImageDb {
    pub fn new<P: AsRef<Path>>(path: P, orb: Slam3ORB, flann: features2d::FlannBasedMatcher) -> Result<Self> {
        let manager = SqliteConnectionManager::file(path);
        let pool = r2d2::Pool::new(manager)?;

        pool.get()?
            .execute_batch(
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

        Ok(Self { pool, orb, flann })
    }

    // TODO: parallel
    pub fn add<S: AsRef<str>>(&mut self, path: S) -> Result<()> {
        let img = utils::imread(path.as_ref())?;
        let (_, des) = utils::detect_and_compute(&mut self.orb, &img)?;

        let mut conn = self.pool.get().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO images (image) VALUES (?1)",
            params![path.as_ref()],
        )?;
        let id = conn.query_row(
            "SELECT id FROM images WHERE image = ?1",
            params![path.as_ref()],
            |row| row.get::<usize, i32>(0),
        )?;

        let tx = conn.transaction()?;
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
        let img = utils::imread(path.as_ref())?;
        let (_, des) = utils::detect_and_compute(&mut self.orb, &img)?;

        let conn = self.pool.get()?;

        let count = conn
            .query_row("SELECT COUNT(*) FROM features", [], |row| {
                row.get::<usize, usize>(0)
            })?;
        for i in (0..count).step_by(500 * 1000) {
            let mut stmt = conn
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
            self.flann.knn_train_match(&des, &old_des, &mut matches, 2, &mask, false)?;

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
