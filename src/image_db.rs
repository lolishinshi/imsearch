use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use crate::config::OPTS;
use crate::slam3_orb::Slam3ORB;
use crate::utils;
use anyhow::Result;
use opencv::features2d;
use opencv::prelude::*;
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;

type PooledSqlite = PooledConnection<SqliteConnectionManager>;

thread_local! {
    static ORB: RefCell<Slam3ORB> = RefCell::new(Slam3ORB::from(&*OPTS));
    static FLANN: RefCell<features2d::FlannBasedMatcher> = RefCell::new(features2d::FlannBasedMatcher::from(&*OPTS));
}

pub struct ImageDb {
    pool: Pool<SqliteConnectionManager>,
}

impl ImageDb {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let manager = SqliteConnectionManager::file(path);
        let pool = Pool::new(manager)?;

        let conn = pool.get()?;
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
        conn.pragma_update(
            rusqlite::DatabaseName::Main.into(),
            "journal_mode",
            &"TRUNCATE",
        )?;

        Ok(Self { pool })
    }

    fn search_image_id_by_path(conn: &PooledSqlite, path: &str) -> Result<i32> {
        Ok(conn.query_row(
            "SELECT id FROM images WHERE image = ?",
            params![path],
            |row| row.get::<usize, i32>(0),
        )?)
    }

    fn search_image_path_by_id(conn: &PooledSqlite, id: i32) -> Result<String> {
        Ok(conn.query_row(
            "SELECT image FROM images WHERE id = ?",
            params![id],
            |row| row.get::<usize, String>(0),
        )?)
    }

    fn search_image_id_by_des(conn: &PooledSqlite, des: &Mat) -> Result<i32> {
        let data = des.data_typed::<u8>()?;
        Ok(conn.query_row(
            "SELECT id FROM features WHERE feature = ?",
            params![data],
            |row| row.get::<usize, i32>(0),
        )?)
    }

    // TODO: parallel
    pub fn add<S: AsRef<str>>(&self, path: S) -> Result<()> {
        let mut conn = self.pool.get().unwrap();
        if Self::search_image_id_by_path(&conn, path.as_ref()).is_ok() {
            return Ok(());
        }

        let img = utils::imread(path.as_ref())?;
        let (_, des) = ORB.with(|f| utils::detect_and_compute(&mut *f.borrow_mut(), &img))?;

        conn.execute(
            "INSERT INTO images (image) VALUES (?1)",
            params![path.as_ref()],
        )?;
        let id = Self::search_image_id_by_path(&conn, path.as_ref())?;

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

    pub fn search<S: AsRef<str>>(&self, path: S) -> Result<Vec<(usize, String)>> {
        let img = utils::imread(path.as_ref())?;
        let (_, des) = ORB.with(|f| utils::detect_and_compute(&mut *f.borrow_mut(), &img))?;

        let conn = self.pool.get()?;

        let count = conn.query_row("SELECT COUNT(*) FROM features", [], |row| {
            row.get::<usize, usize>(0)
        })?;

        let mut results = HashMap::new();

        for i in (0..count).step_by(OPTS.batch_size) {
            let mut stmt =
                conn.prepare("SELECT feature FROM features LIMIT 500 * 1000 OFFSET ?")?;
            let mut rows = stmt.query(params![i])?;
            let mut old_des = vec![];
            while let Some(row) = rows.next()? {
                let data = row.get::<usize, Vec<u8>>(0)?;
                old_des.push(data);
            }
            let old_des = Mat::from_slice_2d(&old_des)?;

            let mut matches = opencv::types::VectorOfVectorOfDMatch::new();
            let mask = Mat::default();
            FLANN.with(|f| {
                f.borrow()
                    .knn_train_match(&des, &old_des, &mut matches, OPTS.knn_k, &mask, false)
            })?;

            for match_ in matches.iter() {
                for point in match_.iter() {
                    let des = old_des.row(point.train_idx)?;
                    let id = Self::search_image_id_by_des(&conn, &des)?;
                    *results.entry(id).or_insert(0) += 1;
                }
            }
        }

        let mut results = results
            .iter()
            .filter(|(_k, v)| **v > 2)
            .map(|(&k, &v)| Self::search_image_path_by_id(&conn, k).map(|k| (v, k)))
            .collect::<Result<Vec<_>>>()?;
        results.sort_unstable_by_key(|v| std::cmp::Reverse(v.0));

        Ok(results.into_iter().take(OPTS.output_count).collect())
    }
}
