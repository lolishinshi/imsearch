use std::path::Path;

use log::info;
use sqlx::{SqlitePool, sqlite::*};

pub mod crud;
pub mod model;

pub use model::*;

pub type Database = SqlitePool;

pub async fn init_db(filename: impl AsRef<Path>) -> Result<Database, sqlx::Error> {
    let filename = filename.as_ref();
    info!("初始化数据库连接: {}", filename.display());

    let options = SqliteConnectOptions::new()
        .journal_mode(SqliteJournalMode::Wal)
        .synchronous(SqliteSynchronous::Normal)
        .filename(filename)
        .create_if_missing(true);

    let pool = SqlitePool::connect_with(options).await?;

    info!("检查数据库迁移");
    sqlx::migrate!().run(&pool).await?;

    Ok(pool)
}
