use sqlx::{Executor, Result, Sqlite, SqlitePool};

use super::VectorIdxRecord;

/// 添加图片记录
pub async fn add_image<'c, E>(executor: E, hash: &[u8], path: &str) -> Result<i64>
where
    E: Executor<'c, Database = Sqlite>,
{
    let result = sqlx::query!(
        r#"
        INSERT INTO image (hash, path)
        VALUES (?, ?)
        RETURNING id
        "#,
        hash,
        path,
    )
    .fetch_one(executor)
    .await?;

    Ok(result.id)
}

/// 检查图片哈希是否存在
pub async fn check_image_hash(executor: &SqlitePool, hash: &[u8]) -> Result<bool> {
    let result = sqlx::query!(
        r#"
        SELECT COUNT(*) as count FROM image WHERE hash = ?
        "#,
        hash
    )
    .fetch_one(executor)
    .await?;

    Ok(result.count > 0)
}

pub async fn get_image_path(executor: &SqlitePool, id: i64) -> Result<String> {
    let result = sqlx::query!(
        r#"
        SELECT path FROM image WHERE id = ?
        "#,
        id
    )
    .fetch_one(executor)
    .await?;

    Ok(result.path)
}

/// 批量设置图片为已索引
pub async fn set_indexed_batch(executor: &SqlitePool, ids: &[i64]) -> Result<()> {
    let mut tx = executor.begin().await?;
    for id in ids {
        sqlx::query!(
            r#"
            UPDATE vector_stats SET indexed = 1 WHERE id = ?
            "#,
            id
        )
        .execute(&mut *tx)
        .await?;
    }
    tx.commit().await?;
    Ok(())
}

/// 根据向量 ID 获取图片 ID
pub async fn get_image_id_by_vector_id(executor: &SqlitePool, id: i64) -> Result<i64> {
    let result = sqlx::query!(
        r#"
        SELECT id as id FROM vector_stats
        WHERE total_vector_count >= ? ORDER BY total_vector_count ASC LIMIT 1
        "#,
        id
    )
    .fetch_one(executor)
    .await?;

    Ok(result.id.unwrap())
}

/// 添加向量
pub async fn add_vector<'c, E>(executor: E, id: i64, vector: &[u8]) -> Result<()>
where
    E: Executor<'c, Database = Sqlite>,
{
    sqlx::query!(
        r#"
        INSERT INTO vector (id, vector)
        VALUES (?, ?)
        "#,
        id,
        vector
    )
    .execute(executor)
    .await?;

    Ok(())
}

/// 添加向量统计信息
pub async fn add_vector_stats<'c, E>(executor: E, id: i64, vector_count: i64) -> Result<()>
where
    E: Executor<'c, Database = Sqlite>,
{
    let last_id = id - 1;
    sqlx::query!(
        r#"
        INSERT INTO vector_stats (id, vector_count, total_vector_count)
        SELECT
            ? as id,
            ? as vector_count,
            COALESCE(
                (SELECT total_vector_count FROM vector_stats WHERE id = ?),
                0
            ) + ? as total_vector_count
        "#,
        id,
        vector_count,
        last_id,
        vector_count
    )
    .execute(executor)
    .await?;

    Ok(())
}

pub async fn count_image_unindexed(executor: &SqlitePool) -> Result<u64> {
    let result = sqlx::query!(
        r#"
        SELECT COUNT(*) as count FROM vector_stats WHERE indexed = 0
        "#,
    )
    .fetch_one(executor)
    .await?;

    Ok(result.count as u64)
}

pub async fn get_vectors(
    executor: &SqlitePool,
    limit: usize,
    offset: usize,
) -> Result<Vec<VectorIdxRecord>> {
    let limit = limit as i64;
    let offset = offset as i64;
    let rows = sqlx::query_as!(
        VectorIdxRecord,
        r#"
        SELECT vector.id as id, vector, total_vector_count
        FROM vector
        JOIN vector_stats ON vector.id = vector_stats.id
        LIMIT ? OFFSET ?
        "#,
        limit,
        offset
    )
    .fetch_all(executor)
    .await?;

    Ok(rows)
}

/// 获取未索引的向量列表
pub async fn get_vectors_unindexed(
    executor: &SqlitePool,
    limit: usize,
    offset: usize,
) -> Result<Vec<VectorIdxRecord>> {
    let limit = limit as i64;
    let offset = offset as i64;
    let rows = sqlx::query_as!(
        VectorIdxRecord,
        r#"
        SELECT vector.id as id, vector, total_vector_count
        FROM vector
        JOIN vector_stats ON vector.id = vector_stats.id
        WHERE vector_stats.indexed = 0
        LIMIT ? OFFSET ?
        "#,
        limit,
        offset
    )
    .fetch_all(executor)
    .await?;

    Ok(rows)
}

/// 删除向量列表
pub async fn delete_vectors(executor: &SqlitePool) -> Result<()> {
    sqlx::query!(
        r#"
        DELETE FROM vector WHERE id IN (
            SELECT vector.id FROM vector JOIN vector_stats ON vector.id = vector_stats.id WHERE vector_stats.indexed = 1
        )
        "#
    )
    .execute(executor)
    .await?;
    sqlx::query!("VACUUM").execute(executor).await?;
    Ok(())
}

/// 删除所有向量列表
pub async fn delete_vectors_all(executor: &SqlitePool) -> Result<()> {
    sqlx::query!(r#"DELETE FROM vector"#,).execute(executor).await?;
    sqlx::query!("VACUUM").execute(executor).await?;
    Ok(())
}

/// 查询数据库中的图片和向量数量
pub async fn get_count(executor: &SqlitePool) -> Result<(i64, i64)> {
    let result = sqlx::query!(
        r#"
        SELECT id, total_vector_count FROM vector_stats ORDER BY id DESC LIMIT 1;
        "#,
    )
    .fetch_one(executor)
    .await?;

    Ok((result.id, result.total_vector_count))
}

/// 获取所有 total_vector_count 记录
///
/// NOTE: 此处假定了 total_vector_count 一定是连续的，中间没有缺失记录
/// 正常情况下这个条件是满足的，因为目前没有删除图片的操作
pub async fn get_all_total_vector_count(executor: &SqlitePool) -> Result<Vec<i64>> {
    let result = sqlx::query!(
        r#"
        SELECT total_vector_count FROM vector_stats ORDER BY id ASC;
        "#,
    )
    .fetch_all(executor)
    .await?;

    Ok(result.into_iter().map(|row| row.total_vector_count).collect())
}
