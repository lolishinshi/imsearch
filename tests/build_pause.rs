use std::fs;
use std::process::Command;

use anyhow::Result;
use assert_cmd::prelude::*;
use imsearch::IMDBBuilder;
use imsearch::db::{crud, init_db};
use imsearch::ivf::{HnswQuantizer, IvfHnsw, Quantizer};

#[tokio::test]
async fn pause_clean_and_resume_preserves_search_and_reclaims_database() -> Result<()> {
    // 分别测试自动清理和手动 clean。分成两批，让恢复时走合并流程。
    for automatic_clean in [true, false] {
        let dir = tempfile::tempdir()?;
        let db = IMDBBuilder::new(dir.path()).wal(false).open().await?;
        db.add_image("first.jpg", &[1; 32], &vec![[0; 32]; 4096]).await?;
        db.add_image("second.jpg", &[2; 32], &vec![[255; 32]; 4096]).await?;
        drop(db);
        HnswQuantizer::init(&[[0; 32], [255; 32]])?.save(dir.path().join("quantizer.bin"))?;

        let mut pause = Command::cargo_bin("imsearch")?;
        pause.arg("-c").arg(dir.path()).args(["build", "-b", "1", "--stop-before-merge"]);
        if automatic_clean {
            // 两个参数一起用时，暂停后应保留特征。
            pause.arg("--clean-before-merge");
        }
        pause.assert().success();
        assert!(dir.path().join("invlists.1").is_file());
        assert!(dir.path().join("invlists.2").is_file());
        assert!(!dir.path().join("invlists.bin").exists());
        let pool = init_db(dir.path(), false).await?;
        assert_eq!(crud::get_vectors(&pool, 10, 0).await?.len(), 2);
        assert_eq!(crud::count_image_unindexed(&pool).await?, 0);
        let metadata: Vec<(i64, Vec<u8>, String, i64, i64)> = sqlx::query_as(
            "SELECT image.id, hash, path, vector_count, total_vector_count \
             FROM image JOIN vector_stats ON image.id = vector_stats.id ORDER BY image.id",
        )
        .fetch_all(&pool)
        .await?;
        pool.close().await;
        let original_size = fs::metadata(dir.path().join("imsearch.db"))?.len();

        if !automatic_clean {
            Command::cargo_bin("imsearch")?
                .arg("-c")
                .arg(dir.path())
                .args(["clean", "--force"])
                .assert()
                .success();
        }
        let mut resume = Command::cargo_bin("imsearch")?;
        resume.arg("-c").arg(dir.path()).arg("build");
        if automatic_clean {
            resume.arg("--clean-before-merge");
        }
        resume.assert().success();

        assert!(dir.path().join("invlists.bin").is_file());
        assert!(!dir.path().join("invlists.1").exists());
        assert!(!dir.path().join("invlists.2").exists());
        let pool = init_db(dir.path(), false).await?;
        assert!(crud::get_vectors(&pool, 10, 0).await?.is_empty());
        let retained: Vec<(i64, Vec<u8>, String, i64, i64)> = sqlx::query_as(
            "SELECT image.id, hash, path, vector_count, total_vector_count \
             FROM image JOIN vector_stats ON image.id = vector_stats.id ORDER BY image.id",
        )
        .fetch_all(&pool)
        .await?;
        assert_eq!(retained, metadata);
        assert!(fs::metadata(dir.path().join("imsearch.db"))?.len() < original_size);

        // 清理后仍能从索引查到特征对应的图片。
        let index = IvfHnsw::open_disk(dir.path(), 2)?;
        for (feature, expected) in [([0; 32], "first.jpg"), ([255; 32], "second.jpg")] {
            let result = index.search(&[feature], 1, 1)?;
            assert_eq!(result.neighbors.len(), 1);
            let neighbor = &result.neighbors[0];
            assert_eq!(neighbor.distance, 0);
            let id = crud::get_image_id_by_vector_id(&pool, neighbor.id as i64).await?;
            assert_eq!(crud::get_image_path(&pool, id).await?, expected);
        }
        pool.close().await;
    }
    Ok(())
}

#[tokio::test]
async fn database_read_failure_does_not_clean_or_merge_pending_segments() -> Result<()> {
    // 数据库读取失败时，应保留特征和临时索引。
    let dir = tempfile::tempdir()?;
    let db = IMDBBuilder::new(dir.path()).wal(false).open().await?;
    db.add_image("first.jpg", &[1; 32], &[[0; 32]]).await?;
    drop(db);
    HnswQuantizer::init(&[[0; 32]])?.save(dir.path().join("quantizer.bin"))?;
    Command::cargo_bin("imsearch")?
        .arg("-c")
        .arg(dir.path())
        .args(["build", "--stop-before-merge"])
        .assert()
        .success();
    let pool = init_db(dir.path(), false).await?;
    // 改列名让 SELECT 失败，但不影响 DELETE，用来检查是否误删了特征。
    sqlx::query("ALTER TABLE vector RENAME COLUMN vector TO unreadable_vector")
        .execute(&pool)
        .await?;
    pool.close().await;

    Command::cargo_bin("imsearch")?
        .arg("-c")
        .arg(dir.path())
        .args(["build", "--clean-before-merge"])
        .assert()
        .failure();
    assert!(dir.path().join("invlists.1").is_file());
    assert!(!dir.path().join("invlists.bin").exists());
    let pool = init_db(dir.path(), false).await?;
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM vector").fetch_one(&pool).await?;
    assert_eq!(count, 1);
    pool.close().await;
    Ok(())
}
