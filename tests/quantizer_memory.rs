#![cfg(target_os = "linux")]

use std::fs;
use std::process::Command;

use anyhow::{Context, Result};
use imsearch::ivf::{HnswQuantizer, Quantizer};

#[test]
fn repeated_quantizer_loads_release_memory() -> Result<()> {
    // 在独立进程中测量，避免其他并行测试的分配影响 RSS。
    if std::env::var_os("IMSEARCH_QUANTIZER_MEMORY_CHILD").is_none() {
        let output = Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "repeated_quantizer_loads_release_memory",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("IMSEARCH_QUANTIZER_MEMORY_CHILD", "1")
            .env("OMP_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        println!("{stdout}");
        assert!(output.status.success(), "内存回归测试失败：\n{stdout}\n{stderr}");
        return Ok(());
    }

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("quantizer.bin");
    let mut state = 0x1234_5678_9abc_def0_u64;
    let centers: Vec<[u8; 32]> = (0..8192)
        .map(|_| {
            std::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
        })
        .collect();
    HnswQuantizer::init(&centers)?.save(&path)?;

    // 预热文件读取和分配器，之后重复模拟 build 的每批加载、销毁。
    for _ in 0..3 {
        drop(HnswQuantizer::<32>::open(&path)?);
    }
    let before = resident_bytes()?;
    for _ in 0..24 {
        let quantizer = HnswQuantizer::<32>::open(&path)?;
        assert_eq!(quantizer.nlist(), centers.len());
        drop(quantizer);
    }
    let after = resident_bytes()?;
    let growth = after.saturating_sub(before);
    // 允许保留数份索引大小的分配空间，但不能随加载次数持续累积。
    let tolerance = fs::metadata(&path)?.len() * 4 + 8 * 1024 * 1024;
    println!("RSS before={before}, after={after}, growth={growth}, tolerance={tolerance} bytes");
    assert!(growth < tolerance, "重复加载量化器后 RSS 增长 {growth} 字节，超过 {tolerance} 字节");
    Ok(())
}

// VmRSS 反映当前驻留内存，而不是不会回落的历史峰值 VmHWM。
fn resident_bytes() -> Result<u64> {
    let status = fs::read_to_string("/proc/self/status")?;
    let rss = status.lines().find(|line| line.starts_with("VmRSS:")).context("缺少 VmRSS")?;
    Ok(rss.split_whitespace().nth(1).context("缺少 VmRSS 数值")?.parse::<u64>()? * 1024)
}
