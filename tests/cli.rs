use std::fs;
use std::process::Command;

use anyhow::Result;
use assert_cmd::prelude::*;
use predicates::prelude::*;
use rstest::*;
use walkdir::{DirEntry, WalkDir};

macro_rules! cargo_run {
    ($cmd:expr, $($args:expr),*) => {
        {
            let mut cmd = Command::cargo_bin($cmd)?;
            $(cmd.arg($args);)*
            cmd.assert()
        }
    };
}

macro_rules! cmd {
    ($cmd:expr, $($args:expr),*) => {{
        {
            let mut cmd = Command::new($cmd);
            $(cmd.arg($args);)*
            cmd.assert()
        }
    }};
}

fn file_from_dir(dir: &str) -> Result<DirEntry> {
    Ok(WalkDir::new(dir)
        .into_iter()
        .filter(|x| x.as_ref().unwrap().file_type().is_file())
        .next()
        .unwrap()?)
}

#[test]
fn add_tar() -> Result<()> {
    let conf_dir = assert_fs::TempDir::new()?;
    let tar_path = conf_dir.path().join("part1.tar");

    fs::copy("tests/quantizer.bin", conf_dir.path().join("quantizer.bin"))?;

    cmd!("tar", "cf", &tar_path, "tests/dataset/part1").success();

    cargo_run!("imsearch", "-c", conf_dir.path(), "add", tar_path).success();
    cargo_run!("imsearch", "-c", conf_dir.path(), "build").success();

    let part1 = file_from_dir("tests/dataset/part1")?;

    cargo_run!("imsearch", "-c", conf_dir.path(), "search", part1.path())
        .stdout(predicate::str::contains(part1.path().to_str().unwrap()));

    Ok(())
}

#[rstest]
#[case::ondisk("-b=500")]
fn build_1_round(#[case] arg: &str) -> Result<()> {
    let conf_dir = assert_fs::TempDir::new()?;

    fs::copy("tests/quantizer.bin", conf_dir.path().join("quantizer.bin"))?;

    cargo_run!("imsearch", "-c", conf_dir.path(), "add", "tests/dataset/part1").success();
    cargo_run!("imsearch", "-c", conf_dir.path(), "build", arg).success();

    let part1 = file_from_dir("tests/dataset/part1")?;

    cargo_run!("imsearch", "-c", conf_dir.path(), "search", part1.path())
        .stdout(predicate::str::contains(part1.path().to_str().unwrap()));

    Ok(())
}

#[rstest]
#[case::ondiskx2("-b=500", "-b=500")]
fn build_2_round(#[case] arg1: &str, #[case] arg2: &str) -> Result<()> {
    let conf_dir = assert_fs::TempDir::new()?;

    fs::copy("tests/quantizer.bin", conf_dir.path().join("quantizer.bin"))?;

    cargo_run!("imsearch", "-c", conf_dir.path(), "add", "tests/dataset/part1").success();
    cargo_run!("imsearch", "-c", conf_dir.path(), "build", arg1).success();
    cargo_run!("imsearch", "-c", conf_dir.path(), "add", "tests/dataset/part2").success();
    cargo_run!("imsearch", "-c", conf_dir.path(), "build", arg2).success();

    let part1 = file_from_dir("tests/dataset/part1")?;
    let part2 = file_from_dir("tests/dataset/part2")?;

    cargo_run!("imsearch", "-c", conf_dir.path(), "search", part1.path())
        .stdout(predicate::str::contains(part1.path().to_str().unwrap()));
    cargo_run!("imsearch", "-c", conf_dir.path(), "search", part2.path())
        .stdout(predicate::str::contains(part2.path().to_str().unwrap()));

    Ok(())
}
