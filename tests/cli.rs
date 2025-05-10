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

fn file_from_dir(dir: &str) -> DirEntry {
    WalkDir::new(dir)
        .into_iter()
        .filter(|x| x.as_ref().unwrap().file_type().is_file())
        .next()
        .unwrap()
        .unwrap()
}

#[rstest]
#[case("-b=500")]
#[case("--on-disk")]
fn build_and_search(#[case] arg: &str) -> Result<()> {
    let conf_dir = assert_fs::TempDir::new()?;

    fs::copy("tests/index.template", conf_dir.path().join("index.template"))?;

    cmd!("tar", "cf", conf_dir.path().join("part2.tar"), "tests/dataset/part2").success();

    cargo_run!("imsearch", "-c", conf_dir.path(), "add", "tests/dataset/part1").success();
    cargo_run!("imsearch", "-c", conf_dir.path(), "add", conf_dir.path().join("part2.tar"))
        .success();

    cargo_run!("imsearch", "-c", conf_dir.path(), "build", arg).success();

    let part1 = file_from_dir("tests/dataset/part1");
    let part2 = file_from_dir("tests/dataset/part2");

    cargo_run!("imsearch", "-c", conf_dir.path(), "search", part1.path())
        .stdout(predicate::str::contains(part1.path().to_str().unwrap()));
    cargo_run!("imsearch", "-c", conf_dir.path(), "search", part2.path())
        .stdout(predicate::str::contains(part2.path().to_str().unwrap()));

    Ok(())
}

#[rstest]
#[case("-b=500", "-b=500")]
#[case("--on-disk", "-b=500")]
#[case("-b=500", "--on-disk")]
#[case("--on-disk", "--on-disk")]
fn build_multi_times(#[case] arg1: &str, #[case] arg2: &str) -> Result<()> {
    let conf_dir = assert_fs::TempDir::new()?;

    fs::copy("tests/index.template", conf_dir.path().join("index.template"))?;

    cargo_run!("imsearch", "-c", conf_dir.path(), "add", "tests/dataset/part1").success();
    cargo_run!("imsearch", "-c", conf_dir.path(), "build", arg1).success();
    cargo_run!("imsearch", "-c", conf_dir.path(), "add", "tests/dataset/part2").success();
    cargo_run!("imsearch", "-c", conf_dir.path(), "build", arg2).success();

    let part1 = file_from_dir("tests/dataset/part1");
    let part2 = file_from_dir("tests/dataset/part2");

    cargo_run!("imsearch", "-c", conf_dir.path(), "search", part1.path())
        .stdout(predicate::str::contains(part1.path().to_str().unwrap()));
    cargo_run!("imsearch", "-c", conf_dir.path(), "search", part2.path())
        .stdout(predicate::str::contains(part2.path().to_str().unwrap()));

    Ok(())
}
