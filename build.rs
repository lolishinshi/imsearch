#[path = "build/cmake_probe.rs"]
mod cmake_probe;
#[path = "build/library.rs"]
mod library;

use std::env;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use once_cell::sync::Lazy;
use semver::Version;

use library::Library;

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

static OUT_DIR: Lazy<PathBuf> =
    Lazy::new(|| PathBuf::from(env::var_os("OUT_DIR").expect("Can't read OUT_DIR env var")));
static MANIFEST_DIR: Lazy<PathBuf> = Lazy::new(|| {
    PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Can't read CARGO_MANIFEST_DIR env var"))
});

fn cleanup_lib_filename(filename: &OsStr) -> Option<&OsStr> {
    let mut strip_performed = false;
    let mut filename_path = Path::new(filename);
    // used to check for the file extension (with dots stripped) and for the part of the filename
    const LIB_EXTS: [&str; 7] = [
        ".so.",
        ".a.",
        ".dll.",
        ".lib.",
        ".dylib.",
        ".framework.",
        ".tbd.",
    ];
    if let (Some(stem), Some(extension)) = (
        filename_path.file_stem(),
        filename_path.extension().and_then(OsStr::to_str),
    ) {
        if LIB_EXTS
            .iter()
            .any(|e| e.trim_matches('.').eq_ignore_ascii_case(extension))
        {
            filename_path = Path::new(stem);
            strip_performed = true;
        }
    }

    if let Some(mut file) = filename_path.file_name().and_then(OsStr::to_str) {
        let orig_len = file.len();
        file = file.strip_prefix("lib").unwrap_or(file);
        LIB_EXTS.iter().for_each(|&inner_ext| {
            if let Some(inner_ext_idx) = file.find(inner_ext) {
                file = &file[..inner_ext_idx];
            }
        });
        if orig_len != file.len() {
            strip_performed = true;
            filename_path = Path::new(file);
        }
    }
    if strip_performed {
        Some(filename_path.as_os_str())
    } else {
        None
    }
}

fn get_version_header(header_dir: &Path) -> Option<PathBuf> {
    let out = header_dir.join("opencv2/core/version.hpp");
    if out.is_file() {
        Some(out)
    } else {
        let out = header_dir.join("Headers/core/version.hpp");
        if out.is_file() {
            Some(out)
        } else {
            None
        }
    }
}

fn get_version_from_headers(header_dir: &Path) -> Option<Version> {
    let version_hpp = get_version_header(header_dir)?;
    let mut major = None;
    let mut minor = None;
    let mut revision = None;
    let mut line = String::with_capacity(256);
    let mut reader = BufReader::new(File::open(version_hpp).ok()?);
    while let Ok(bytes_read) = reader.read_line(&mut line) {
        if bytes_read == 0 {
            break;
        }
        if let Some(line) = line.strip_prefix("#define CV_VERSION_") {
            let mut parts = line.split_whitespace();
            if let (Some(ver_spec), Some(version)) = (parts.next(), parts.next()) {
                match ver_spec {
                    "MAJOR" => {
                        major = Some(version.to_string());
                    }
                    "MINOR" => {
                        minor = Some(version.to_string());
                    }
                    "REVISION" => {
                        revision = Some(version.to_string());
                    }
                    _ => {}
                }
            }
            if major.is_some() && minor.is_some() && revision.is_some() {
                break;
            }
        }
        line.clear();
    }
    if let (Some(major), Some(minor), Some(revision)) = (major, minor, revision) {
        Some(Version::new(
            major.parse().ok()?,
            minor.parse().ok()?,
            revision.parse().ok()?,
        ))
    } else {
        Some(Version::new(0, 0, 0))
    }
}

fn main() {
    println!("cargo:rerun-if-changed=src/ORB_SLAM3");
    println!("cargo:rustc-link-lib=gomp");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=blas");
    println!("cargo:rustc-link-lib=lapack");

    let library = Library::probe().unwrap();
    cc::Build::new()
        .file("src/ORB_SLAM3/ORBextractor.cc")
        .file("src/ORB_SLAM3/ORBwrapper.cc")
        .includes(&library.include_paths)
        .flag("-Wno-unused")
        .define(
            "OCVRS_FFI_EXPORT_SUFFIX",
            // NOTE: 0.94.4 is the version of OpenCV
            &*format!("_{}", "0.94.4".replace(".", "_")),
        )
        .compile("ORBextractor3");
}
