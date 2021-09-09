fn main() {
    println!("cargo:rerun-if-changed=src/ORB_SLAM2");
    println!("cargo:rerun-if-changed=src/ORB_SLAM3");
    let library = pkg_config::probe_library("opencv4").unwrap();
    cc::Build::new()
        .file("src/ORB_SLAM3/ORBextractor.cc")
        .file("src/ORB_SLAM3/ORBwrapper.cc")
        .includes(&library.include_paths)
        .compile("ORBextractor3");
    cc::Build::new()
        .file("src/ORB_SLAM2/ORBextractor.cc")
        .file("src/ORB_SLAM2/ORBwrapper.cc")
        .includes(&library.include_paths)
        .compile("ORBextractor2");
}
