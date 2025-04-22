use crate::cmd::SubCommandExtend;
use crate::utils;
use crate::{Opts, Slam3ORB, IMDB};
use log::info;
use opencv::imgcodecs;
use opencv::prelude::*;
use rouille::{post_input, router, try_or_400, Response};
use serde_json::json;
use std::sync::RwLock;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug, Clone)]
pub struct StartServer {
    /// Listen address
    #[structopt(long, default_value = "127.0.0.1:8000")]
    pub addr: String,
}

impl SubCommandExtend for StartServer {
    fn run(&self, opts: &Opts) -> anyhow::Result<()> {
        let db = IMDB::new(opts.conf_dir.clone(), true)?;

        let mut index = db.get_index(opts.mmap);
        index.set_nprobe(opts.nprobe);

        let index = RwLock::new(index);
        let opts = opts.clone();

        info!("starting server at http://{}", &self.addr);
        rouille::start_server(&self.addr, move |request| {
            let opts = opts.clone();
            router!(request,
                (POST) (/search) => {
                    let data = try_or_400!(post_input!(request, {
                        file: rouille::input::post::BufferedFile,
                        orb_scale_factor: Option<f32>,
                    }));
                    let mat = try_or_400!(Mat::from_slice(&data.file.data));
                    let img = try_or_400!(imgcodecs::imdecode(&mat, imgcodecs::IMREAD_GRAYSCALE));

                    info!("searching {:?}", data.file.filename);

                    let mut opts = opts.clone();
                    if let Some(orb_scale_factor) = data.orb_scale_factor {
                        opts.orb_scale_factor = orb_scale_factor;
                    }
                    let mut orb = Slam3ORB::from(&opts);

                    let start = Instant::now();
                    let result = utils::detect_and_compute(&mut orb, &img)
                            .and_then(|(_, descriptors)| {
                            let index = index.read().expect("failed to acquire rw lock");
                            db.search_des(&*index, descriptors, opts.knn_k, opts.distance)
                        });
                    let elapsed = start.elapsed().as_secs_f32();

                    match result {
                        Ok(mut result) => {
                            result.truncate(opts.output_count);
                            Response::json(&json!({
                                "time": elapsed,
                                "result": result,
                            }))
                        },
                        Err(err) => {
                            // TODO: 此处错误处理很简陋
                            Response::json(&err.to_string()).with_status_code(400)
                        },
                    }
                },
                (POST) (/set_nprobe) => {
                    let data = try_or_400!(post_input!(request, {
                        n: usize,
                    }));
                    try_or_400!(index.write()).set_nprobe(data.n);
                    Response::text("").with_status_code(200)
                },
                _ => {
                    Response::html(r#"
                <p>
                http --form http://127.0.0.1/search file@test.jpg orb_scale_factor=1.2</br>
                http --form http://127.0.0.1/set_nprobe n=128
                </p>
                "#).with_status_code(404)
                }
            )
        });
    }
}
