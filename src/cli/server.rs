use clap::Parser;
use log::{error, info};
use prometheus::{BasicAuthentication, labels};
use rand::distr::{Alphanumeric, SampleString};
use tokio::net::TcpListener;
use tokio::task::spawn_blocking;
use tokio::time::{Duration, sleep};

use crate::cli::SubCommandExtend;
use crate::config::{OrbOptions, SearchOptions};
use crate::{IMDBBuilder, Opts, server};

#[derive(Parser, Debug, Clone)]
pub struct ServerCommand {
    #[command(flatten)]
    pub orb: OrbOptions,
    #[command(flatten)]
    pub search: SearchOptions,
    /// 监听地址
    #[arg(long, default_value = "127.0.0.1:8000")]
    pub addr: String,
    /// 请求验证 token，不填则随机生成
    #[arg(long, default_value_t = String::new())]
    pub token: String,
    /// prometheus 主动推送地址
    #[arg(long, value_name = "URL")]
    pub prometheus_push: Option<String>,
    /// 自定义 instance 标签值
    #[arg(long, value_name = "NAME")]
    pub prometheus_instance: Option<String>,
    /// prometheus 认证信息，格式为 username:password
    #[arg(long, value_name = "AUTH")]
    pub prometheus_auth: Option<String>,
}

impl SubCommandExtend for ServerCommand {
    async fn run(&self, opts: &Opts) -> anyhow::Result<()> {
        let db = IMDBBuilder::new(opts.conf_dir.clone())
            .cache(true)
            .score_type(self.search.score_type)
            .open()
            .await?;

        let index = db.get_index(!self.search.no_mmap)?;

        let mut self_clone = self.clone();
        if self_clone.token.is_empty() {
            self_clone.token = Alphanumeric.sample_string(&mut rand::rng(), 32);
            info!("鉴权 token: {}", self_clone.token);
        }

        // 创建应用状态
        let state = server::AppState::new(index, db, self_clone);

        // 创建应用
        let app = server::create_app(state);

        if let Some(url) = self.prometheus_push.clone() {
            let instance = self.prometheus_instance.clone().unwrap_or_else(|| self.addr.clone());
            let auth = self.prometheus_auth.clone().map(|s| {
                let (username, password) = s.split_once(':').unwrap();
                (username.to_string(), password.to_string())
            });
            tokio::spawn(async move {
                loop {
                    let metric_families = prometheus::gather();
                    let url = url.clone();
                    let instance = instance.clone();
                    let auth = auth.clone();
                    let r = spawn_blocking(move || {
                        prometheus::push_metrics(
                            "imsearch",
                            labels! {
                                "instance".to_string() => instance.clone(),
                            },
                            &url,
                            metric_families,
                            auth.map(|(username, password)| BasicAuthentication {
                                username,
                                password,
                            }),
                        )
                    })
                    .await
                    .unwrap();
                    if let Err(e) = r {
                        error!("推送指标失败: {e}");
                    }
                    sleep(Duration::from_secs(30)).await;
                }
            });
        }

        // 启动服务器
        info!("服务器启动：http://{}", &self.addr);
        let listener = TcpListener::bind(&self.addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}
