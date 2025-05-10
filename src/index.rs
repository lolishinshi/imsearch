use std::env::set_current_dir;
use std::fs;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressIterator};
use log::{Level, info, log_enabled, warn};

use crate::config::ConfDir;
use crate::faiss::{FaissHStackInvLists, FaissIndex, FaissOnDiskInvLists};
use crate::utils::pb_style;

#[derive(Debug)]
pub struct IndexManager {
    conf_dir: ConfDir,
}

impl IndexManager {
    pub fn new(conf_dir: ConfDir) -> Self {
        if !conf_dir.path().exists() {
            fs::create_dir_all(conf_dir.path()).unwrap();
        }
        Self { conf_dir }
    }

    /// 获取模板索引
    pub fn get_template_index(&self) -> FaissIndex {
        let index_file = self.conf_dir.template_index();
        if index_file.exists() {
            FaissIndex::from_file(index_file, false)
        } else {
            panic!("模板索引不存在，请先训练索引，并保存为 {}", index_file.display());
        }
    }

    /// 获取主索引
    fn get_main_index(&self, mut mmap: bool) -> FaissIndex {
        let index_file = self.conf_dir.index();
        if !index_file.exists() {
            return self.get_template_index();
        }

        if self.conf_dir.ondisk_ivf().exists() {
            if mmap {
                mmap = false;
                warn!("OnDisk 倒排无法使用 mmap 模式加载");
            }
            // NOTE: 此处切换路径的原因是，faiss 会根据「当前路径」而不是 index 所在路径来加载倒排列表
            set_current_dir(self.conf_dir.path()).unwrap();
        }

        info!("正在加载索引 {}, mmap: {}", index_file.display(), mmap);
        let index = FaissIndex::from_file(index_file, mmap);
        info!("faiss 版本   : {}", index.faiss_version());
        info!("已添加特征点 : {}", index.ntotal());
        info!("倒排列表数量 : {}", index.nlist());
        info!("不平衡度     : {}", index.imbalance_factor());
        if log_enabled!(Level::Debug) {
            index.print_stats();
        }
        index
    }

    /// 获取子索引
    fn get_sub_index(&self, mmap: bool) -> impl Iterator<Item = FaissIndex> {
        self.conf_dir.all_sub_index().into_iter().map(move |file| FaissIndex::from_file(file, mmap))
    }

    /// 获取聚合索引
    pub fn get_aggregate_index(&self, mmap: bool) -> FaissIndex {
        if self.conf_dir.all_sub_index().is_empty() {
            self.get_main_index(mmap)
        } else {
            info!("正在添加子索引……");
            let mut template = self.get_template_index();

            let mut invfs = vec![];
            let mut ntotal = 0;
            for mut sub_index in self.get_sub_index(mmap) {
                let ivf = sub_index.invlists();
                invfs.push(ivf);
                sub_index.set_own_invlists(false);
                ntotal += sub_index.ntotal();
            }

            if self.conf_dir.index().exists() {
                let mut index = FaissIndex::from_file(self.conf_dir.index(), mmap);
                index.set_own_invlists(false);
                invfs.push(index.invlists());
            }

            let htack = FaissHStackInvLists::new(invfs);
            template.replace_invlists(htack, true);
            template.set_ntotal(ntotal as i64);

            info!("已添加特征点 : {}", template.ntotal());
            info!("倒排列表数量 : {}", template.nlist());
            info!("不平衡度     : {}", template.imbalance_factor());
            if log_enabled!(Level::Debug) {
                template.print_stats();
            }

            template
        }
    }

    /// 在内存中合并所有子索引
    pub fn merge_index_on_memory(&self) -> Result<()> {
        info!("在内存中合并所有索引……");
        let mut index = self.get_main_index(false);

        // 注意在内存中合并时，子索引不能用 mmap 模式加载
        let sub_index_files = self.conf_dir.all_sub_index();
        let pb = ProgressBar::new(sub_index_files.len() as u64).with_style(pb_style());
        for sub_index in self.get_sub_index(false).progress_with(pb) {
            index.merge_from(&sub_index, 0);
        }

        info!("保存索引……");
        index.write_file(self.conf_dir.index());

        for file in sub_index_files {
            fs::remove_file(file)?;
        }
        Ok(())
    }

    /// 使用 OnDiskInvertedLists 合并索引
    ///
    /// 注意这种模式下，必须合并到一个空索引中
    pub fn merge_index_on_disk(&self) -> Result<()> {
        info!("在磁盘上合并所有索引……");

        let mut invfs = vec![];
        for mut sub_index in self.get_sub_index(true) {
            invfs.push(sub_index.invlists());
            sub_index.set_own_invlists(false);
        }

        // 旧索引
        if self.conf_dir.index().exists() {
            let mut index = if self.conf_dir.ondisk_ivf().exists() {
                // OnDiskIVF 格式永远是 mmap，不能设置 mmap
                FaissIndex::from_file(self.conf_dir.index(), false)
            } else {
                FaissIndex::from_file(self.conf_dir.index(), true)
            };
            index.set_own_invlists(false);
            invfs.push(index.invlists());
        }

        let mut template = self.get_template_index();

        let mut invlists = FaissOnDiskInvLists::new(
            template.nlist(),
            template.code_size() as usize,
            self.conf_dir.ondisk_ivf_tmp(),
        );

        info!("合并倒排列表……");
        let ntotal = invlists.merge_from_multiple(invfs, false, true);
        invlists.set_filename(self.conf_dir.ondisk_ivf());
        fs::rename(self.conf_dir.ondisk_ivf_tmp(), self.conf_dir.ondisk_ivf())?;

        template.replace_invlists(invlists, true);
        template.set_ntotal(ntotal as i64);

        info!("保存索引……");
        template.write_file(self.conf_dir.index());

        for file in self.conf_dir.all_sub_index() {
            fs::remove_file(file)?;
        }
        Ok(())
    }
}
