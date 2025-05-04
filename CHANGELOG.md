# Changelog

## 未发布

- HTTP 增加鉴权
- 增加 `--max-features` 参数指定最大特征点数量
- 允许使用 blake3 以外的哈希算法去重

## [2.1.0] - 2025-04-30

- ORB 特征点提取增加 `--max-size` 和 `--max-aspect-ratio` 参数
- `add` 增加 `--min-keypoints`
- 改进图片缩放逻辑为按宽度缩放
- `export` 命令无视图片是否已经索引
- 允许旧版的构建逻辑
- 修复 `update-db` 未正确迁移 id 的问题
- `server` 增加 `--hnsw`，用于将量化器转换为 HNSW
- `update-db` 鲁棒性增加
- HTTP 增加 `/stats` 和 `/reset_stats` 接口
- 修复 `cargo install` 无法使用 `--features=rocksdb` 的问题
- HTTP 接口增加 `/build`，用于构建索引
- HTTP 接口增加 `/add`，用于添加图片
- `add` 命令支持 `--regex` 参数，用于提取文件路径中需要保存的部分
- HTTP 接口增加 `/reload`，用于重新加载索引
- 修复被查询图片无特征点导致的崩溃
- HTTP 接口 `/search` 支持一次性查询多张图片
- `build` 命令支持 `--on-disk` 参数，以 OnDiskInvertedLists 格式合并索引
- 将除 `--conf-dir` 外的所有命令行参数移到对应的子命令中
- 所有命令改为单个单词，如 `build-index` -> `build`
- HTTP 模式下缓存所有的 ID 来加速查询
- HTTP 接口 `/docs` 下增加 Swagger UI

## [2.0.0] - 2025-04-25

- `mark-as-indexed` 命令已移除
- `clear-cache` 命令的 `--unindexed` 重命名为 `--all`
- 增加 `update-db` 命令，用于从 rocksdb 迁移数据
- `build-index` 的 `--batch-size` 含义更改为图片数量而不是特征点数量
- `build-index` 移除 `--start` 和 `--end` 参数，增加 `--mmap` 参数
- `build-index` 命令改为分段构建，再合并索引，需要添加 `index.template`
- 数据库从 rocksdb 改为 sqlite3
- 加载索引时打印索引的统计信息
- 引入进度条库 indicatif，美化添加图片时的日志打印
- 移除内置的训练支持，改用 train.py 训练，同时增加 HNSW 格式的提示
- 允许为单轮搜索单独指定 `--nprobe` 和 `--max-codes` 参数
- HTTP 服务从同步改为异步
- 移除多 index 支持
- faiss 库改为内置 avx2 版本
