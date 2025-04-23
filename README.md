# imsearch

基于特征点匹配的的局部图像搜索工具

主要基于以下项目：

- [ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) - 解决了传统 ORB 算法中存在的特征点过于集中的问题
- [faiss](https://github.com/facebookresearch/faiss) - 对大规模向量进行搜索

## 安装方式

1. 安装 OpenCV

2. `cargo install --git https://github.com/lolishinshi/imsearch`

## 用法

### 训练

首次运行时，需要根据大概需要添加的图片数量训练索引：

- 2k ～ 2w： K 取 65536，需要至少 5.2k 张图片训练
- 2w ～ 20w：K 取 262144，至少需要 21k 张图片训练
- 20w ~ 200w：K 取 1048576，至少需要 82k 张图片训练

然后将训练图片放到 train 文件夹内，并使用以下命令提取图片特征点：

```bash
imsearch add-images ./train
imsearch export-data
# 最终在当前目录下生成一个 train.npy
```

DESCRIPTION 可以为 BIVF{K} 或 BIVF{k}\_HNSW32。

前者精度更高、但速度稍慢，后者精度略低、但速度更快。以下均采用前者作为例子。

```bash
python utils/train.py DESCRIPTION train.npy
# 训练结果会保存为 {DESCRIPTION}.train
```

注：大数据集上的训练非常耗时，在 K = 1048576，训练图片为 100k 张时，两张 3080 花了 16 个小时才训练完成。

### 添加图片

使用 `imsearch add-images DIR` 添加指定目录下的所有图片

### 构建索引

使用 `imsearch build-index` 构建索引，这个过程同样非常慢，在 3970x 上，需要约 20 ～ 40 分钟构建 10k 张图片的索引

注：可以设置 `RUST_LOG=debug` 来打印详细日志以观察进度

### 搜索图片

```shell
# 让 imsearch 打印详细日志
export RUST_LOG=debug

# 以默认参数直接搜索单张图片
imsearch search-image test.jpg

# --mmap：不需要加载整个 index 到内存
# --nprobe=128：搜索附近的 128 的 bucket，提高了精度但耗费更多时间
imsearch --mmap --nprobe=128 search-image test.jpg

# 启动服务器，监听 127.0.0.1:8000 端口
imsearch --mmap start-server

# 使用 httpie 通过 web api 搜索图片
http --form http://127.0.0.1:8000/search file@test.jpg
```

搜索耗时：250w 张图片的索引，在 3970x 上搜索一次耗时约 0.5s
