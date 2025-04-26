# imsearch

这是一个基于特征点匹配的大规模相似图片搜索工具，开发目的在于实现用「一小块截图」实现搜索完整图片的功能。

## 安装方式

`cargo install --git https://github.com/lolishinshi/imsearch`

imsearch 依赖 opencv，请确保安装了 opencv 和 cmake 等基本构建工具。

## 基本用法

### 1. 选择索引

参考 [Guidelines-to-choose-an-index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)，给出推荐如下。

| 图片数量   | 索引描述           |
| ---------- | ------------------ |
| 2K ~ 20K   | BIVF65536          |
| 20K ~ 200K | BIVF262144_HNSW32  |
| 200K ~ 2M  | BIVF1048576_HNSW32 |

> 注意：以上选择基于每张图片提取 500 个特征点这一默认参数，下同

### 2. 训练索引

BIVF 索引需要一定量的数据训练聚类器，推荐数据量为 50 倍桶数量。
即对于 K = 65536，需要 65536 \* 50 / 500 =~ 6.5k 张图片。

训练集需要有代表性，如果和实际数据集相差过大会导致索引不平衡，影响搜索速度。

将训练集放到 train 文件夹中，使用下列命令提取训练集中的特征点，并导出 train.npy 文件：

```bash
imsearch -c ./train.db add ./train
imsearch -c ./train.db export
```

由于训练需要 GPU 参与，此处采用 faiss 的 python 绑定来进行训练（纯 CPU 训练非常慢）。
可以参见[官方教程](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md#installing-faiss-via-conda)使用 conda 安装或直接用 pip 安装第三方编译的 faiss-gpu-cu12，然后使用以下命令进行训练：

```python
python utils/train.py DESCRIPTION train.npy
```

训练结束后，会得到一个 `{DESCRIPTION}.train` 文件。

### 3. 添加图片

使用 `imsearch add DIR` 添加指定目录下的所有图片。

默认只扫描 jpg 和 png，可以使用 `-s jpg,png,webp` 增加其他格式。

### 4. 构建索引

将先前训练得到的 `{DESCRIPTION}.train` 重命名为 `index.template` 并保存到配置目录中（可以使用 `imsearch --help` 查看默认目录）。

然后使用 `imsearch build` 构建索引，注意这个过程需要大量内存，并且非常慢。

### 5. 搜索图片

```shell
# 让 imsearch 打印详细日志
export RUST_LOG=imsearch=debug

# 以默认参数直接搜索单张图片
imsearch search test.jpg

# --nprobe=128：搜索倒排列表中最接近的 128 个 bucket，提高了精度但耗费更多时间
imsearch search --nprobe=128 test.jpg
```

其他高级参数请使用 `--help` 查看。

### 6. HTTP API

```bash
# 由于 HTTP 服务需要长期运行，这里采用一次性加载索引到内存的方式来提升速度
# 但需要注意内存容量是否满足
imsearch --no-mmap server
```

启动 http 服务后，访问 `http://127.0.0.1:8000/docs` 可以看到 API 列表。

## 致谢

- [ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [faiss](https://github.com/facebookresearch/faiss)
