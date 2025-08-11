# imsearch

这是一个基于特征点匹配的大规模相似图片搜索工具，开发目的在于实现用「一小块截图」实现搜索完整图片的功能。

## 安装方式

### Docker

```
docker run -it -v ./imsearch:/root/.config/imsearch aloxaf/imsearch:latest --help
```

### 手动安装

```bash
cargo install --git https://github.com/lolishinshi/imsearch --locked
```

依赖（以 Ubuntu 24.04 为例，不建议使用更老的版本）：

```bash
apt install cmake clang libopencv-dev libopenblas-dev
```

## 基本用法

### 1. 添加图片

使用 `imsearch add DIR` 添加指定目录下的所有图片。

默认只扫描 jpg 和 png，可以使用 `-s jpg,png,webp` 增加其他格式。

### 2. 选择聚类大小

参考 [Guidelines-to-choose-an-index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)，给出推荐如下。

| 图片数量   | 索引描述                                                    |
| ---------- | ----------------------------------------------------------- |
| < 2K       | $K=4\sqrt{N} \sim 16\sqrt{N}$, $N=\text{图片数量}\times500$ |
| 2K ~ 20K   | K=65536                                                     |
| 20K ~ 200K | K=262144                                                    |
| 200K ~ 2M  | K=1048576                                                   |

> 注意：以上选择基于每张图片提取 500 个特征点这一默认参数，下同

### 3. 训练索引

BIVF 索引需要一定量的数据训练聚类器，推荐数据量为 K 的 30 ~ 256 倍。
即对于 K = 65536，需要 65536 \* 50 / 500 =~ 6.5k 张图片。

训练集需要有代表性，如果和实际数据集相差过大会导致索引不平衡，影响搜索速度。

```bash
imsearch train -c 65536 -i 6500
```

训练结束后，会得到一个 `quantizer.bin` 文件。

### 4. 构建索引

将先前训练得到的 `quantizer.bin` 移动到配置目录中（可以使用 `imsearch --help` 查看默认目录）。

然后使用 `imsearch build` 构建索引。

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
