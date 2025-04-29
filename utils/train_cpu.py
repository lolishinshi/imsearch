#!/usr/bin/env python3
"""
实验性的 CPU 2 级聚类实现，召回率有一定损失，但是速度更快
"""
from sys import argv
from time import time
import faiss
import numpy as np

def binary_kmeans(d, nc, xt, **kwargs):
    """
    适用于二进制数据的 Kmeans 聚类
    """
    cp = faiss.ClusteringParameters()
    for k, v in kwargs.items():
        getattr(cp, k) # 确认参数存在
        setattr(cp, k, v)

    clus = faiss.Clustering(d, nc, cp)
    index_tmp = faiss.IndexFlatL2(d)
    codec = faiss.IndexLSH(d, d, False, False)
    clus.train_encoded(xt, codec, index_tmp)

    x_b = np.zeros((d // 8 * clus.k, 1), dtype=np.uint8)
    faiss.real_to_binary(d * clus.k, clus.centroids.data(), faiss.swig_ptr(x_b))
    x_b = x_b.reshape(clus.k, d // 8)
    return x_b

t0 = time()

def log(msg, *args, **kwargs):
    print(f"[{time() - t0:.2f} s] {msg}", *args, **kwargs)

def main():
    if len(argv) != 3:
        print(f"Usage: {argv[0]} DESCRIPTION train.npy")
        return

    description = argv[1]
    d = 256

    index = faiss.index_binary_factory(d, description)
    xt = np.load(argv[2], mmap_mode="r")

    if not (256 * index.nlist >= len(xt) >= 30 * index.nlist):
        print(f"警告：训练集数量 {len(xt)} 不在合理范围内（{256 * index.nlist} - {30 * index.nlist}）")

    nc1 = int(np.sqrt(index.nlist))
    nc2 = index.nlist

    log(f"对向量 {xt.shape} 进行 2 级聚类，1 级聚类数量 = {nc1}，总数 = {nc2}")
    log("开始一级聚类...")
    x_b = binary_kmeans(d, nc1, xt, niter=25, max_points_per_centroid=2000, verbose=True)

    log("将训练集分配到一级聚类中心")
    index_tmp = faiss.IndexBinaryFlat(d)
    index_tmp.reset()
    index_tmp.add_c(nc1, faiss.swig_ptr(x_b))
    _, assign1 = index_tmp.search(xt, 1)
    assign1 = assign1.ravel()

    bc = np.bincount(assign1, minlength=nc1)
    log(f"聚类大小：{min(bc)} - {max(bc)}")
    o = assign1.argsort()

    bc_sum = np.cumsum(bc)
    all_nc2 = bc_sum * nc2 // bc_sum[-1]
    all_nc2[1:] -= all_nc2[:-1]
    assert sum(all_nc2) == nc2
    log(f"二级聚类中心点数量：{min(all_nc2)} - {max(all_nc2)}")

    i0 = 0
    c2 = []
    for c1 in range(nc1):
        nc2 = int(all_nc2[c1])
        log(f"训练子聚类 {c1}/{nc1} nc2={nc2} \r", end='')
        i1 = i0 + bc[c1]
        subset = o[i0:i1]
        assert np.all(assign1[subset] == c1)
        x_b = binary_kmeans(d, nc2, xt[subset])
        c2.append(x_b)
        i0 = i1

    centroids = np.vstack(c2)
    index.quantizer.train(centroids)
    index.quantizer.add(centroids)
    index.train(xt)

    faiss.write_index_binary(index, f'{description}.train')

if __name__ == '__main__':
    main()
