"""
该脚本用于转换一个模板索引的量化器
"""

import faiss
import os
import numpy as np
from sys import argv

def main():
    if len(argv) != 4:
        print(f"Usage: {argv[0]} [--flat|--hnsw] <index_path> <output_path>")
        return

    target = argv[1]
    index_path = argv[2]
    output_path = argv[3]

    print("正在读取索引...")
    index = faiss.read_index_binary(index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)

    if index.ntotal != 0:
        print("警告：索引不为空，转换后不会保留旧数据")

    quantizer = faiss.downcast_IndexBinary(index.quantizer)

    match quantizer:
        case faiss.IndexBinaryHNSW():
            storage = faiss.downcast_IndexBinary(quantizer.storage)
        case faiss.IndexBinaryFlat():
            storage = quantizer
        case _:
            print("索引的量化器不是 faiss.IndexBinaryHNSW 或 faiss.IndexBinaryFlat")
            return

    match target:
        case "--flat":
            new_quantizer = faiss.IndexBinaryFlat(index.d)
        case "--hnsw":
            new_quantizer = faiss.IndexBinaryHNSW(index.d)
        case "--hnsw-lsg":
            # https://github.com/19-hanhan/LSG
            new_quantizer = faiss.IndexBinaryHNSW(index.d)
            if not hasattr(new_quantizer, "add_lsg"):
                print("当前 faiss 版本不支持 LSG 优化，请使用 https://github.com/Aloxaf/faiss/tree/imsearch_lsg 分支")
                return
        case _:
            print("参数错误")
            return

    new_quantizer.reset()
    if target == "--hnsw-lsg":
        print("正在计算平均距离...")
        xb = faiss.rev_swig_ptr(storage.xb.data(), storage.xb.c_size)
        xb = xb.reshape((storage.ntotal, storage.code_size))
        size = int(min(max(xb.shape[0] * 0.01, 10000), xb.shape[0]))
        ri = np.random.choice(xb.shape[0], size=size, replace=False)
        t = faiss.IndexBinaryFlat(index.d)
        t.add(xb[ri])
        D, I = t.search(xb, 10)
        avgdis = np.mean(D, axis=1, dtype=np.float32)
        print("正在转换索引...")
        new_quantizer.add_lsg(storage.ntotal, storage.xb.data(), faiss.swig_ptr(avgdis), 1.0)
    else:
        print("正在转换索引...")
        new_quantizer.add_c(storage.ntotal, storage.xb.data())
    new_quantizer.is_trained = True
    # 直接替换旧索引的 quantizer，得到的结果大小不一致
    # 干脆重新创建一个索引，只保留新的量化器，不允许保留旧数据
    index = faiss.IndexBinaryIVF(new_quantizer, index.d, storage.ntotal)

    print("正在写入索引...")
    if os.path.exists(output_path):
        print("目标索引已存在，是否覆盖？(y/N) ", end="")
        if input() != "y":
            print("取消转换")
            return
    faiss.write_index_binary(index, output_path)
    print("转换完成")

if __name__ == "__main__":
    main()
