"""
该脚本用于转换一个模板索引的量化器
"""

import faiss
import os
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

    print("正在转换索引...")

    quantizer = faiss.downcast_IndexBinary(index.quantizer)
    if target == "--flat":
        if not isinstance(quantizer, faiss.IndexBinaryHNSW):
            print("索引的量化器不是 faiss.IndexBinaryHNSW")
            return
        storage = faiss.downcast_IndexBinary(quantizer.storage)
        new_quantizer = faiss.IndexBinaryFlat(index.d)

    elif target == "--hnsw":
        if not isinstance(quantizer, faiss.IndexBinaryFlat):
            print("索引的量化器不是 faiss.IndexBinaryFlat")
            return
        storage = quantizer
        new_quantizer = faiss.IndexBinaryHNSW(index.d)
    else:
        print("参数错误")
        return

    new_quantizer.reset()
    new_quantizer.train_c(storage.ntotal, storage.xb.data())
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
