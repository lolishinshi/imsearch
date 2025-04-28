#!/usr/bin/env python3
"""
GPU 索引训练
"""
from sys import argv
import faiss
import numpy as np


def main():
    if len(argv) != 3:
        print(f"Usage: {argv[0]} DESCRIPTION train.npy")
        return
    
    description = argv[1]
    d = 256

    index = faiss.index_binary_factory(d, description)
    index.verbose = True
    index.cp.verbose = True

    clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(d))
    index.clustering_index = clustering_index

    tr = np.load(argv[2], mmap_mode="r")
    index.train(tr)

    faiss.write_index_binary(index, f'{description}.train')


if __name__ == '__main__':
    main()
