from pathlib import Path
from sys import argv
import faiss
import numpy as np


def main():
    if len(argv) != 2:
        print(f"Usage: {argv[0]} K train.npy")

    k = int(argv[1])
    d = 256
    quantizer = faiss.IndexBinaryFlat(d)
    index = faiss.IndexBinaryIVF(quantizer, d, k)
    clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(d))
    index.clustering_index = clustering_index

    tr = np.load(argv[2], mmap_mode="r")

    index.train(tr)
    faiss.write_index_binary(index, str(Path.home() / '.config/imsearch/index'))


if __name__ == '__main__':
    main()
