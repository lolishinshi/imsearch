#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <cstdio>
#include <iostream>

using namespace faiss;
using idx_t = int64_t;

extern "C" {
IndexBinary* faiss_indexBinary_factory(int d, const char* description)
{
    try {
        return faiss::index_binary_factory(d, description);
    } catch (std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
}

void faiss_IndexBinary_train(IndexBinary* index, idx_t n, const uint8_t* x)
{
    try {
        return index->train(n, x);
    } catch (std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
}

void faiss_IndexBinary_add(IndexBinary* index, idx_t n, const uint8_t* x)
{
    try {
        return index->add(n, x);
    } catch (std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
}

void faiss_IndexBinary_add_with_ids(IndexBinary* index, idx_t n, const uint8_t* x, const idx_t* xids)
{
    return index->add_with_ids(n, x, xids);
}

void faiss_IndexBinary_search(IndexBinary* index, idx_t n, const uint8_t* x, idx_t k, int32_t* distances, idx_t* labels)
{
    try {
        return index->search(n, x, k, distances, labels);
    } catch (std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
}

void faiss_IndexBinary_delete(IndexBinary* index)
{
    try {
        delete index;
    } catch (std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
}

void faiss_write_index_binary(const IndexBinary* idx, const char *fname)
{
    return write_index_binary(idx, fname);
}

IndexBinary* faiss_read_index_binary(const char *fname, int io_flags)
{
    return read_index_binary(fname, io_flags);
}

}
