#include "flann/flann.hpp"
#include <opencv2/opencv.hpp>
#include <stdio.h>

using Index = flann::Index<flann::Hamming<unsigned char>>;

extern "C" {
    void* knn_searcher_init(cv::Mat &points, unsigned int table_number, unsigned int key_size, unsigned int multi_probe_level)
    {
        auto features = flann::Matrix<unsigned char>(points.data, points.rows, points.cols);
        auto params = flann::LshIndexParams(table_number, key_size, multi_probe_level);
        auto index = new Index(features, params);
        return index;
    }

    void knn_searcher_add(Index *self, cv::Mat &points)
    {
        try {
            auto features = flann::Matrix<unsigned char>(points.data, points.rows, points.cols);
            self->addPoints(features);
        } catch (std::runtime_error &e) {
            flann::Logger::error("Caught exception: %s\n",e.what());
        }
    }

    void knn_searcher_build_index(Index *self)
    {
        self->buildIndex();
    }

    int knn_searcher_search(Index *self, cv::Mat &points, size_t *indices, unsigned int *dists, size_t knn, int checks)
    {
        //try {
            const auto _points = flann::Matrix<unsigned char>(points.data, points.rows, points.cols);
            auto _indices = flann::Matrix<size_t>(indices, points.rows, knn);
            auto _dists = flann::Matrix<unsigned int>(dists, points.rows, knn);
            auto search_params = flann::SearchParams(checks);
            search_params.cores = 0;
            return self->knnSearch(_points, _indices, _dists, knn, search_params);
        //} catch (std::runtime_error &e) {
        //    flann::Logger::error("Caught exception: %s\n",e.what());
        //}
    }

    void knn_searcher_delete(Index *self)
    {
        delete self;
    }
}