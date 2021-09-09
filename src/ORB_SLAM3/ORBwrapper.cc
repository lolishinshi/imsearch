#include "ORBextractor.h"

extern "C" {
void *slam3_ORB_create(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST) {
    return new ORB_SLAM3::ORBextractor(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST);
}

void slam3_ORB_delete(ORB_SLAM3::ORBextractor *self) {
    delete self;
}

void slam3_ORB_detect_and_compute(ORB_SLAM3::ORBextractor *self, cv::InputArray _image, cv::InputArray _mask,
                                  std::vector<cv::KeyPoint> &_keypoints,
                                  cv::OutputArray _descriptors, std::vector<int> &vLappingArea) {
    self->operator()(_image, _mask, _keypoints, _descriptors, vLappingArea);
}
}