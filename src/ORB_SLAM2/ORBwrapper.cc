#include "ORBextractor.h"

extern "C" {
void *slam2_ORB_create(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST) {
    return new ORB_SLAM2::ORBextractor(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST);
}

void slam2_ORB_delete(ORB_SLAM2::ORBextractor *self) {
    delete self;
}

void slam2_ORB_detect_and_compute(ORB_SLAM2::ORBextractor *self, cv::InputArray _image, cv::InputArray _mask,
                                  std::vector<cv::KeyPoint> &_keypoints,
                                  cv::OutputArray _descriptors) {
    self->operator()(_image, _mask, _keypoints, _descriptors);
}
}