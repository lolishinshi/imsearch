#include "ORBextractor.h"
#include "ocvrs_common.hpp"

extern "C" {
void slam3_ORB_create(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST, int interpolation, bool angle, Result<void*>* ocvrs_return) {
    try {
        return Ok<void*>(new ORB_SLAM3::ORBextractor(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST, interpolation, angle), ocvrs_return);
    } OCVRS_CATCH(ocvrs_return)
}

void slam3_ORB_delete(ORB_SLAM3::ORBextractor *self) {
    delete self;
}

void slam3_ORB_detect_and_compute(ORB_SLAM3::ORBextractor *self, cv::InputArray _image, cv::InputArray _mask,
                                  std::vector<cv::KeyPoint> &_keypoints,
                                  cv::OutputArray _descriptors, std::vector<int> &vLappingArea, Result<int>* ocvrs_return) {
    try {
        return Ok<int>(self->operator()(_image, _mask, _keypoints, _descriptors, vLappingArea), ocvrs_return);
    } OCVRS_CATCH(ocvrs_return)
}
}