#include "ORBextractor.h"
#include "ocvrs_common.hpp"

extern "C" {
Result<void*> slam3_ORB_create(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST, int interpolation, bool angle) {
    try {
        return Ok<void*>(new ORB_SLAM3::ORBextractor(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST, interpolation, angle));
    } OCVRS_CATCH(Result<void *>)
}

void slam3_ORB_delete(ORB_SLAM3::ORBextractor *self) {
    delete self;
}

Result_void slam3_ORB_detect_and_compute(ORB_SLAM3::ORBextractor *self, cv::InputArray _image, cv::InputArray _mask,
                                  std::vector<cv::KeyPoint> &_keypoints,
                                  cv::OutputArray _descriptors, std::vector<int> &vLappingArea) {
    try {
        self->operator()(_image, _mask, _keypoints, _descriptors, vLappingArea);
        return Ok();
    } OCVRS_CATCH(Result_void)
}
}