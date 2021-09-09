#include "ORBextractor.h"
#include "ocvrs_common.hpp"

extern "C" {
Result<void*> slam2_ORB_create(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST) {
    try {
        return Ok<void*>(new ORB_SLAM2::ORBextractor(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST));
    } OCVRS_CATCH(Result<void *>)
}

void slam2_ORB_delete(ORB_SLAM2::ORBextractor *self) {
    delete self;
}

Result_void slam2_ORB_detect_and_compute(ORB_SLAM2::ORBextractor *self, cv::InputArray _image, cv::InputArray _mask,
                                  std::vector<cv::KeyPoint> &_keypoints,
                                  cv::OutputArray _descriptors) {
    try {
        self->operator()(_image, _mask, _keypoints, _descriptors);
        return Ok();
    } OCVRS_CATCH(Result_void)
}
}