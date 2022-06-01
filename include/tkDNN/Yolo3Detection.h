#ifndef Yolo3Detection_H
#define Yolo3Detection_H
#include <opencv2/videoio.hpp>
#include "opencv2/opencv.hpp"

#include "DetectionNN.h"
#include "DarknetParser.h"

namespace tk { namespace dnn {
class Yolo3Detection : public DetectionNN
{
private:
    int num = 0;
    int nMasks = 0;
    int nDets = 0;
    tk::dnn::Yolo::detection *dets = nullptr;
    tk::dnn::Yolo* yolo[3];

    tk::dnn::Yolo* getYoloLayer(int n=0);

    cv::Mat bgr_h;

#ifdef OPENCV_CUDACONTRIB
    cv::cuda::GpuMat orig_img, img_resized;
#endif
    
public:
    Yolo3Detection()
#ifdef OPENCV_CUDACONTRIB    
    :
    orig_img(contiguousAllocator()),
    img_resized(contiguousAllocator())
#endif    
    {};
    ~Yolo3Detection() {}; 

    bool init(const std::string& tensor_path, const int n_classes=80, const int n_batches=1, const float conf_thresh=0.3, const std::vector<std::string>& class_names = {}, bool cuda_graph = false) override;
    void preprocess(cv::Mat &frame, const int bi=0) override;
    void postprocess(const int bi=0,const bool mAP=false) override;
};


} // namespace dnn
} // namespace tk

#endif /* Yolo3Detection_H*/
