#ifndef CONTIGUOUSALLOCATOR_H
#define CONTIGUOUSALLOCATOR_H

#include <iostream>
#include <signal.h>
#include <stdlib.h>    
#ifdef __linux__
#include <unistd.h>
#endif 

#include <mutex>
#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tkdnn.h"

//#define OPENCV_CUDACONTRIB //if OPENCV has been compiled with CUDA and contrib.

#ifdef OPENCV_CUDACONTRIB
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

namespace tk {

#ifdef OPENCV_CUDACONTRIB
class ContiguousAllocator : public cv::cuda::GpuMat::Allocator {
    public:
        bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize) override;
        void free(cv::cuda::GpuMat* mat) override;
};

extern ContiguousAllocator cudaContiguousAllocator;
extern cv::cuda::GpuMat::Allocator* g_contiguousAllocator;

cv::cuda::GpuMat::Allocator* contiguousAllocator();

#endif
}

#endif