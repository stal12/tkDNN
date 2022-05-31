#include "ContiguousAllocator.h"


namespace tk {

#ifdef OPENCV_CUDACONTRIB

bool ContiguousAllocator::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
{
    checkCuda( cudaMalloc(&mat->data, elemSize * cols * rows) );
    mat->step = elemSize * cols;
    mat->refcount = (int*) cv::fastMalloc(sizeof(int));

    return true;
}

void ContiguousAllocator::free(cv::cuda::GpuMat* mat)
{
    checkCuda(cudaFree(mat->datastart));
    cv::fastFree(mat->refcount);
}

ContiguousAllocator cudaContiguousAllocator;
cv::cuda::GpuMat::Allocator* g_contiguousAllocator = &cudaContiguousAllocator;

cv::cuda::GpuMat::Allocator* contiguousAllocator()
{
    return g_contiguousAllocator;
}

#endif

}