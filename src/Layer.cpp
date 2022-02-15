#include <iostream>

#include "tkDNN/Layer.h"

namespace tk { namespace dnn {

Layer::Layer(Network *net) {

    this->net = net;
    this->final = false;
    if(net != nullptr) {
        this->input_dim = net->getOutputDim();
        this->output_dim = input_dim;
        
        checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );

        if(!net->addLayer(this))
            FatalError("Net reached max number of layers");    
    }

    feature_map_size = input_dim.tot() + output_dim.tot();
}

Layer::~Layer() {

    checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
    checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );

    if(dstData != nullptr) {
        cudaFree(dstData);
        dstData = nullptr;
    }
}

}}