#ifndef DETECTIONNN3D_H
#define DETECTIONNN3D_H

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

// #define OPENCV_CUDACONTRIB //if OPENCV has been compiled with CUDA and contrib.

#ifdef OPENCV_CUDACONTRIB
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#endif


namespace tk { namespace dnn {

class DetectionNN3D {

    protected:
        tk::dnn::NetworkRT *netRT = nullptr;
        dnnType *input_d;

        std::vector<cv::Size> originalSize;

        cv::Scalar colors[256];

        int nBatches = 1;

#ifdef OPENCV_CUDACONTRIB
        cv::cuda::GpuMat bgr[3];
        cv::cuda::GpuMat imagePreproc;
#else
        cv::Mat bgr[3];
        cv::Mat imagePreproc;
        dnnType *input;
#endif

        /**
         * This method preprocess the image, before feeding it to the NN.
         *
         * @param frame original frame to adapt for inference.
         * @param bi batch index
         */
        virtual void preprocess(cv::Mat &frame, const int bi=0) = 0;

        /**
         * This method postprocess the output of the NN to obtain the correct 
         * boundig boxes. 
         * 
         * @param bi batch index
         * @param mAP set to true only if all the probabilities for a bounding 
         *            box are needed, as in some cases for the mAP calculation
         */
        virtual void postprocess(const int bi=0,const bool mAP=false) = 0;

    public:
        int classes = 0;
        float confThreshold = 0.3; /*threshold on the confidence of the boxes*/
        
        std::vector<tk::dnn::box3D> detected3D; /*bounding boxes in output*/
        std::vector<std::vector<tk::dnn::box3D>> batchDetected; /*bounding boxes in output*/
        std::vector<double> pre_stats, stats, post_stats, visual_stats; /*keeps track of inference times (ms)*/
        std::vector<std::string> classesNames;

        DetectionNN3D(){};
        ~DetectionNN3D(){};

        /**
         * Method used to initialize the class, allocate memory and compute 
         * needed data.
         * 
         * @param tensor_path path to the rt file of the NN.
         * @param n_classes number of classes for the given dataset.
         * @param n_batches maximum number of batches to use in inference.
         * @return true if everything is correct, false otherwise.
         */
        virtual bool init(const std::string& tensor_path, const int n_classes=3, const int n_batches=1, 
                            const float conf_thresh=0.3, const std::vector<cv::Mat>& k_calibs=std::vector<cv::Mat>()) = 0;

        /**
         * This method performs the whole detection of the NN.
         * 
         * @param frames frames to run detection on.
         * @param cur_batches number of batches to use in inference.
         * @param save_times if set to true, preprocess, inference and postprocess times 
         *        are saved on a csv file, otherwise not.
         * @param times pointer to the output stream where to write times.
         * @param mAP set to true only if all the probabilities for a bounding 
         *            box are needed, as in some cases for the mAP calculation.
         */
        void update(std::vector<cv::Mat>& frames, const int cur_batches=1, bool save_times=false, 
                    std::ofstream *times=nullptr, const bool mAP=false){
            if(save_times && times==nullptr)
                FatalError("save_times set to true, but no valid ofstream given");
            if(cur_batches > nBatches)
                FatalError("A batch size greater than nBatches cannot be used");

            originalSize.clear();
            if(TKDNN_VERBOSE) printCenteredTitle(" TENSORRT detection ", '=', 30); 
            {
                TKDNN_TSTART
                for(int bi=0; bi<cur_batches;++bi){
                    if(!frames[bi].data)
                        FatalError("No image data feed to detection");
                    originalSize.push_back(frames[bi].size());
                    preprocess(frames[bi], bi);   
                }
                TKDNN_TSTOP
                pre_stats.push_back(t_ns); 
                if(save_times) *times<<t_ns<<";";
            }

            //do inference
            tk::dnn::dataDim_t dim = netRT->input_dim;
            dim.n = cur_batches;
            {
                if(TKDNN_VERBOSE) dim.print();
                TKDNN_TSTART
                netRT->infer(dim, input_d);
                TKDNN_TSTOP
                if(TKDNN_VERBOSE) dim.print();
                stats.push_back(t_ns);
                if(save_times) *times<<t_ns<<";";
            }

            batchDetected.clear();
            {
                TKDNN_TSTART
                for(int bi=0; bi<cur_batches;++bi)
                    postprocess(bi, mAP);
                TKDNN_TSTOP
                post_stats.push_back(t_ns);
                if(save_times) *times<<t_ns<<"\n";
            }
        }

        /**
         * Method to draw bounding boxes and labels on a frame.
         * 
         * @param frames original frame to draw bounding box on.
         */    
        virtual void draw(std::vector<cv::Mat>& frames){};
          
};

}}

#endif /* DETECTIONNN3D_H*/
