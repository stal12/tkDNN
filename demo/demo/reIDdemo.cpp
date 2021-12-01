#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
//#include <unistd.h>
#include <mutex>

#include "tkDNN/CenternetDetection.h"
#include "tkDNN/MobilenetDetection.h"
#include "tkDNN/Yolo3Detection.h"

#include "tkDNN/EmbeddingNN.h"

bool gRun;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[]) {

    signal(SIGINT, sig_handler);

    // get config file path and read it
    #ifdef __linux__ 
        std::string config_file = "../demo/demoConfig.yaml";
    #elif _WIN32
        std::string config_file = "..\\..\\..\\demo\\demoConfig.yaml";
    #endif
    if(argc > 1)
        config_file = argv[1]; 
    
    YAML::Node conf =  YAMLloadConf(config_file);
    if(!conf)
        FatalError("Problem with config file");

    // read settings from config file
    std::string net = YAMLgetConf<std::string>(conf, "net", "yolo4tiny_fp32.rt");
    if(!fileExist(net.c_str()))
        FatalError("The given network does not exist. Create the rt first.");

    #ifdef __linux__ 
        std::string input = YAMLgetConf<std::string>(conf, "input", "../demo/yolo_test.mp4");
    #elif _WIN32
        std::string input = YAMLgetConf(conf, "win_input", "..\\..\\..\\demo\\yolo_test.mp4");
    #endif
    if(!fileExist(input.c_str()))
        FatalError("The given input video does not exist.");
    
    char ntype          = YAMLgetConf<char>(conf, "ntype", 'y');
    int n_classes       = YAMLgetConf<int>(conf, "n_classes", 80);
    float conf_thresh   = YAMLgetConf<float>(conf, "conf_thresh", 0.3);
    bool show           = YAMLgetConf<bool>(conf, "show", true);
    bool save           = YAMLgetConf<bool>(conf, "save", false);

    std::cout   <<"Net settings - net: "<< net
                <<", ntype: "<< ntype
                <<", n_classes: "<< n_classes
                <<", conf_thresh: "<< conf_thresh<<"\n"; 
    std::cout   <<"Demo settings - input: "<< input
                <<", show: "<< show
                <<", save: "<< save<<"\n\n"; 
    
    // create detection network
    tk::dnn::Yolo3Detection yolo;
    tk::dnn::CenternetDetection cnet;
    tk::dnn::MobilenetDetection mbnet;  

    tk::dnn::DetectionNN *detNN;  

    switch(ntype)
    {
        case 'y':
            detNN = &yolo;
            break;
        case 'c':
            detNN = &cnet;
            break;
        case 'm':
            detNN = &mbnet;
            n_classes++;
            break;
        default:
        FatalError("Network type not allowed (3rd parameter)\n");
    }

    detNN->init(net, n_classes, 1, conf_thresh);

    // create embedding network
    int n_batch = 5; // batch of the feature extractor
    std::string emb_net = "resnet101_fp32.rt";

    tk::dnn::EmbeddingNN embNN;
    embNN.init(emb_net, n_batch);

    // open video stream
    cv::VideoCapture cap(input);
    if(!cap.isOpened())
        gRun = false; 
    else
        std::cout<<"camera started\n";

    cv::VideoWriter resultVideo;
    if(save) {
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        resultVideo.open("result.mp4", cv::VideoWriter::fourcc('M','P','4','V'), 30, cv::Size(w, h));
    }

    if(show)
        cv::namedWindow("detection", cv::WINDOW_NORMAL);

    cv::Mat frame;
    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;

    cv::Mat croppedDet;
    std::vector<cv::Mat> batch_detections;
    std::vector<cv::Mat> batch_embdnn_input;

    // start detection loop
    gRun = true;
    while(gRun) {
        batch_dnn_input.clear();
        batch_frame.clear();
        batch_detections.clear();
        batch_embdnn_input.clear();
        
        //read frame
        cap >> frame; 
        if(!frame.data) 
            break;
        batch_frame.push_back(frame);
        batch_dnn_input.push_back(frame.clone());
    
        //inference
        detNN->update(batch_dnn_input, 1);

        //extract detections of pedestrians
        for(int i = 0; i<detNN->detected.size(); ++i){
            if(detNN->detected[i].cl == 0){ // pedestrian

                float x = detNN->detected[i].x;
                float y = detNN->detected[i].y;
                float w = detNN->detected[i].w;
                float h = detNN->detected[i].h;

                if(x < 0) x = 0;
                if(w < 0) w = 0;
                if(x + w > frame.cols) w = frame.cols - x; 
                if(y < 0) y = 0;
                if(h < 0) h = 0;
                if(y + h > frame.rows) h = frame.rows - y; 

                croppedDet = frame(cv::Rect(x,y,w,h));
                // cv::imshow("croppedDet", croppedDet);
                // cv::waitKey(0);

                batch_detections.push_back(croppedDet);
                batch_embdnn_input.push_back(croppedDet.clone());
            } 
        }

        if(!batch_embdnn_input.empty()){
            embNN.update(batch_embdnn_input, batch_embdnn_input.size());

            // for(int i=0; i<batch_embdnn_input.size(); ++i){
            //     for(int j=0; j<batch_embdnn_input.size(); ++j){
            //         float d = tk::dnn::computeEmbeddingsDistance(embNN.embeddings[i],embNN.embeddings[j] );
            //         std::cout<<" Distance among "<<i<<" and "<< j<< " is "<<d<< "\n";
            //     }
            // }
        }

        detNN->draw(batch_frame);
        if(show){
            cv::imshow("detection", batch_frame[0]);
            cv::waitKey(1);

        }
    }

    std::cout<<"detection end\n";   
    
    double mean = 0; 
    std::cout<<COL_GREENB<<"\n\nTime stats embedding:\n";
    std::cout<<"Min: "<<*std::min_element(embNN.stats.begin(), embNN.stats.end())<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(embNN.stats.begin(), embNN.stats.end())<<" ms\n";    
    for(int i=0; i<embNN.stats.size(); i++) mean += embNN.stats[i]; mean /= embNN.stats.size();
    std::cout<<"Avg: "<<mean<<" ms\t"<<1000/(mean)<<" FPS\n";   

    mean = 0;
    std::cout<<"\n\nTime stats detection:\n";
    std::cout<<"Min: "<<*std::min_element(detNN->stats.begin(), detNN->stats.end())<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(detNN->stats.begin(), detNN->stats.end())<<" ms\n";    
    for(int i=0; i<detNN->stats.size(); i++) mean += detNN->stats[i]; mean /= detNN->stats.size();
    std::cout<<"Avg: "<<mean<<" ms\t"<<1000/(mean)<<" FPS\n"<<COL_END;   


    return 0;
}

