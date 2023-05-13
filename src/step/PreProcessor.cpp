#include "../../include/step/PreProcessor.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
using step::PreProcessor;

PreProcessor::PreProcessor(CameraParameters cameraParameters,GlobalConfiguration globalConfiguration)
    :cameraParameters(cameraParameters),globalConfiguration(globalConfiguration){
}

FrameData PreProcessor::preProcess(const cv::Mat& depth_map){
    //传入的为16UC1的深度图，需要先转为32FC1
    cv::Mat depth_map_32f;
    depth_map.convertTo(depth_map_32f, CV_32F, 1.0/globalConfiguration.depthScale * 1000);
    //初始化帧数据
    int numLevels = globalConfiguration.numLevels;         
    FrameData data(numLevels,cameraParameters);

    // 上传原始的深度图到GPU
    data.depth_pyramid[0].upload(depth_map_32f);

    cv::cuda::Stream stream;
    //生成深度图图像金字塔
    for (int level = 1; level < numLevels; level++)
        cv::cuda::pyrDown(data.depth_pyramid[level - 1], data.depth_pyramid[level], stream);
    //对图像金字塔中的每一张图像都进行双边滤波操作
    for (int level = 0; level < numLevels; level++) {
        cv::cuda::bilateralFilter(data.depth_pyramid[level],            // source
                                  data.smoothed_depth_pyramid[level],   // destination
                                  globalConfiguration.kernalSize,       // 双边滤波器滤波的窗口大小
                                  globalConfiguration.colorSigma,
                                  globalConfiguration.spatialSigma,
                                  cv::BORDER_DEFAULT,                   // 默认边缘的补充生成方案
                                  stream);
    }
    
    // 等待操作完成
    stream.waitForCompletion();
    return data;
}

