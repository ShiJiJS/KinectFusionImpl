#include "Configuration.h"
#include <opencv2/opencv.hpp>
#include "step/PreProcessor.h"
#include "step/SurfaceMeasurement.h"

using config::CameraParameters;
using config::GlobalConfiguration;
using step::PreProcessor;
using step::SurfaceMeasurement;

class Scheduler{
private:
    //配置
    CameraParameters cameraParameters; //相机参数
    GlobalConfiguration configuration;  //全局设置
    int frameId;
    //处理器
    PreProcessor preProcessor; //预处理器
    SurfaceMeasurement surfaceMeasurement;
public:
    Scheduler(CameraParameters cameraParameters,GlobalConfiguration configuration,PreProcessor preprocessor,SurfaceMeasurement surfaceMeasurement);
    //添加新一帧的信息
    //depth_map 为16U1C的深度图，color_map为8U3C的RGB图
    bool process_new_frame(const cv::Mat& depth_map, const cv::Mat& color_map);
};

class SchedulerFactory{
public:
    static Scheduler build();
};