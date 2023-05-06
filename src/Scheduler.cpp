
#include "../include/Scheduler.h";
#include "../include/Configuration.h"
#include "../include/DataTypes.h"
#include "../include/Utils.h"



Scheduler::Scheduler(CameraParameters cameraParameters, GlobalConfiguration configuration, PreProcessor preprocessor,SurfaceMeasurement surfaceMeasurement)
    : cameraParameters(cameraParameters), configuration(configuration), preProcessor(preprocessor), surfaceMeasurement(surfaceMeasurement),frameId(0){

}


bool Scheduler::process_new_frame(const cv::Mat& depth_map, const cv::Mat& color_map){
    //预处理，转换格式，生成金字塔，双边滤波
    FrameData frameData = preProcessor.preProcess(depth_map);
    //生成顶点图和法向量图
    surfaceMeasurement.genVertexAndNormalMap(frameData);
    //将顶点图保存为点云
    cv::Mat downloaded_vertex_map;
    frameData.vertex_pyramid[0].download(downloaded_vertex_map);
    utils::vertexMapToPointCloudAndSave(downloaded_vertex_map);
    return true;

}


//构造调度器。工厂模式
Scheduler SchedulerFactory::build(){
    //设置参数
    CameraParameters cameraParameters(
        config::IMAGE_WIDTH,
        config::IMAGE_HEIGHT,
        config::FOCAL_X,
        config::FOCAL_Y,
        config::PRINCIPAL_X,
        config::PRINCIPAL_Y
    );

    GlobalConfiguration configuration(
        config::DEPTH_CUTOFF,
        config::KERNAL_SIZE,
        config::COLOR_SIGMA,
        config::SPATIAL_SIGMA,
        config::NUM_LEVELS,
        config::DISTANCE_THRESHOLD,
        config::ANGLE_THRESHOLD
    );
    //初始化构造器
    PreProcessor preProcessor(cameraParameters,configuration);
    SurfaceMeasurement surfaceMeasurement(cameraParameters,configuration);
    Scheduler scheduler(cameraParameters,configuration,preProcessor,surfaceMeasurement);
    return scheduler;
}