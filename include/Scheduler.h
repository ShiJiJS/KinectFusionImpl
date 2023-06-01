#include "Configuration.h"
#include <opencv2/opencv.hpp>
#include "step/PreProcessor.h"
#include "step/SurfaceMeasurement.h"
#include "step/PoseEstimator.h"
#include "step/SurfaceReconstructor.h"
#include "step/SurfacePredictor.h"

using config::CameraParameters;
using config::GlobalConfiguration;
using step::PreProcessor;
using step::SurfaceMeasurement;
using step::PoseEstimator;
using step::SurfaceReconstructor;
using step::SurfacePredictor;



class Scheduler{
private:
    //配置
    CameraParameters cameraParameters; //相机参数
    GlobalConfiguration configuration;  //全局设置
    int frameId;
    //处理器
    PreProcessor preProcessor; //预处理器
    SurfaceMeasurement surfaceMeasurement;
    PoseEstimator poseEstimator;
    SurfaceReconstructor surfaceReconstructor;
    SurfacePredictor surfacePredictor;
    Eigen::Matrix4f Tgk_Matrix; //相机位姿的转移矩阵
    PredictionResult predictionResult; //Raycasting算出的顶点图和法向图
    std::vector<Eigen::Matrix4f> pose_history;//存储之前的位姿结果
    
public:
    Scheduler(CameraParameters cameraParameters,GlobalConfiguration configuration,
        PreProcessor preprocessor,SurfaceMeasurement surfaceMeasurement,
        PoseEstimator poseEstimator, SurfaceReconstructor surfaceReconstructor,
        SurfacePredictor surfacePredictor);

    //添加新一帧的信息
    //depth_map 为16U1C的深度图，color_map为8U3C的RGB图
    bool process_new_frame(const cv::Mat& depth_map, const cv::Mat& color_map);

    void extract_and_save_pointcloud();
    //将位姿信息保存到文件中
    void save_poses();
};

class SchedulerFactory{
public:
    static Scheduler build(std::string configFilePath);
};