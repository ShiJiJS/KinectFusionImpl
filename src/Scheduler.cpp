
#include "../include/Scheduler.h"
#include "../include/Configuration.h"
#include "../include/DataTypes.h"
#include "../include/Utils.h"



Scheduler::Scheduler(CameraParameters cameraParameters, GlobalConfiguration configuration, PreProcessor 
preprocessor,SurfaceMeasurement surfaceMeasurement,PoseEstimator poseEstimator,
SurfaceReconstructor surfaceReconstructor,SurfacePredictor surfacePredictor)
    : cameraParameters(cameraParameters), configuration(configuration), preProcessor(preprocessor), 
    surfaceMeasurement(surfaceMeasurement),poseEstimator(poseEstimator),
    predictionResult(configuration.numLevels,cameraParameters),surfaceReconstructor(surfaceReconstructor),
    surfacePredictor(surfacePredictor),frameId(0){
        //初始化Tgk为单位矩阵
        this->Tgk_Matrix = Eigen::Matrix4f::Identity();
        // The pose starts in the middle of the cube, offset along z by the initial depth
        // 第一帧的相机位姿设置在 Volume 的中心, 然后在z轴上拉远一点
        Tgk_Matrix(0, 3) = configuration.volumeSize.x / 2 * configuration.voxelScale;
        Tgk_Matrix(1, 3) = configuration.volumeSize.y / 2 * configuration.voxelScale;
        Tgk_Matrix(2, 3) = configuration.volumeSize.z / 2 * configuration.voxelScale - configuration.initDepth;
}


bool Scheduler::process_new_frame(const cv::Mat& depth_map, const cv::Mat& color_map){
    //预处理，转换格式，生成金字塔，双边滤波
    FrameData frameData = preProcessor.preProcess(depth_map);
    //生成顶点图和法向量图
    surfaceMeasurement.genVertexAndNormalMap(frameData);
    // //将顶点图保存为点云
    // cv::Mat downloaded_vertex_map;
    // frameData.vertex_pyramid[0].download(downloaded_vertex_map);
    // utils::vertexMapToPointCloudAndSave(downloaded_vertex_map);
    // return true;
    frameData.color_pyramid[0].upload(color_map);

    bool icp_success { true };
    if (this->frameId > 0) { // Do not perform ICP for the very first frame
        // 不在第一帧进行位姿估计
        icp_success = this->poseEstimator.pose_estimation(
            this->Tgk_Matrix,                               // 输入: 上一帧的相机位姿; 输出: 当前帧得到的相机位姿
            frameData,                                     // 当前帧的彩色图/深度图/顶点图/法向图数据
            this->predictionResult,                        // 上一帧图像输入后, 推理出的平面模型，使用顶点图、法向图来表示
            cameraParameters,                              // 相机内参
            configuration.numLevels,                       // 金字塔层数
            configuration.distanceThreshold,               // icp 匹配过程中视为 outlier 的距离差
            configuration.angleThreshold,                  // icp 匹配过程中视为 outlier 的角度差 (deg)
            configuration.icpIterations);                  // icp 过程的迭代次数
    }
    // 如果 icp 过程不成功, 那么就说明当前失败了
    if (!icp_success){
        std::cout << "ICP失败" <<std::endl;
        // icp失败之后本次处理退出,但是上一帧推理的得到的平面将会一直保持, 每次新来一帧都会重新icp后一直都在尝试重新icp, 尝试重定位回去
        return false;
    }
   

    this->surfaceReconstructor.surface_reconstruction(
            frameData.depth_pyramid[0],                        // 金字塔底层的深度图像
            frameData.color_pyramid[0],                        // 金字塔底层的彩色图像
            cameraParameters,                                  // 相机内参
            configuration.truncationDistance,                  // 截断距离u
            this->Tgk_Matrix.inverse());
    std::cout << "重建完成" <<std::endl;
    
    for (int level = 0; level < configuration.numLevels; ++level)
        // 对每层图像的数据都进行表面的推理
        this->surfacePredictor.surface_prediction(
            surfaceReconstructor.getModelData(),              // Global Volume
            predictionResult.vertex_pyramid[level],               // 推理得到的平面的顶点图
            predictionResult.normal_pyramid[level],               // 推理得到的平面的法向图 
            predictionResult.color_pyramid[level],                // 推理得到的彩色图
            cameraParameters.level(level),                 // 当前图层的相机内参
            configuration.truncationDistance,              // 截断距离
            this->Tgk_Matrix);
    std::cout << "RayCasting完成" <<std::endl;

    this->frameId ++;
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
        config::ANGLE_THRESHOLD,
        config::VOXEL_SCALE,
        config::TRUNCATION_DISTANCE,
        config::INIT_DEPTH,
        config::DEPTH_SCALE
    );

    //初始化构造器
    PreProcessor preProcessor(cameraParameters,configuration);
    SurfaceMeasurement surfaceMeasurement(cameraParameters,configuration);

    PoseEstimator poseEstimator(cameraParameters,configuration);
    SurfaceReconstructor surfaceReconstructor(cameraParameters,configuration);
    SurfacePredictor surfacePredictor(cameraParameters,configuration);
    Scheduler scheduler(cameraParameters,configuration,preProcessor,surfaceMeasurement,
                        poseEstimator,surfaceReconstructor,surfacePredictor);
    return scheduler;
}