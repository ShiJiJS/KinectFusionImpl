
#include "../include/Scheduler.h"
#include "../include/Configuration.h"
#include "../include/DataTypes.h"
#include "../include/Utils.h"

using utils::readConfigFile;

Scheduler::Scheduler(CameraParameters cameraParameters, GlobalConfiguration configuration, PreProcessor 
preprocessor,SurfaceMeasurement surfaceMeasurement,PoseEstimator poseEstimator,
SurfaceReconstructor surfaceReconstructor,SurfacePredictor surfacePredictor)
    : cameraParameters(cameraParameters), configuration(configuration), preProcessor(preprocessor), 
    surfaceMeasurement(surfaceMeasurement),poseEstimator(poseEstimator),
    predictionResult(configuration.numLevels,cameraParameters),surfaceReconstructor(surfaceReconstructor),
    surfacePredictor(surfacePredictor),frameId(0){
        //初始化Tgk为单位矩阵
        this->Tgk_Matrix = Eigen::Matrix4f::Identity();
        // 第一帧的相机位姿设置在 Volume 的中心, 然后在z轴上拉远一点
        Tgk_Matrix(0, 3) = configuration.volumeSize.x / 2 * configuration.voxelScale;
        Tgk_Matrix(1, 3) = configuration.volumeSize.y / 2 * configuration.voxelScale;
        Tgk_Matrix(2, 3) = configuration.volumeSize.z / 2 * configuration.voxelScale - configuration.initDepth;
}


bool Scheduler::process_new_frame(const cv::Mat& depth_map, const cv::Mat& color_map){
    //预处理，转换格式，生成金字塔，双边滤波
    FrameData frameData = preProcessor.preProcess(depth_map);
    //std::cout << "Preprocess done." << std::endl;
    //生成顶点图和法向量图
    surfaceMeasurement.genVertexAndNormalMap(frameData);
    //将对应的颜色图也上传到gpu中
    frameData.color_pyramid[0].upload(color_map);

    //pose estimation过程
    bool icp_success { true };
    if (this->frameId > 0) {
        // 不在第一帧进行位姿估计
        icp_success = this->poseEstimator.pose_estimation(
            this->Tgk_Matrix,               
            frameData,                      
            this->predictionResult,         
            cameraParameters,               
            configuration.numLevels,        
            configuration.distanceThreshold,
            configuration.angleThreshold,   
            configuration.icpIterations);   
    }
    // 如果 icp 过程不成功, 直接返回不执行后续过程
    if (!icp_success){
        std::cout << "ICP Failed." <<std::endl;
        return false;
    }
    //将每次迭代的位姿结果保存
    this->pose_history.push_back(this->Tgk_Matrix);


    //Surface Reconstruction过程
    this->surfaceReconstructor.surface_reconstruction(
            frameData.depth_pyramid[0],         
            frameData.color_pyramid[0],         
            cameraParameters,                   
            configuration.truncationDistance,   
            this->Tgk_Matrix.inverse());
    
    //Surface Prediction过程
    for (int level = 0; level < configuration.numLevels; ++level)
        this->surfacePredictor.surface_prediction(
            surfaceReconstructor.getModelData(),    
            predictionResult.vertex_pyramid[level], 
            predictionResult.normal_pyramid[level], 
            predictionResult.color_pyramid[level],  
            cameraParameters.level(level),          
            configuration.truncationDistance,       
            this->Tgk_Matrix);
    std::cout << "All steps succeeded." << std::endl;
    this->frameId ++;
    return true;

}

//保存位姿信息
void Scheduler::save_poses(){
    std::ofstream outfile("poses.txt");
    for(const auto& pose : pose_history) {
        for(int i=0; i<3; ++i) {
            for(int j=0; j<4; ++j) {
                outfile << pose(i,j);
                if(j<3) outfile << " ";
            }
            outfile << "\n";
        }
        outfile << "\n";
    }
    outfile.close();
}


//构造调度器。工厂模式
Scheduler SchedulerFactory::build(std::string configFilePath){
    //设置参数
    std::map<std::string, float> parameters;
    parameters = readConfigFile(configFilePath);
    for (auto pair : parameters) {
        std::cout << pair.first << " = " << pair.second << "\n";
    }

    CameraParameters cameraParameters(
        parameters["IMAGE_WIDTH"],
        parameters["IMAGE_HEIGHT"],
        parameters["FOCAL_X"],
        parameters["FOCAL_Y"],
        parameters["PRINCIPAL_X"],
        parameters["PRINCIPAL_Y"]
    );

    GlobalConfiguration configuration(
        parameters["DEPTH_CUTOFF"],
        parameters["KERNAL_SIZE"],
        parameters["COLOR_SIGMA"],
        parameters["SPATIAL_SIGMA"],
        parameters["NUM_LEVELS"],
        parameters["DISTANCE_THRESHOLD"],
        parameters["ANGLE_THRESHOLD"],
        parameters["VOXEL_SCALE"],
        parameters["TRUNCATION_DISTANCE"],
        parameters["INIT_DEPTH"],
        parameters["DEPTH_SCALE"]
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