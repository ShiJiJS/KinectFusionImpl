
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
    //std::cout << "Preprocess done." << std::endl;
    //生成顶点图和法向量图
    surfaceMeasurement.genVertexAndNormalMap(frameData);
    //std::cout << "Surface Measurement done." << std::endl;
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
        std::cout << "ICP Failed." <<std::endl;
        // icp失败之后本次处理退出,但是上一帧推理的得到的平面将会一直保持, 每次新来一帧都会重新icp后一直都在尝试重新icp, 尝试重定位回去
        return false;
    }
    //将每次迭代的位姿结果保存
    this->pose_history.push_back(this->Tgk_Matrix);

    this->surfaceReconstructor.surface_reconstruction(
            frameData.depth_pyramid[0],                        // 金字塔底层的深度图像
            frameData.color_pyramid[0],                        // 金字塔底层的彩色图像
            cameraParameters,                                  // 相机内参
            configuration.truncationDistance,                  // 截断距离u
            this->Tgk_Matrix.inverse());
    //std::cout << "Reconstruct complete." <<std::endl;
    
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
    //std::cout << "Raycasting done." <<std::endl;
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
Scheduler SchedulerFactory::build(){
    //设置参数

    // create a map of strings and floats
    std::map<std::string, float> parameters;

    // create an input file stream
    std::ifstream infile("./config.ini");

    // check if the file is opened
    if (infile.is_open()) {
        // create a string variable to store each line
        std::string line;

        // loop through each line in the file
        while (std::getline(infile, line)) {
            // create a string variable to store the parameter name
            std::string name;

            // create a float variable to store the parameter value
            float value;

            // use string::find() to find the position of '='
            size_t pos = line.find('=');

            // use string::substr() to extract the parameter name
            name = line.substr(0, pos - 1);

            // use string::substr() to extract the parameter value
            std::string valstr = line.substr(pos + 1);

            // check if the parameter value has a decimal point
            if (valstr.find('.') != std::string::npos) {
                // convert it to a float using std::stof()
                value = std::stof(valstr);
            }
            else {
                // convert it to an int using std::stoi()
                value = std::stoi(valstr);
            }

            // insert the parameter name and value into the map
            parameters.insert(std::make_pair(name, value));
        }

        // close the file
        infile.close();
    }
    else {
        // print an error message
        std::cout << "Unable to open file\n";
    }

    // print the map contents
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