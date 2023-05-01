#include "../include/pipeline.h"
#include "../include/step.h"
#include "../include/utils.h"




//pipeline.h的实现
namespace kinectfusion{
    Pipeline::Pipeline(const CameraParameters _camera_parameters, const GlobalConfiguration _configuration)
        : camera_parameters(_camera_parameters), configuration(_configuration), //显式初始化，无需再提供默认构造函数
            prediction_result(this->configuration.num_levels,this->camera_parameters){ 
        this->frame_id = 0;//初始化frame_id
        //初始化Tgk为单位矩阵
        this->Tgk_Matrix = Eigen::Matrix4f::Identity();
        // The pose starts in the middle of the cube, offset along z by the initial depth
        // 第一帧的相机位姿设置在 Volume 的中心, 然后在z轴上拉远一点
        Tgk_Matrix(0, 3) = 512 / 2 * 2.f;
        Tgk_Matrix(1, 3) = 512 / 2 * 2.f;
        Tgk_Matrix(2, 3) = 512 / 2 * 2.f - 1000.f;
    }

    bool Pipeline::process_frame(const cv::Mat& depth_map, const cv::Mat& color_map){
        //传入的为16UC1的深度图，需要先转为32FC1
        cv::Mat depth_map_32f;
        depth_map.convertTo(depth_map_32f, CV_32F);
        
        //步骤1 surface measurement
        //传入深度图，返回生成的顶点图和法向量图
        FrameData frame_data = step::surface_measurement(
            depth_map_32f,
            this->camera_parameters,
            this->configuration.num_levels,
            this->configuration.depth_cutoff,
            this->configuration.kernal_size,
            this->configuration.color_sigma,
            this->configuration.spatial_sigma);

        // 将原始彩色图上传到显存中
        frame_data.color_pyramid[0].upload(color_map);

        //步骤2 Pose estimation
        //表示icp过程是否成功的变量
        bool icp_success { true };
        if (frame_id > 0) { // Do not perform ICP for the very first frame
            // 不在第一帧进行位姿估计
            icp_success = step::pose_estimation(
                this->Tgk_Matrix,                               // 输入: 上一帧的相机位姿; 输出: 当前帧得到的相机位姿
                frame_data,                                     // 当前帧的彩色图/深度图/顶点图/法向图数据
                this->prediction_result,                        // 上一帧图像输入后, 推理出的平面模型，使用顶点图、法向图来表示
                camera_parameters,                              // 相机内参
                configuration.num_levels,                       // 金字塔层数
                configuration.distance_threshold,               // icp 匹配过程中视为 outlier 的距离差
                configuration.angle_threshold,                  // icp 匹配过程中视为 outlier 的角度差 (deg)
                configuration.icp_iterations);                  // icp 过程的迭代次数
        }
        
        // 如果 icp 过程不成功, 那么就说明当前失败了
        if (!icp_success)
            // icp失败之后本次处理退出,但是上一帧推理的得到的平面将会一直保持, 每次新来一帧都会重新icp后一直都在尝试重新icp, 尝试重定位回去
            return false;
        // 记录当前帧的位姿
        // poses.push_back(current_pose);
        if(this->frame_id > 0){
            std::cout << "aaaaaaaa" << std::endl;
            std::cout << this->Tgk_Matrix << std::endl;
        }

        if(this->frame_id == 0){
            std::cout << "bbbbbbbbbb" << std::endl;
            this->prediction_result.vertex_pyramid = frame_data.vertex_pyramid;
            this->prediction_result.normal_pyramid = frame_data.normal_pyramid;
        }
        frame_id ++;
        return true;
        


    }

}

