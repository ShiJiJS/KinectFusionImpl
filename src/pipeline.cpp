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

        cv::Mat downloaded_vertex_map;
        frame_data.vertex_pyramid[0].download(downloaded_vertex_map);
        utils::vertexMapToPointCloudAndSave(downloaded_vertex_map);

        return true;
        


    }

}

