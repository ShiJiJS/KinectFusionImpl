
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#ifndef DATA_TYPES_H
#define DATA_TYPES_H

using cv::cuda::GpuMat;

namespace kinectfusion{
    struct FrameData {
        std::vector<GpuMat> depth_pyramid;                              // 原始深度图的金字塔
        std::vector<GpuMat> smoothed_depth_pyramid;                     // 滤波后的深度图金字塔
        std::vector<GpuMat> color_pyramid;                              // 彩色图像的金字塔

        std::vector<GpuMat> vertex_pyramid;                             // 3D点金字塔
        std::vector<GpuMat> normal_pyramid;                             // 法向量金字塔

        // frame data
        FrameData(const size_t pyramid_height) :
                depth_pyramid(pyramid_height), smoothed_depth_pyramid(pyramid_height),
                color_pyramid(pyramid_height), vertex_pyramid(pyramid_height), normal_pyramid(pyramid_height){ };
    };


    struct CameraParameters {
        int image_width, image_height;
        float focal_x, focal_y;
        float principal_x, principal_y;

        // CameraParameters(int image_width,int image_height,float focal_x,float focal_y,float principal_x,float principal_y){
        //     this->image_width = image_width;
        //     this->image_height = image_height;
        //     this->focal_x = focal_x;
        //     this->focal_y = focal_y;
        //     this->principal_x = principal_x;
        //     this->principal_y = principal_y;
        // }

        // construct function
        CameraParameters(int _image_width, int _image_height, float _focal_x, float _focal_y, float _principal_x, float _principal_y)
        : image_width(_image_width), image_height(_image_height), focal_x(_focal_x), focal_y(_focal_y), principal_x(_principal_x), principal_y(_principal_y) {}

        // 根据不同图层返回不同CameraParameters数据用
        CameraParameters level(const size_t level) const
        {
            if (level == 0) return *this;
            //计算一下缩放的标度
            const float scale_factor = powf(0.5f, static_cast<float>(level));
            return CameraParameters { image_width >> level, image_height >> level,          // 左移。将图像的宽度和高度分别除以 2 的 level 次幂
                                      focal_x * scale_factor, focal_y * scale_factor,
                                      (principal_x + 0.5f) * scale_factor - 0.5f,      //首先将主点坐标加上0.5，然后乘以缩放因子，再减去0.5。这个计算方法尽可能利用了四舍五入的性质，使得在缩放图像时，主点坐标仍然保持正确的位置     
                                      (principal_y + 0.5f) * scale_factor - 0.5f };         
        }
    };


    struct GlobalConfiguration {
        // common.h
        float depth_cutoff;//截断深度
        int kernal_size; //双边滤波器的窗口（核）的大小
        float color_sigma; //值域滤波的方差
        float spatial_sigma; //空间域滤波的方差
        int num_levels; //生成金字塔的层数
        float distance_threshold;// ICP 匹配过程中视为外点的距离差
        float angle_threshold; //匹配过程中视为外点的角度差（以度为单位）
        std::vector<int> icp_iterations {10, 5, 4}; //迭代次数,即第一层迭代10次,第二层5次,第三层4次

        GlobalConfiguration(float _depth_cutoff, int _kernal_size, float _color_sigma, float _spatial_sigma, int _num_levels,float _distance_threshold,float _angle_threshold)
        : depth_cutoff(_depth_cutoff), kernal_size(_kernal_size), color_sigma(_color_sigma), spatial_sigma(_spatial_sigma), 
        num_levels(_num_levels),distance_threshold(_distance_threshold),angle_threshold(_angle_threshold){};
    };


    struct PredictionResult {
        std::vector<GpuMat> vertex_pyramid;                     // 三维点的金字塔
        std::vector<GpuMat> normal_pyramid;                     // 法向量的金字塔

         // 构造函数
        PredictionResult(const size_t pyramid_height, const CameraParameters camera_parameters) :
                // 初始化三个"图像"金字塔的高度
                vertex_pyramid(pyramid_height), normal_pyramid(pyramid_height)
        {
            // 遍历每一层金字塔
            for (size_t level = 0; level < pyramid_height; ++level) {
                // 分配内存
                vertex_pyramid[level] =
                        cv::cuda::createContinuous(camera_parameters.level(level).image_height,
                                                   camera_parameters.level(level).image_width,
                                                   CV_32FC3);
                normal_pyramid[level] =
                        cv::cuda::createContinuous(camera_parameters.level(level).image_height,
                                                   camera_parameters.level(level).image_width,
                                                   CV_32FC3);
                // 然后清空为0
                vertex_pyramid[level].setTo(0);
                normal_pyramid[level].setTo(0);
            }// 遍历每一层金字塔
        }
    };
}


#endif