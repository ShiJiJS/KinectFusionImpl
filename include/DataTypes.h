#pragma once
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/core.hpp>
#include "Configuration.h"
// #include <cuda_runtime.h>
#include <vector_types.h>

using cv::cuda::GpuMat;
using config::CameraParameters;

struct FrameData {
    std::vector<GpuMat> depth_pyramid;                              // 原始深度图的金字塔
    std::vector<GpuMat> smoothed_depth_pyramid;                     // 滤波后的深度图金字塔
    std::vector<GpuMat> color_pyramid;
    std::vector<GpuMat> vertex_pyramid;                             // 3D点金字塔
    std::vector<GpuMat> normal_pyramid;                             // 法向量金字塔

    // frame data
    FrameData(const size_t pyramid_height,CameraParameters cameraParameters) :
            depth_pyramid(pyramid_height), smoothed_depth_pyramid(pyramid_height),
            vertex_pyramid(pyramid_height), normal_pyramid(pyramid_height),color_pyramid(pyramid_height){
        //为金字塔的每一层分配数据
        for (int level = 0; level < pyramid_height; ++level) {
            //获取图像大小
            const int width = cameraParameters.level(level).image_width;
            const int height = cameraParameters.level(level).image_height;
            //分配每个金字塔"图像"的存储空间
            //该部分内存由GPUMat自动管理。无需手动释放
            this->depth_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC1);
            this->smoothed_depth_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC1);
            this->vertex_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC3);
            this->normal_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC3);
            this->color_pyramid[level] = cv::cuda::createContinuous(height, width, CV_8UC3);
        }
    };
};


struct ModelData {
    // OpenCV 提供的在GPU上的图像数据类型
    GpuMat tsdfVolume; //short2
    GpuMat colorVolume; //uchar4
    int3 volumeSize;
    float voxelScale;
    // 构造函数
    ModelData(const int3 volumeSize, const float voxelScale) :
            // 注意 TSDF 是2通道的, 意味着其中一个通道存储TSDF函数值, 另外一个通道存储其权重
            tsdfVolume(cv::cuda::createContinuous(volumeSize.y * volumeSize.z, volumeSize.x, CV_16SC2)),
            colorVolume(cv::cuda::createContinuous(volumeSize.y * volumeSize.z, volumeSize.x, CV_8UC3)),
            volumeSize(volumeSize), voxelScale(voxelScale)
    {
        // 全部清空
        tsdfVolume.setTo(0);
        colorVolume.setTo(0);
    }
};


struct PredictionResult {
    std::vector<GpuMat> vertex_pyramid;                     // 三维点的金字塔
    std::vector<GpuMat> normal_pyramid;                     // 法向量的金字塔
    std::vector<GpuMat> color_pyramid;

     // 构造函数
    PredictionResult(const size_t pyramid_height, const CameraParameters camera_parameters) :
            // 初始化三个"图像"金字塔的高度
            vertex_pyramid(pyramid_height), normal_pyramid(pyramid_height),color_pyramid(pyramid_height)
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
            color_pyramid[level] =
                            cv::cuda::createContinuous(camera_parameters.level(level).image_height,
                                                       camera_parameters.level(level).image_width,
                                                       CV_8UC3);
            // 然后清空为0
            vertex_pyramid[level].setTo(0);
            normal_pyramid[level].setTo(0);
        }// 遍历每一层金字塔
    }
};



struct CloudData {
    GpuMat vertices;
    GpuMat normals;
    GpuMat color;
    cv::Mat host_vertices;
    cv::Mat host_normals;
    cv::Mat host_color;
    int* point_num;
    int host_point_num;
    explicit CloudData(const int max_number) :
            vertices{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
            normals{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
            color{cv::cuda::createContinuous(1, max_number, CV_8UC3)},
            host_vertices{}, host_normals{}, host_color{}, point_num{nullptr}, host_point_num{}
    {
        vertices.setTo(0.f);
        normals.setTo(0.f);
        color.setTo(0.f);
        cudaMalloc(&point_num, sizeof(int));
        cudaMemset(point_num, 0, sizeof(int));
    }
    // No copying
    CloudData(const CloudData&) = delete;
    CloudData& operator=(const CloudData& data) = delete;
    void download()
    {
        vertices.download(host_vertices);
        normals.download(host_normals);
        color.download(host_color);
        cudaMemcpy(&host_point_num, point_num, sizeof(int), cudaMemcpyDeviceToHost);
    }
};
