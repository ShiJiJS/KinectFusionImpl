//定义参数使用
#pragma once
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

namespace config{  
    // 迭代次数，即第一层迭代10次,第二层5次,第三层4次
    enum ICP_ITERATIONS{
        FIRST = 10,
        SECOND = 5,
        THIRD = 4};

    struct CameraParameters {
        int image_width, image_height;
        float focal_x, focal_y;
        float principal_x, principal_y;

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
        float depthCutoff; //截断深度
        int kernalSize; //双边滤波器的窗口（核）的大小
        float colorSigma; //值域滤波的方差
        float spatialSigma; //空间域滤波的方差
        int numLevels; //生成金字塔的层数
        double distanceThreshold; // ICP 匹配过程中视为外点的距离差
        float angleThreshold; //匹配过程中视为外点的角度差（以度为单位）
        float voxelScale;
        float truncationDistance;
        float initDepth;
        float depthScale;
        std::vector<int> icpIterations {10, 5, 4}; //迭代次数,即第一层迭代10次,第二层5次,第三层4次
        int3 volumeSize { make_int3(512, 512, 512) };//tsdf的尺寸

        GlobalConfiguration(float _depthCutoff, int _kernalSize, float _colorSigma, 
        float _spatialSigma, int _numLevels, float _distanceThreshold, 
        float _angleThreshold,float voxelScale,float truncationDistance,float initDepth,float depthScale)
        : depthCutoff(_depthCutoff), kernalSize(_kernalSize), colorSigma(_colorSigma), spatialSigma(_spatialSigma),
        numLevels(_numLevels), distanceThreshold(_distanceThreshold), angleThreshold(_angleThreshold),
        voxelScale(voxelScale),truncationDistance(truncationDistance),initDepth(initDepth),depthScale(depthScale) {};
    };
}