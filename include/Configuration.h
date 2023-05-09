//定义参数使用
#pragma once
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

namespace config{
    //相机参数
    constexpr int IMAGE_WIDTH = 640;        //图像宽度
    constexpr int IMAGE_HEIGHT = 480;       //图像高度
    constexpr float FOCAL_X = 525.0;        //焦距x
    constexpr float FOCAL_Y = 525.0;        //焦距y
    constexpr float PRINCIPAL_X = 319.5;    //光心x
    constexpr float PRINCIPAL_Y = 239.5;    //光心y

    //生成金字塔参数
    constexpr int NUM_LEVELS = 3;//生成金字塔的层数
    //滤波参数
    // kernel_size：5或7。较大的核可能会导致较大的计算成本，但也可能提供更好的滤波效果。
    // color_sigma：取决于深度图像的数值范围。通常，可以从范围的10%到30%开始尝试。
    // spatial_sigma：取决于图像的尺寸和噪声水平。可以从5到15的范围内尝试不同的值。  
    constexpr int KERNAL_SIZE = 5;// 双边滤波器使用的窗口（核）大小
    constexpr float DEPTH_CUTOFF = 1000.f;//截断深度
    constexpr float COLOR_SIGMA = 1.f;// 值域滤波的方差
    constexpr float SPATIAL_SIGMA = 1.f;// 空间域滤波的方差

    //ICP配准参数
    // distance_threshold：这个参数表示在 ICP 匹配过程中视为外点的距离差。匹配点对之间的距离如果大于这个阈值，它们将被视为外点并从配准过程中排除。
    //                     这有助于减小错误匹配点对对配准结果的影响。推荐取值范围通常在 0.01 到 0.1 米之间，具体取值需要根据应用场景和传感器的精度来确定。
    // angle_threshold：这个参数表示在 ICP 匹配过程中视为外点的角度差（以度为单位）。
                            // 匹配点对之间的法向量夹角如果大于这个阈值，它们将被视为外点并从配准过程中排除。这同样有助于减小错误匹配点对对配准结果的影响。
                            // 推荐取值范围通常在 1 到 30 度之间，具体取值需要根据应用场景和传感器的精度来确定。
    // icp_iterations：这个参数表示 ICP 配准过程的迭代次数。增加迭代次数可能会提高配准精度，但也会增加计算时间。
                            // 推荐的取值范围通常在 5 到 50 之间，具体取值需要根据实时性要求和计算资源来确定。
    constexpr float DISTANCE_THRESHOLD { 200.f };
    constexpr float ANGLE_THRESHOLD { 50.f };

    constexpr float VOXEL_SCALE = 2.f;
    constexpr float TRUNCATION_DISTANCE = 25.f;
    constexpr float INIT_DEPTH = 1000.f;
    constexpr float DEPTH_SCALE = 5000; //深度图的缩放尺度
    
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
        float distanceThreshold; // ICP 匹配过程中视为外点的距离差
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