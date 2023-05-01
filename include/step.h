#include "data_types.h"

#ifndef STEP_H
#define STEP_H

namespace kinectfusion{
  namespace step{
    FrameData surface_measurement(const cv::Mat& input_frame,               //输入的原始深度图像 CV32F
                                  const CameraParameters& camera_params,    //相机内参矩阵参数
                                  const size_t num_levels,                  //生成金字塔的层数
                                  const float depth_cutoff,                 //截断深度
                                  const int kernel_size,                   // 双边滤波器使用的窗口（核）大小
                                  const float color_sigma,                  // 值域滤波的方差
                                  const float spatial_sigma);               // 空间域滤波的方差

    bool pose_estimation(Eigen::Matrix4f& pose,
                         const FrameData& frame_data,
                         const PredictionResult& model_data,
                         const CameraParameters& cam_params,
                         const int pyramid_height,
                         const float distance_threshold, const float angle_threshold,
                         const std::vector<int>& iterations);
  }
  
}

#endif //KINECTFUSION_H