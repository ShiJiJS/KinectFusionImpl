
#include "data_types.h"

#ifndef PIPELINE_H
#define PIPELINE_H


namespace kinectfusion{
    class Pipeline{
    private:
      CameraParameters camera_parameters; //相机参数
      GlobalConfiguration configuration;  //全局设置
      int frame_id; //帧的id
      Eigen::Matrix4f Tgk_Matrix; //相机位姿的转移矩阵
      PredictionResult prediction_result; //来自上一帧surface prediction过程生成的结果
    public:
      Pipeline(const CameraParameters _camera_parameters,GlobalConfiguration _configuration);
    
      //添加新一帧的信息
      //depth_map 为16U1C的深度图，color_map为8U3C的RGB图
      bool process_frame(const cv::Mat& depth_map, const cv::Mat& color_map);
    };
}


#endif