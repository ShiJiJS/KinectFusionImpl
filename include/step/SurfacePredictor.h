#include "../Configuration.h"
#include "../DataTypes.h"
#include <Eigen/Eigen>
using config::CameraParameters;
using config::GlobalConfiguration;

//预处理器
//负责完成图片类型的转换，金字塔生成，以及双边滤波
namespace step{

    class SurfacePredictor{
    public:
        SurfacePredictor(CameraParameters cameraParameters,GlobalConfiguration globalConfiguration);
        void surface_prediction(
                const ModelData& volume,                   // Global Volume
                GpuMat& model_vertex,                       // 推理得到的顶点图
                GpuMat& model_normal,                       // 推理得到的法向图
                GpuMat& model_color,                        // 推理得到的颜色
                const CameraParameters& cam_parameters,     // 当前图层的相机内参
                const float truncation_distance,            // 截断距离
                const Eigen::Matrix4f& pose);               // 当前帧的相机位姿
    private:
        CameraParameters cameraParameters;
        GlobalConfiguration globalConfiguration;
    };
    
}