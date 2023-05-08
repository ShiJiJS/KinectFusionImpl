
#include "../DataTypes.h"
#include "../Configuration.h"
#include <Eigen/Eigen>
#include "../Matrix.h"

using config::CameraParameters;
using config::GlobalConfiguration;

using Matf31da   = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;

namespace step {
    class PoseEstimator{
    public:
        PoseEstimator(CameraParameters cameraParameters,GlobalConfiguration globalConfiguration);
        //位姿估计
        bool pose_estimation(
            Eigen::Matrix4f& pose,                  // 输入: 上一帧的相机位姿; 输出: 当前帧得到的相机位姿
            const FrameData& frame_data,            // 当前帧中的数据(顶点图+法向图)
            const PredictionResult& model_data,            // 上一帧对Global TSDF Model 进行表面推理得到的表面模型数据(Vertex Map + Normal Map)
            const CameraParameters& cam_params,     // 相机的内参
            const int pyramid_height,               // 金字塔的图层数目
            const float distance_threshold,         // ICP 过程中视为外点的距离阈值
            const float angle_threshold,            // ICP 过程中视为外点的角度阈值
            const std::vector<int>& iterations);     // 每一个图层上的 ICP 迭代次数
    private:
        CameraParameters cameraParameters;
        GlobalConfiguration configuration;
    };


    namespace kernel {
        void estimate_step(
            const Eigen::Matrix3f& rotation_current_eigenMf33,            // 上次迭代得到的旋转 Rwc
            const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_current_eigenMf31,                // 上次迭代得到的平移 twc
            const cv::cuda::GpuMat& vertex_map_current,         // 当前帧对应图层的的顶点图
            const cv::cuda::GpuMat& normal_map_current,         // 当前帧对应图层的的法向图
            const Eigen::Matrix3f& rotation_previous_inv_eigenMf33,       // 上一帧相机外参中的旋转的逆, Rcw
            const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_previous_eigenMf31,               // 上一帧相机的平移 twc
            const CameraParameters& cam_params,                 // 当前图层的相机内参
            const cv::cuda::GpuMat& vertex_map_previous,        // 对应图层的推理得到的平面顶点图
            const cv::cuda::GpuMat& normal_map_previous,        // 对应图层的推理得到的平面法向图
            float distance_threshold,                           // ICP迭代过程中视为外点的距离阈值
            float angle_threshold,                              // ICP迭代过程中视为外点的角度阈值(角度变正弦值)
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A,    // 计算得到的矩阵 A, 行优先
            Eigen::Matrix<double, 6, 1>& b);
    }
}