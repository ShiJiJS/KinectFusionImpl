#include "../Configuration.h"
#include "../DataTypes.h"
#include <Eigen/Eigen>
using config::CameraParameters;
using config::GlobalConfiguration;

//预处理器
//负责完成图片类型的转换，金字塔生成，以及双边滤波
namespace step{

    class SurfaceReconstructor{
    public:
        SurfaceReconstructor(CameraParameters cameraParameters,GlobalConfiguration globalConfiguration);
        void surface_reconstruction(const cv::cuda::GpuMat& depth_image, const cv::cuda::GpuMat& color_image,
                                  const CameraParameters& cam_params, const float truncation_distance,
                                  const Eigen::Matrix4f& model_view);
        ModelData& getModelData(){return this->modelData;}
    private:
      CameraParameters cameraParameters;
      GlobalConfiguration globalConfiguration;
      ModelData modelData;//模型数据，存储tsdf和颜色值
    };
    
}