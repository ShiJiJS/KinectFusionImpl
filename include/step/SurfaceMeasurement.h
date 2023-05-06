
#include "../DataTypes.h"
#include "../Configuration.h"

using config::CameraParameters;
using config::GlobalConfiguration;

namespace step {
    class SurfaceMeasurement{
    public:
        SurfaceMeasurement(CameraParameters cameraParameters,GlobalConfiguration globalConfiguration);
        //生成顶点图和法向量图，返回值表示成功与否
        void genVertexAndNormalMap(FrameData &frameData);
    private:
        CameraParameters cameraParameters;
        GlobalConfiguration configuration;
    };


    namespace kernel {
        /**
             * @brief 计算某层深度图像的顶点图
             * @param[in]  depth_map        某层滤波后的深度图
             * @param[out] vertex_map       计算得到的顶点图
             * @param[in]  depth_cutoff     不考虑的过远的点的距离
             * @param[in]  cam_params       该层图像下的相机内参
             */
            void compute_vertex_map(const GpuMat& depth_map, GpuMat& vertex_map, const float depth_cutoff,
                                    const CameraParameters cam_params);
            /**
             * @brief 根据某层顶点图计算法向图
             * @param[in]  vertex_map       某层顶点图
             * @param[out] normal_map       计算得到的法向图
             */
            void compute_normal_map(const GpuMat& vertex_map, GpuMat& normal_map);
    }
}