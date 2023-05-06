#include "../../include/step/SurfaceMeasurement.h"

namespace step{

    SurfaceMeasurement::SurfaceMeasurement(CameraParameters cameraParameters,GlobalConfiguration configuration)
        :cameraParameters(cameraParameters),configuration(configuration){

    }

    void SurfaceMeasurement::genVertexAndNormalMap(FrameData &frameData){
        // 对于每一层图像, 使用GPU计算顶点图和法向图
        for (size_t level = 0; level < configuration.numLevels; ++level) {
            // 根据深度图计算顶点图
            kernel::compute_vertex_map(frameData.smoothed_depth_pyramid[level], frameData.vertex_pyramid[level],
                                     configuration.depthCutoff, cameraParameters.level(level));
            // 根据顶点图来计算法向量
            kernel::compute_normal_map(frameData.vertex_pyramid[level], frameData.normal_pyramid[level]);
        }
    }

}
