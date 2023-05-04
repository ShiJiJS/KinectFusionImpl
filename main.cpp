#include "pipeline.h"
#include "utils.h"
#include "common.h"
#include "data_types.h"

using kinectfusion::Pipeline;
using kinectfusion::CameraParameters;
using kinectfusion::GlobalConfiguration;

int main() {
    
    // Pipeline pipeline();
    
    // std::string index_file = "depth.txt";
    // std::vector<std::string> depth_image_paths = kinectfusion::utils::read_depth_image_paths(index_file);
    
    // for (const auto& path : depth_image_paths) {
    //     std::cout << path << std::endl;
    // }
    
    //设置参数
    CameraParameters parameters(
        kinectfusion::config::IMAGE_WIDTH,
        kinectfusion::config::IMAGE_HEIGHT,
        kinectfusion::config::FOCAL_X,
        kinectfusion::config::FOCAL_Y,
        kinectfusion::config::PRINCIPAL_X,
        kinectfusion::config::PRINCIPAL_Y
    );

    GlobalConfiguration configuration(
        kinectfusion::config::DEPTH_CUTOFF,
        kinectfusion::config::KERNAL_SIZE,
        kinectfusion::config::COLOR_SIGMA,
        kinectfusion::config::SPATIAL_SIGMA,
        kinectfusion::config::NUM_LEVELS,
        kinectfusion::config::DISTANCE_THRESHOLD,
        kinectfusion::config::ANGLE_THRESHOLD
    );

    // 读入深度图
    cv::Mat depth_image = cv::imread("depth_image.png", cv::IMREAD_UNCHANGED);
    if (depth_image.empty()) {
        std::cerr << "Failed to read depth image!" << std::endl;
        return 1;
    }

    // 读入 RGB 图像
    cv::Mat rgb_image = cv::imread("rgb_image.png", cv::IMREAD_COLOR);
    if (rgb_image.empty()) {
        std::cerr << "Failed to read RGB image!" << std::endl;
        return 1;
    }


    Pipeline pipeline(parameters,configuration);
    pipeline.process_frame(depth_image,rgb_image);
    return 0;
}