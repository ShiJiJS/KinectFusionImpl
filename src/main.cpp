#include "../include/Scheduler.h"

int main(){
    Scheduler scheduler = SchedulerFactory::build();

     // 读入深度图
    cv::Mat depth_image = cv::imread("/home/shiji/KinectFusionImpl/build/depth_image.png", cv::IMREAD_UNCHANGED);
    if (depth_image.empty()) {
        std::cerr << "Failed to read depth image!" << std::endl;
        return 1;
    }

    // 读入 RGB 图像
    cv::Mat rgb_image = cv::imread("/home/shiji/KinectFusionImpl/build/rgb_image.png", cv::IMREAD_COLOR);
    if (rgb_image.empty()) {
        std::cerr << "Failed to read RGB image!" << std::endl;
        return 1;
    }
    
    scheduler.process_new_frame(depth_image,rgb_image);
    return 0;
}