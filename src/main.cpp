#include "../include/Scheduler.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

struct TimestampedImage {
    double timestamp;
    std::string filename;
};

std::vector<TimestampedImage> read_timestamps_and_filenames(const std::string& filepath) {
    std::vector<TimestampedImage> result;
    std::ifstream file(filepath);
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        TimestampedImage entry;
        iss >> entry.timestamp >> entry.filename;

        result.push_back(entry);
    }

    return result;
}



int main(){
    Scheduler scheduler = SchedulerFactory::build();


    std::string base_path = "/home/shiji/KinectFusionImpl/build/";
    std::vector<TimestampedImage> depth_images = read_timestamps_and_filenames(base_path + "depth.txt");
    std::vector<TimestampedImage> rgb_images = read_timestamps_and_filenames(base_path + "rgb.txt");

    for (size_t i = 0; i < depth_images.size() && i < rgb_images.size(); ++i) {
        // 读取深度图像
        cv::Mat depth_image_16U = cv::imread(base_path + depth_images[i].filename, cv::IMREAD_UNCHANGED);
        cv::Mat rgb_image = cv::imread(base_path + rgb_images[i].filename, cv::IMREAD_COLOR);

        if (depth_image_16U.empty() || rgb_image.empty()) {
            std::cerr << "Failed to load image: " << depth_images[i].filename << " or " << rgb_images[i].filename << std::endl;
            continue;
        }

        bool success = scheduler.process_new_frame(depth_image_16U, rgb_image);

        if (!success) {
            std::cerr << "Failed to process frame: " << depth_images[i].filename << " and " << rgb_images[i].filename << std::endl;
        }
    }

    return 0;
}