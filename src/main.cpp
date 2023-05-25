#include "../include/Scheduler.h"
#include "../include/Configuration.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>

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

    std::string base_path = "./";
    std::vector<TimestampedImage> depth_images = read_timestamps_and_filenames(base_path + "depth.txt");
    std::vector<TimestampedImage> rgb_images = read_timestamps_and_filenames(base_path + "rgb.txt");
    std::cout << "Image sequences found." << std::endl;

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();

    size_t i = 0;

    for (i = 0; i < depth_images.size() && i < rgb_images.size(); ++i) {
        // 读取深度图像
        cv::Mat depth_image_16U = cv::imread(base_path + depth_images[i].filename, cv::IMREAD_UNCHANGED);
        cv::Mat rgb_image = cv::imread(base_path + rgb_images[i].filename, cv::IMREAD_COLOR);

        if (depth_image_16U.empty() || rgb_image.empty()) {
            std::cerr << "Failed to load image: " << depth_images[i].filename << " or " << rgb_images[i].filename << std::endl;
            continue;
        }
        
        std::cout << "Successfully read image " << i << "." << std::endl;
        bool success = scheduler.process_new_frame(depth_image_16U, rgb_image);

        if (!success) {
            std::cerr << "Failed to process frame: " << depth_images[i].filename << " and " << rgb_images[i].filename << std::endl;
        }
    }



    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();

    // 计算执行时间
    std::chrono::duration<double> duration = end - start;

    // 将执行时间转换为毫秒
    double milliseconds = duration.count() * 1000;

    // 输出执行时间
    std::cout << "重建总消耗时间: " << milliseconds << " 毫秒" << std::endl;
    // 计算平均每帧时间花费
    std::cout << "平均每帧花费时间: " << milliseconds * 1.0 / (i + 1) << " 毫秒" << std::endl;
    //FPS
    std::cout << "每秒平均生成帧: " << 1000.0 / (milliseconds * 1.0 / (i + 1))<< "帧" << std::endl;

    std::cout << "已处理完所有图片，正在导出点云" << std::endl;
    scheduler.extract_and_save_pointcloud();
    std::cout << "点云已保存至output.pcd 和 output.ply" << std::endl;
    system("pause");

    return 0;
}