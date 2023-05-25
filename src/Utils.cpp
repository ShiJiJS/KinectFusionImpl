#include "../include/Utils.h"

namespace utils{
    //读入深度图索引文件并返回包含文件路径的vector
    std::vector<std::string> read_depth_image_paths(const std::string& index_file) {
        std::vector<std::string> depth_image_paths;
        std::ifstream input_file(index_file);

        if (!input_file.is_open()) {
            std::cerr << "Failed to open the index file!" << std::endl;
            return depth_image_paths;
        }

        std::string line;

        while (std::getline(input_file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }

            std::istringstream line_stream(line);
            double timestamp;
            std::string depth_image_path;

            line_stream >> timestamp >> depth_image_path;
            depth_image_paths.push_back(depth_image_path);
        }

        input_file.close();
        return depth_image_paths;
    }

    //输出mat的类型
    std::string matTypeToString(const cv::Mat& mat) {
        int type = mat.type();
        std::string typeName;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar channels = 1 + (type >> CV_CN_SHIFT);

        switch (depth) {
            case CV_8U:  typeName = "8U"; break;
            case CV_8S:  typeName = "8S"; break;
            case CV_16U: typeName = "16U"; break;
            case CV_16S: typeName = "16S"; break;
            case CV_32S: typeName = "32S"; break;
            case CV_32F: typeName = "32F"; break;
            case CV_64F: typeName = "64F"; break;
            default:     typeName = "Unknown"; break;
        }

        typeName += "C";
        typeName += (channels + '0');
        return typeName;
    }

    // Convert vertex map and normal map to PCL point cloud
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr create_point_cloud(const cv::Mat& vertex_map, const cv::Mat& normal_map) {
       pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

       for (int y = 0; y < vertex_map.rows; ++y) {
           for (int x = 0; x < vertex_map.cols; ++x) {
               pcl::PointXYZRGBNormal point;

               // Set position from vertex map
               cv::Vec3f position = vertex_map.at<cv::Vec3f>(y, x);
               point.x = position[0];
               point.y = position[1];
               point.z = position[2];

               // Set normal from normal map
               cv::Vec3f normal = normal_map.at<cv::Vec3f>(y, x);
               point.normal_x = normal[0];
               point.normal_y = normal[1];
               point.normal_z = normal[2];

               // Set color
               point.r = 255;
               point.g = 255;
               point.b = 255;

               cloud->push_back(point);
           }
       }

       return cloud;
    }

    // void vertexMapToPointCloudAndSave(const cv::Mat& vertex_map) {
    //     // 遍历顶点图的每个像素
    //     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    //     for (int y = 0; y < vertex_map.rows; ++y) {
    //         for (int x = 0; x < vertex_map.cols; ++x) {
    //             // 获取顶点坐标（X, Y, Z）
    //             cv::Vec3f vertex = vertex_map.at<cv::Vec3f>(y, x);
    //             //忽略无效的点（例如深度为0的点）
    //             if (vertex[2] == 0) continue;
    //             // 将三维点添加到点云中
    //             cloud->push_back(pcl::PointXYZ(vertex[0] / 1000, vertex[1] /1000, vertex[2]/1000));
    //         }
    //     }
    //     // 保存点云为PCD文件
    //     pcl::io::savePCDFileASCII("output.pcd", *cloud);
    // }

    void vertexMapToPointCloudAndSave(const cv::Mat& vertex_map, const cv::Mat& color_map) {
        // 遍历顶点图的每个像素
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (int y = 0; y < vertex_map.rows; ++y) {
            for (int x = 0; x < vertex_map.cols; ++x) {
                // 获取顶点坐标（X, Y, Z）
                cv::Vec3f vertex = vertex_map.at<cv::Vec3f>(y, x);
                // 忽略无效的点（例如深度为0的点）
                if (vertex[2] == 0) continue;
    
                // 获取对应的颜色
                cv::Vec3b color = color_map.at<cv::Vec3b>(y, x);
    
                // 将三维点添加到点云中，并设置RGB值
                pcl::PointXYZRGB point;
                point.x = vertex[0] / 1000;
                point.y = vertex[1] / 1000;
                point.z = vertex[2] / 1000;
                point.r = color[2];
                point.g = color[1];
                point.b = color[0];
    
                cloud->push_back(point);
            }
        }
        // 保存点云为PCD文件
        pcl::io::savePCDFileASCII("output.pcd", *cloud);
    }
}