#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
// #include <pcl/visualization/cloud_viewer.h>
// #include <pcl/point_types.h>
// #include <pcl/point_cloud.h>
// #include <pcl/io/pcd_io.h>
#include "Matrix.h"
#pragma once
// using pcl::PointCloud;


namespace utils{
    //读入深度图索引文件并返回存储路径的vector
    std::vector<std::string> read_depth_image_paths(const std::string& index_file);
    //输出mat的类型
    std::string matTypeToString(const cv::Mat& mat);
    //生成含有坐标法向量和RGB的点云
    // PointCloud<pcl::PointXYZRGBNormal>::Ptr create_point_cloud(const cv::Mat& vertex_map, const cv::Mat& normal_map);
    // //生成仅有坐标的点云
    // void vertexMapToPointCloudAndSave(const cv::Mat& vertex_map);
    //完成Matrix3f到Matf33的转化
    mpe::Matf33 eigenMatrix3fToMatf33(const Eigen::Matrix3f& eigen_mat);
    //完成Matrix31f到Matf31的转化
    mpe::Matf31 eigenMatrix3x1ToMatf31(const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& eigen_mat);
}
