#include "../include/Utils.h"


namespace utils{
    mpe::Matf33 eigenMatrix3fToMatf33(const Eigen::Matrix3f& eigen_mat) {
        mpe::Matf33 mat;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                mat.at(i, j) = eigen_mat(i, j);
            }
        }
        return mat;
    }

    mpe::Matf31 eigenMatrix3x1ToMatf31(const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& eigen_mat) {
        mpe::Matf31 mat;
        for (int i = 0; i < 3; ++i) {
            mat.at(i) = eigen_mat(i);
        }
        return mat;
    }
}