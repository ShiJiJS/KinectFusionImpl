#pragma once
#ifndef __CUDACC__
#define __host__
#define __device__
#endif // __CUDACC__

#include <vector_types.h> 

//matrix of pose estimation
namespace mpe {

    class Matf31 {  
        friend class Matf33;
    public:
        float x00, x10, x20;
        __host__ __device__ Matf31();
        __host__ __device__ Matf31(float x00, float x10, float x20);

        __host__ __device__ float& at(int i);

        __host__ __device__ Matf31& operator=(const Matf31& other);

        __host__ __device__ Matf31 operator+(const Matf31& other) const;
        __host__ __device__ Matf31 operator-(const Matf31& other) const;
        __host__ __device__ float dot(const Matf31& other) const;
        __host__ __device__ float& x();
        __host__ __device__ float& y();
        __host__ __device__ float& z();

        __host__ __device__ float norm() const;
        __host__ __device__ Matf31 cross(const Matf31& other) const;

        __host__ __device__ void print() const;
    };


    class Matf33 {
    public:
        float x00, x01, x02, x10, x11, x12, x20, x21, x22;
        __host__ __device__ Matf33();
        __host__ __device__ Matf33(float x00, float x01, float x02, float x10, float x11, float x12, float x20, float x21, float x22);

        __host__ __device__ float& at(int i, int j);

        __host__ __device__ Matf33& operator=(const Matf33& other);

        __host__ __device__ Matf33 operator+(const Matf33& other) const;
        __host__ __device__ Matf33 operator-(const Matf33& other) const;
        __host__ __device__ Matf33 operator*(const Matf33& other);
        // __host__ __device__ Matf31 operator*(const Matf31& other) const;
        __host__ __device__ Matf31 operator*(const Matf31& other) const;

        __host__ __device__ void print() const;
    };

    //完成Matrix3f到Matf33的转化
    Matf33 eigenMatrix3fToMatf33(const Eigen::Matrix3f& eigen_mat);
    //完成Matrix31f到Matf31的转化
    Matf31 eigenMatrix3x1ToMatf31(const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& eigen_mat);

}




//matrix of surface reconstruction
namespace msr {

    class Matf31 {  
        friend class Matf33;
    public:
        float x00, x10, x20;
        __host__ __device__ Matf31();
        __host__ __device__ Matf31(float x00, float x10, float x20);

        __host__ __device__ float& at(int i);

        __host__ __device__ Matf31& operator=(const Matf31& other);

        __host__ __device__ Matf31 operator+(const Matf31& other) const;
        __host__ __device__ Matf31 operator-(const Matf31& other) const;
        __host__ __device__ float dot(const Matf31& other) const;
        __host__ __device__ float& x();
        __host__ __device__ float& y();
        __host__ __device__ float& z();

        __host__ __device__ float norm() const;
        __host__ __device__ Matf31 cross(const Matf31& other) const;

        __host__ __device__ void print() const;
    };


    class Matf33 {
    public:
        float x00, x01, x02, x10, x11, x12, x20, x21, x22;
        __host__ __device__ Matf33();
        __host__ __device__ Matf33(float x00, float x01, float x02, float x10, float x11, float x12, float x20, float x21, float x22);

        __host__ __device__ float& at(int i, int j);

        __host__ __device__ Matf33& operator=(const Matf33& other);

        __host__ __device__ Matf33 operator+(const Matf33& other) const;
        __host__ __device__ Matf33 operator-(const Matf33& other) const;
        __host__ __device__ Matf33 operator*(const Matf33& other);
        // __host__ __device__ Matf31 operator*(const Matf31& other) const;
        __host__ __device__ Matf31 operator*(const Matf31& other) const;

        __host__ __device__ void print() const;
    };

    //完成Matrix3f到Matf33的转化
    Matf33 eigenMatrix3fToMatf33(const Eigen::Matrix3f& eigen_mat);
    //完成Matrix31f到Matf31的转化
    Matf31 eigenMatrix3x1ToMatf31(const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& eigen_mat);

}



//matrix of surface prediction
namespace msp {

    class Matf31 {  
        friend class Matf33;
    public:
        float x00, x10, x20;
        __host__ __device__ Matf31();
        __host__ __device__ Matf31(float x00, float x10, float x20);

        __host__ __device__ float& at(int i);

        __host__ __device__ Matf31& operator=(const Matf31& other);

        __host__ __device__ Matf31 operator+(const Matf31& other) const;
        __host__ __device__ Matf31 operator-(const Matf31& other) const;
        __host__ __device__ float dot(const Matf31& other) const;
        __host__ __device__ float& x();
        __host__ __device__ float& y();
        __host__ __device__ float& z();
        __host__ __device__ float x() const;
        __host__ __device__ float y() const;
        __host__ __device__ float z() const;

        __host__ __device__ float norm() const;
        __host__ __device__ Matf31 cross(const Matf31& other) const;

        __host__ __device__ void print() const;
        __host__ __device__ void normalize();
        // 在类定义的公共部分添加以下声明：
        __host__ __device__ Matf31 operator*(float scalar) const;
        __host__ __device__ Matf31 operator/(float scalar) const;
        __host__ __device__ int3 castToInt() const;
    };


    class Matf33 {
    public:
        float x00, x01, x02, x10, x11, x12, x20, x21, x22;
        __host__ __device__ Matf33();
        __host__ __device__ Matf33(float x00, float x01, float x02, float x10, float x11, float x12, float x20, float x21, float x22);

        __host__ __device__ float& at(int i, int j);

        __host__ __device__ Matf33& operator=(const Matf33& other);

        __host__ __device__ Matf33 operator+(const Matf33& other) const;
        __host__ __device__ Matf33 operator-(const Matf33& other) const;
        __host__ __device__ Matf33 operator*(const Matf33& other);
        // __host__ __device__ Matf31 operator*(const Matf31& other) const;
        __host__ __device__ Matf31 operator*(const Matf31& other) const;

        __host__ __device__ void print() const;
    };

    //完成Matrix3f到Matf33的转化
    Matf33 eigenMatrix3fToMatf33(const Eigen::Matrix3f& eigen_mat);
    //完成Matrix31f到Matf31的转化
    Matf31 eigenMatrix3x1ToMatf31(const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& eigen_mat);

}


//matrix of extract point cloud
namespace mepc{
    class Matf31 {  
        friend class Matf33;
    public:
        float x00, x10, x20;
        __host__ __device__ Matf31();
        __host__ __device__ Matf31(float x00, float x10, float x20);

        __host__ __device__ float& at(int i);

        __host__ __device__ Matf31& operator=(const Matf31& other);

        __host__ __device__ Matf31 operator+(const Matf31& other) const;
        __host__ __device__ Matf31 operator-(const Matf31& other) const;
        __host__ __device__ float dot(const Matf31& other) const;
        __host__ __device__ float& x();
        __host__ __device__ float& y();
        __host__ __device__ float& z();
        __host__ __device__ float x() const;
        __host__ __device__ float y() const;
        __host__ __device__ float z() const;

        __host__ __device__ float norm() const;
        __host__ __device__ Matf31 cross(const Matf31& other) const;

        __host__ __device__ void print() const;
        __host__ __device__ void normalize();
        // 在类定义的公共部分添加以下声明：
        __host__ __device__ Matf31 operator*(float scalar) const;
        __host__ __device__ Matf31 operator/(float scalar) const;
        __host__ __device__ int3 castToInt() const;
    };


    class Matf33 {
    public:
        float x00, x01, x02, x10, x11, x12, x20, x21, x22;
        __host__ __device__ Matf33();
        __host__ __device__ Matf33(float x00, float x01, float x02, float x10, float x11, float x12, float x20, float x21, float x22);

        __host__ __device__ float& at(int i, int j);

        __host__ __device__ Matf33& operator=(const Matf33& other);

        __host__ __device__ Matf33 operator+(const Matf33& other) const;
        __host__ __device__ Matf33 operator-(const Matf33& other) const;
        __host__ __device__ Matf33 operator*(const Matf33& other);
        // __host__ __device__ Matf31 operator*(const Matf31& other) const;
        __host__ __device__ Matf31 operator*(const Matf31& other) const;

        __host__ __device__ void print() const;
    };
}