#ifndef __CUDACC__
#define __host__
#define __device__
#endif // __CUDACC__
#pragma once

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

}