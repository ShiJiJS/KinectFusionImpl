#ifndef __CUDACC__
#define __host__
#define __device__
#endif // __CUDACC__
#pragma once
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

    // __host__ __device__ Matf33 inverse() const {
    //     float det = x00 * (x11 * x22 - x21 * x12) - x01 * (x10 * x22 - x20 * x12) + x02 * (x10 * x21 - x20 * x11);
        
    //     // Check if the matrix is singular (non-invertible)
    //     if (fabs(det) < 1e-6) {
    //         // Return an identity matrix or throw an exception, depending on your error handling approach
    //         return Matf33(1.0f, 0, 0, 0, 1.0f, 0, 0, 0, 1.0f);
    //     }
        
    //     float inv_det = 1.0f / det;
    //     Matf33 inv(
    //         (x11 * x22 - x21 * x12) * inv_det,
    //         (x02 * x21 - x01 * x22) * inv_det,
    //         (x01 * x12 - x02 * x11) * inv_det,
    //         (x12 * x20 - x10 * x22) * inv_det,
    //         (x00 * x22 - x02 * x20) * inv_det,
    //         (x10 * x02 - x00 * x12) * inv_det,
    //         (x10 * x21 - x20 * x11) * inv_det,
    //         (x20 * x01 - x00 * x21) * inv_det,
    //         (x00 * x11 - x01 * x10) * inv_det
    //     );
    
    //     return inv;
    // }
    
};