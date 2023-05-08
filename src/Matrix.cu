#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/Matrix.h"

//Matf31
__host__ __device__ Matf31::Matf31() : x00(0), x10(0), x20(0) {}

__host__ __device__ Matf31::Matf31(float x00, float x10, float x20) :
    x00(x00), x10(x10), x20(x20) {}

__host__ __device__ float& Matf31::at(int i) {
    return (i == 0 ? x00 : (i == 1 ? x10 : x20));
}

__host__ __device__ Matf31& Matf31::operator=(const Matf31& other) {
    if (this != &other) {
        x00 = other.x00;
        x10 = other.x10;
        x20 = other.x20;
    }
    return *this;
}

__host__ __device__ Matf31 Matf31::operator+(const Matf31& other) const {
    return Matf31(
        x00 + other.x00,
        x10 + other.x10,
        x20 + other.x20);
}

__host__ __device__ Matf31 Matf31::operator-(const Matf31& other) const {
    return Matf31(
        x00 - other.x00,
        x10 - other.x10,
        x20 - other.x20);
}

__host__ __device__ float Matf31::dot(const Matf31& other) const {
    return x00 * other.x00 + x10 * other.x10 + x20 * other.x20;
}

__host__ __device__ float& Matf31::x() {
    return x00;
}

__host__ __device__ float& Matf31::y() {
    return x10;
}

__host__ __device__ float& Matf31::z() {
    return x20;
}

__host__ __device__ void Matf31::print() const {
    for (int i = 0; i < 3; ++i) {
        printf("%f\n", (i == 0 ? x00 : (i == 1 ? x10 : x20)));
    }
}

__host__ __device__ float Matf31::norm() const {
    return sqrt(x00 * x00 + x10 * x10 + x20 * x20);
}

__host__ __device__ Matf31 Matf31::cross(const Matf31& other) const {
    return Matf31(
        x10 * other.x20 - x20 * other.x10,
        x20 * other.x00 - x00 * other.x20,
        x00 * other.x10 - x10 * other.x00);
}



//Matf33
__host__ __device__ Matf33::Matf33() : x00(0), x01(0), x02(0), x10(0), x11(0), x12(0), x20(0), x21(0), x22(0) {}

__host__ __device__ Matf33::Matf33(float x00, float x01, float x02, float x10, float x11, float x12, float x20, float x21, float x22) :
    x00(x00), x01(x01), x02(x02), x10(x10), x11(x11), x12(x12), x20(x20), x21(x21), x22(x22) {}

__host__ __device__ float& Matf33::at(int i, int j) {
    return (i == 0 ? (j == 0 ? x00 : (j == 1 ? x01 : x02)) : (i == 1 ? (j == 0 ? x10 : (j == 1 ? x11 : x12)) : (j == 0 ? x20 : (j == 1 ? x21 : x22))));
}

__host__ __device__ Matf33& Matf33::operator=(const Matf33& other) {
    if (this != &other) {
        x00 = other.x00; x01 = other.x01; x02 = other.x02;
        x10 = other.x10; x11 = other.x11; x12 = other.x12;
        x20 = other.x20; x21 = other.x21; x22 = other.x22;
    }
    return *this;
}

__host__ __device__ Matf33 Matf33::operator+(const Matf33& other) const {
    return Matf33(
        x00 + other.x00, x01 + other.x01, x02 + other.x02,
        x10 + other.x10, x11 + other.x11, x12 + other.x12,
        x20 + other.x20, x21 + other.x21, x22 + other.x22);
}

__host__ __device__ Matf33 Matf33::operator-(const Matf33& other) const {
    return Matf33(
        x00 - other.x00, x01 - other.x01, x02 - other.x02,
        x10 - other.x10, x11 - other.x11, x12 - other.x12,
        x20 - other.x20, x21 - other.x21, x22 - other.x22);
}

__host__ __device__ Matf33 Matf33::operator*(const Matf33& other) {
    return Matf33(
        x00 * other.x00 + x01 * other.x10 + x02 * other.x20,
        x00 * other.x01 + x01 * other.x11 + x02 * other.x21,
        x00 * other.x02 + x01 * other.x12 + x02 * other.x22,
        x10 * other.x00 + x11 * other.x10 + x12 * other.x20,
        x10 * other.x01 + x11 * other.x11 + x12 * other.x21,
        x10 * other.x02 + x11 * other.x12 + x12 * other.x22,
        x20 * other.x00 + x21 * other.x10 + x22 * other.x20,
        x20 * other.x01 + x21 * other.x11 + x22 * other.x21,
        x20 * other.x02 + x21 * other.x12 + x22 * other.x22);
}

__host__ __device__ Matf31 Matf33::operator*(const Matf31& other) const{
    return Matf31(
        x00 * other.x00 + x01 * other.x10 + x02 * other.x20,
        x10 * other.x00 + x11 * other.x10 + x12 * other.x20,
        x20 * other.x00 + x21 * other.x10 + x22 * other.x20);
}

__host__ __device__ void Matf33::print() const {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i == 0) {
                printf("%f ", j == 0 ? x00 : (j == 1 ? x01 : x02));
            } else if (i == 1) {
                printf("%f ", j == 0 ? x10 : (j == 1 ? x11 : x12));
            } else {
                printf("%f ", j == 0 ? x20 : (j == 1 ? x21 : x22));
            }
        }
        printf("\n");
    }
}


