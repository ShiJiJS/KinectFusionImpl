#include "../../../include/Scheduler.h"
#include "../../../include/Utils.h"
#include "../../../include/Matrix.h"

#define DIVSHORTMAX 0.0000305185f       // (1.f / SHRT_MAX);
#define SHORTMAX    32767               // SHRT_MAX;
#define MAX_WEIGHT  128                 // Global TSDF Volume 更新过程中, 允许的最大权重值

using cv::cuda::PtrStep;
using cv::cuda::PtrStepSz;

using mepc::Matf31;


__global__
void extract_points_kernel(const PtrStep<short2> tsdf_volume, const PtrStep<uchar3> color_volume,
                           const int3 volume_size, const float voxel_scale,
                           PtrStep<float3> vertices, PtrStep<float3> normals, PtrStep<uchar3> color,
                           int *point_num)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= volume_size.x - 1 || y >= volume_size.y - 1)
        return;
    for (int z = 0; z < volume_size.z - 1; ++z) {
        const short2 value = tsdf_volume.ptr(z * volume_size.y + y)[x];
        const float tsdf = static_cast<float>(value.x) * DIVSHORTMAX;
        if (tsdf == 0 || tsdf <= -0.99f || tsdf >= 0.99f)
            continue;
        short2 vx = tsdf_volume.ptr((z) * volume_size.y + y)[x + 1];
        short2 vy = tsdf_volume.ptr((z) * volume_size.y + y + 1)[x];
        short2 vz = tsdf_volume.ptr((z + 1) * volume_size.y + y)[x];
        if (vx.y <= 0 || vy.y <= 0 || vz.y <= 0)
            continue;
        const float tsdf_x = static_cast<float>(vx.x) * DIVSHORTMAX;
        const float tsdf_y = static_cast<float>(vy.x) * DIVSHORTMAX;
        const float tsdf_z = static_cast<float>(vz.x) * DIVSHORTMAX;
        const bool is_surface_x = ((tsdf > 0) && (tsdf_x < 0)) || ((tsdf < 0) && (tsdf_x > 0));
        const bool is_surface_y = ((tsdf > 0) && (tsdf_y < 0)) || ((tsdf < 0) && (tsdf_y > 0));
        const bool is_surface_z = ((tsdf > 0) && (tsdf_z < 0)) || ((tsdf < 0) && (tsdf_z > 0));
        if (is_surface_x || is_surface_y || is_surface_z) {
            Matf31 normal;
            normal.x() = (tsdf_x - tsdf);
            normal.y() = (tsdf_y - tsdf);
            normal.z() = (tsdf_z - tsdf);
            if (normal.norm() == 0)
                continue;
            normal.normalize();
            int count = 0;
            if (is_surface_x) count++;
            if (is_surface_y) count++;
            if (is_surface_z) count++;
            int index = atomicAdd(point_num, count);
            Matf31 position((static_cast<float>(x) + 0.5f) * voxel_scale,
                             (static_cast<float>(y) + 0.5f) * voxel_scale,
                             (static_cast<float>(z) + 0.5f) * voxel_scale);
            if (is_surface_x) {
                position.x() = position.x() - (tsdf / (tsdf_x - tsdf)) * voxel_scale;
                vertices.ptr(0)[index] = float3{position.at(0), position.at(1), position.at(2)};
                normals.ptr(0)[index] = float3{normal.at(0), normal.at(1), normal.at(2)};
                color.ptr(0)[index] = color_volume.ptr(z * volume_size.y + y)[x];
                index++;
            }
            if (is_surface_y) {
                position.y() -= (tsdf / (tsdf_y - tsdf)) * voxel_scale;
                vertices.ptr(0)[index] = float3{position.at(0), position.at(1), position.at(2)};;
                normals.ptr(0)[index] = float3{normal.at(0), normal.at(1), normal.at(2)};
                color.ptr(0)[index] = color_volume.ptr(z * volume_size.y + y)[x];
                index++;
            }
            if (is_surface_z) {
                position.z() -= (tsdf / (tsdf_z - tsdf)) * voxel_scale;
                vertices.ptr(0)[index] = float3{position.at(0), position.at(1), position.at(2)};;
                normals.ptr(0)[index] = float3{normal.at(0), normal.at(1), normal.at(2)};
                color.ptr(0)[index] = color_volume.ptr(z * volume_size.y + y)[x];
                index++;
            }
        }
    }
}


void Scheduler::extract_and_save_pointcloud() {
    int pointcloud_buffer_size { 3 * 2000000 };
    CloudData cloud_data { pointcloud_buffer_size };
    ModelData modelData = this->surfaceReconstructor.getModelData();
    dim3 threads(32, 32);
    dim3 blocks((modelData.volumeSize.x + threads.x - 1) / threads.x,
                (modelData.volumeSize.y + threads.y - 1) / threads.y);
    extract_points_kernel<<<blocks, threads>>>(modelData.tsdfVolume, modelData.colorVolume,
            modelData.volumeSize, modelData.voxelScale,
            cloud_data.vertices, cloud_data.normals, cloud_data.color,
            cloud_data.point_num);
    cudaThreadSynchronize();
    cloud_data.download();
    utils::vertexMapToPointCloudAndSave(cloud_data.host_vertices);
    std::cout << "点云输出成功" << std::endl;
    
}



namespace mepc{
//矩阵部分的定义。因为分离编译的问题没有解决。所以只能将device函数的定义与调用部分放到一个文件中
    // //Matf31
    //矩阵部分的定义。因为分离编译的问题没有解决。所以只能将device函数的定义与调用部分放到一个文件中
    // //Matf31
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

    __host__ __device__ float Matf31::x() const {
        return x00;
    }

    __host__ __device__ float Matf31::y() const {
        return x10;
    }
    
    __host__ __device__ float Matf31::z() const {
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


    __host__ __device__ void Matf31::normalize() {
        float len = norm();
        if (len != 0.0f) {
            x00 /= len;
            x10 /= len;
            x20 /= len;
        }
    }

    __host__ __device__ Matf31 Matf31::operator*(float scalar) const {
        return Matf31(x00 * scalar, x10 * scalar, x20 * scalar);
    }

    __host__ __device__ Matf31 Matf31::operator/(float scalar) const {
        // 可能需要处理除以零的情况，这里直接返回一个未经修改的副本
        if (scalar == 0.0f) {
            return *this;
        }
        return Matf31(x00 / scalar, x10 / scalar, x20 / scalar);
    }
    __host__ __device__ int3 Matf31::castToInt() const {
        int3 result;
        result.x = static_cast<int>(x00);
        result.y = static_cast<int>(x10);
        result.z = static_cast<int>(x20);
        return result;
    }
}