
#include "../../../include/step/SurfaceReconstructor.h"
#include "../../../include/Matrix.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include "../../../include/DataTypes.h"


#define DIVSHORTMAX 0.0000305185f       // (1.f / SHRT_MAX);
#define SHORTMAX    32767               // SHRT_MAX;
#define MAX_WEIGHT  128                 // Global TSDF Volume 更新过程中, 允许的最大权重值

using cv::cuda::GpuMat;
using cv::cuda::PtrStep;
using cv::cuda::PtrStepSz;

using msr::Matf31;
using msr::Matf33;

namespace step {
    // 更新 TSDF 模型的核函数
    __global__
    void update_tsdf_kernel(
        const PtrStepSz<float> depth_image,                         // 原始大小深度图
        const PtrStepSz<uchar3> color_image,                        // 原始大小彩色图
        PtrStepSz<short2> tsdf_volume, 
        PtrStepSz<uchar3> color_volume,
        int3 volume_size, 
        float voxel_scale,
        CameraParameters cam_params,                                // 原始图层上的相机内参
        const float truncation_distance,
        Matf33 rotation,      // 旋转矩阵 -- 这里要求Eigen编译的时候使能cuda
        Matf31 translation)                                        // 平移向量
            {
                // step 1 获取当前线程的id, 并检查是否落在 volume 中.
                // 这里实际上是每个线程对应(x,y,*),每一个线程负责处理z轴上的所有数据
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                // 合法性检查
                if (x >= volume_size.x || y >= volume_size.y)
                    return;

                // step 2 处理z轴上的每一个体素的数据
                for (int z = 0; z < volume_size.z; ++z) {
                    // step 2.1 计算该体素中心点在当前帧相机坐标系下的坐标, 然后投影到图像中得到投影点坐标, 其中进行合法性检查
                    // 获取当前要处理的体素中心点在空间中的实际位置. 其中的0.5表示的是计算得到体素的中心, * voxel_scale 对应为实际空间尺度下体素的中心
                    const Matf31 position((static_cast<float>(x) + 0.5f) * voxel_scale,
                                           (static_cast<float>(y) + 0.5f) * voxel_scale,
                                           (static_cast<float>(z) + 0.5f) * voxel_scale);
                    // 将上面的在世界坐标系下的表示变换到在当前相机坐标系下的坐标
                    Matf31 camera_pos = rotation * position + translation;
                    // 合法性检查1: 如果这个体素相机看不到那么我们就不管了
                    if (camera_pos.z() <= 0)
                        continue;

                    // int __float2int_rn(float) : 求最近的偶数 // ?  为什么要求偶数? -- 我怀疑作者写错了, 这里应该是求整数吧
                    // 计算空间点在图像上的投影点, 并且认为这个投影点就是对这个空间体素的观测
                    const int u = __float2int_rn(camera_pos.x() / camera_pos.z() * cam_params.focal_x + cam_params.principal_x);
                    const int v = __float2int_rn(camera_pos.y() / camera_pos.z() * cam_params.focal_y + cam_params.principal_y);

                    // 合法性检查2: 查看投影点是否正确地投影在了图像范围内
                    if (u < 0 || u >= depth_image.cols || v < 0 || v >= depth_image.rows)
                        continue;
                    // 获取该体素中心点的深度的观测值(相对于当前图像来说)
                    const float depth = depth_image.ptr(v)[u];
                    // 合法性检查3: 深度的观测值应该非负
                    if (depth <= 0)
                        continue;

                    const Matf31 xylambda(
                            (u - cam_params.principal_x) / cam_params.focal_x,             // (x/z)
                            (v - cam_params.principal_y) / cam_params.focal_y,             // (y/z)
                            1.f);
                    // 计算得到公式7中的 lambda
                    const float lambda = xylambda.norm();
                    const float sdf = (-1.f) * ((1.f / lambda) * camera_pos.norm() - depth);
                    if (sdf >= -truncation_distance) {
                        // 说明当前的 SDF 表示获得的观测的深度值, 在我们构建的TSDF模型中, 是可观测的
                        // step 2.4.1 计算当前次观测得到的 TSDF 值
                        // 注意这里的 TSDF 值一直都是小于1的. 后面会利用这个特点来将浮点型的 TSDF 值保存为 uint16_t 类型
                        const float new_tsdf = fmin(1.f, sdf / truncation_distance);

                        short2 voxel_tuple = tsdf_volume.ptr(z * volume_size.y + y)[x];
                        const float current_tsdf = static_cast<float>(voxel_tuple.x) * DIVSHORTMAX;
                        const int current_weight = voxel_tuple.y;

                        // step 2.4.3 更新 TSDF 值和权重值
                        // 见下
                        const int add_weight = 1;
                        
                        const float updated_tsdf = (current_weight * current_tsdf + add_weight * new_tsdf) /
                                                   (current_weight + add_weight);
                        // 论文公式 13 对权重 进行更新
                        const int new_weight = min(current_weight + add_weight, MAX_WEIGHT);
                        // 将 浮点的 TSDF 值经过 int32_t 类型 保存为 uint16_t 类型. 限幅是因为理想情况下 无论是当前帧计算的还是融合之后的TSDF值都应该是小于1的
                        // (所以对应的值属于 -SHORTMAX ~ SHORTMAX)
                        //  类型中转是因为不这样做, updated_tsdf 一旦越界会出现截断, 导致 min max 函数都无法有效工作
                        const int new_value  = max(-SHORTMAX, min(SHORTMAX, static_cast<int>(updated_tsdf * SHORTMAX)));

                        // step 2.4.4 保存计算结果
                        tsdf_volume.ptr(z * volume_size.y + y)[x] = make_short2(static_cast<short>(new_value),
                                                                                static_cast<short>(new_weight));

                        // step 2.4.5 对 彩色图进行更新
                        // 前提是当前的这个体素的中心观测值在 TSDF 的1/2未截断区域内. 注意这里的约束其实更加严格, 这里是截断距离除了2
                        if (sdf <= truncation_distance / 2 && sdf >= -truncation_distance / 2) {
                            // step 2.4.5.1 获取当前体素对应的投影点的颜色的观测值和之前的储存值
                            // 储存值
                            uchar3& model_color = color_volume.ptr(z * volume_size.y + y)[x];
                            // 观测值
                            const uchar3 image_color = color_image.ptr(v)[u];

                            // step 2.4.5.2 颜色均匀化之后再写入, 仿照 TSDF 值的加权更新方式
                            model_color.x = static_cast<uchar>(
                                    (current_weight * model_color.x + add_weight * image_color.x) /
                                    (current_weight + add_weight));
                            model_color.y = static_cast<uchar>(
                                    (current_weight * model_color.y + add_weight * image_color.y) /
                                    (current_weight + add_weight));
                            model_color.z = static_cast<uchar>(
                                    (current_weight * model_color.z + add_weight * image_color.z) /
                                    (current_weight + add_weight));
                        }// 对彩色图进行更新
                    }// 如果根据我们得到的 SDF 告诉我们, 这个距离我们能够观测到, 那么更新 TSDF
                }// 处理z轴上的每一个体素的数据
            }// 核函数

    // 实现表面的重建, 即将当前帧的相机位姿已知的时候, 根据当前帧的surface mearsurment,融合到Global TSDF Model 中
    // 主机端函数
    void SurfaceReconstructor::surface_reconstruction(const cv::cuda::GpuMat& depth_image, const cv::cuda::GpuMat& color_image,
                                const CameraParameters& cam_params, const float truncation_distance,
                                const Eigen::Matrix4f& model_view)
    {
        // step 1 根据TSDF Volume的大小, 计算核函数的大小
        const dim3 threads(32, 32);
        const dim3 blocks((modelData.volumeSize.x + threads.x - 1) / threads.x,
                          (modelData.volumeSize.y + threads.y - 1) / threads.y);

        Matf33 rotation_matrix = msr::eigenMatrix3fToMatf33(model_view.block(0, 0, 3, 3));
        Matf31 translation_vector = msr::eigenMatrix3x1ToMatf31(model_view.block(0, 3, 3, 1));

        // step 2 启动核函数
        update_tsdf_kernel<<<blocks, threads>>>(
            depth_image,                            // 原始大小的深度图像
            color_image,                            // 原始大小的彩色图像
            modelData.tsdfVolume,                     // TSDF Volume, GpuMat
            modelData.colorVolume,                    // color Volume, GpuMat
            modelData.volumeSize,                     // Volume 的大小, int3
            modelData.voxelScale,                     // 尺度缩放, float
            cam_params,                             // 在当前图层上的相机内参
            truncation_distance,                    // 截断距离u
            rotation_matrix,                        // 旋转矩阵
            translation_vector);                    // 平移向量

        // step 3 等待所有的并行线程结束
        cudaThreadSynchronize();
    }


    SurfaceReconstructor::SurfaceReconstructor(CameraParameters cameraParameters,GlobalConfiguration globalConfiguration):
        modelData(globalConfiguration.volumeSize,globalConfiguration.voxelScale),cameraParameters(cameraParameters),globalConfiguration(globalConfiguration){
        
    }
}



namespace msr {
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

    Matf33 eigenMatrix3fToMatf33(const Eigen::Matrix3f& eigen_mat) {
        Matf33 mat;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                mat.at(i, j) = eigen_mat(i, j);
            }
        }
        return mat;
    }

    Matf31 eigenMatrix3x1ToMatf31(const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& eigen_mat) {
        Matf31 mat;
        for (int i = 0; i < 3; ++i) {
            mat.at(i) = eigen_mat(i);
        }
        return mat;
    }
}
