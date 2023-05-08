#include "../../../include/step/SurfacePredictor.h"
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

using msp::Matf31;
using msp::Matf33;



namespace step{
    __device__ __forceinline__
    // 三线型插值
    float interpolate_trilinearly(
        const Matf31& point,               // 想要得到 TSDF 数值的点的坐标(非整数)
        const PtrStepSz<short2>& volume,    // TSDF Volume 对象
        const int3& volume_size,            // TSDF Volume 对象 大小
        const float voxel_scale)            // TSDF Volume 中的体素坐标和现实世界中长度的度量关系
    {
        // 本函数中考虑的, 都是

        // 这个点在 Volume 下的坐标, 转换成为整数下标标的表示
        int3 point_in_grid = point.castToInt();

        // 恢复成体素中心点的坐标
        const float vx = (static_cast<float>(point_in_grid.x) + 0.5f);
        const float vy = (static_cast<float>(point_in_grid.y) + 0.5f);
        const float vz = (static_cast<float>(point_in_grid.z) + 0.5f);

        point_in_grid.x = (point.x() < vx) ? (point_in_grid.x - 1) : point_in_grid.x;
        point_in_grid.y = (point.y() < vy) ? (point_in_grid.y - 1) : point_in_grid.y;
        point_in_grid.z = (point.z() < vz) ? (point_in_grid.z - 1) : point_in_grid.z;

        // +0.5f 的原因是, point_in_grid 处体素存储的TSDF值是体素的中心点的TSDF值
        // 三线型插值, ref: https://en.wikipedia.org/wiki/Trilinear_interpolation
        // 计算精确的(浮点型)的点坐标和整型化之后的点坐标的差
        const float a = (point.x() - (static_cast<float>(point_in_grid.x) + 0.5f));
        const float b = (point.y() - (static_cast<float>(point_in_grid.y) + 0.5f));
        const float c = (point.z() - (static_cast<float>(point_in_grid.z) + 0.5f));

        return 
            static_cast<float>(volume.ptr((point_in_grid.z) * volume_size.y + point_in_grid.y)[point_in_grid.x].x) * DIVSHORTMAX 
                // volume[ x ][ y ][ z ], C000
                * (1 - a) * (1 - b) * (1 - c) +
            static_cast<float>(volume.ptr((point_in_grid.z + 1) * volume_size.y + point_in_grid.y)[point_in_grid.x].x) * DIVSHORTMAX 
                // volume[ x ][ y ][z+1], C001
                * (1 - a) * (1 - b) * c +
            static_cast<float>(volume.ptr((point_in_grid.z) * volume_size.y + point_in_grid.y + 1)[point_in_grid.x].x) * DIVSHORTMAX 
                // volume[ x ][y+1][ z ], C010
                * (1 - a) * b * (1 - c) +
            static_cast<float>(volume.ptr((point_in_grid.z + 1) * volume_size.y + point_in_grid.y + 1)[point_in_grid.x].x) * DIVSHORTMAX 
                // volume[ x ][y+1][z+1], C011
                * (1 - a) * b * c +
            static_cast<float>(volume.ptr((point_in_grid.z) * volume_size.y + point_in_grid.y)[point_in_grid.x + 1].x) * DIVSHORTMAX 
                // volume[x+1][ y ][ z ], C100
                * a * (1 - b) * (1 - c) +
            static_cast<float>(volume.ptr((point_in_grid.z + 1) * volume_size.y + point_in_grid.y)[point_in_grid.x + 1].x) * DIVSHORTMAX 
                // volume[x+1][ y ][z+1], C101
                * a * (1 - b) * c +
            static_cast<float>(volume.ptr((point_in_grid.z) * volume_size.y + point_in_grid.y + 1)[point_in_grid.x + 1].x) * DIVSHORTMAX 
                // volume[x+1][y+1][ z ], C110
                * a * b * (1 - c) +
            static_cast<float>(volume.ptr((point_in_grid.z + 1) * volume_size.y + point_in_grid.y + 1)[point_in_grid.x + 1].x) * DIVSHORTMAX 
                // volume[x+1][y+1][z+1], C111
                * a * b * c;
    }



    // __forceinline__: 强制为内联函数
    __device__ __forceinline__
    // 求射线为了射入Volume, 在给定步长下所需要的最少的前进次数(也可以理解为前进所需要的时间)
    float get_min_time(
        const float3&   volume_max,     // 体素的范围(真实尺度)
        const Matf31&  origin,         // 出发点, 也就是相机当前的位置
        const Matf31&  direction)      // 射线方向
    {
        // 分别计算三个轴上的次数, 并且返回其中最大; 当前进了这个最大的次数之后, 三个轴上射线的分量就都已经射入volume了
        float txmin = ((direction.x() > 0 ? 0.f : volume_max.x) - origin.x()) / direction.x();
        float tymin = ((direction.y() > 0 ? 0.f : volume_max.y) - origin.y()) / direction.y();
        float tzmin = ((direction.z() > 0 ? 0.f : volume_max.z) - origin.z()) / direction.z();
        
        return fmax(fmax(txmin, tymin), tzmin);
    }

    __device__ __forceinline__
    // 求射线为了射出Volume, 在给定步长下所需要的最少的前进次数(也可以理解为前进所需要的时间)
    float get_max_time(const float3& volume_max, const Matf31& origin, const Matf31& direction)
    {
        // 分别计算三个轴上的次数, 并且返回其中最小. 当前进了这个最小的次数后, 三个轴上的射线的分量中就有一个已经射出了volume了
        float txmax = ((direction.x() > 0 ? volume_max.x : 0.f) - origin.x()) / direction.x();
        float tymax = ((direction.y() > 0 ? volume_max.y : 0.f) - origin.y()) / direction.y();
        float tzmax = ((direction.z() > 0 ? volume_max.z : 0.f) - origin.z()) / direction.z();

        return fmin(fmin(txmax, tymax), tzmax);
    }

    __global__
    void raycast_tsdf_kernel(
        const PtrStepSz<short2>     tsdf_volume,                        // Global TSDF Volume
        const PtrStepSz<uchar3>     color_volume,                       // Global Color Volume
        PtrStepSz<float3>           model_vertex,                       // 推理出来的顶点图
        PtrStepSz<float3>           model_normal,                       // 推理出来的法向图
        PtrStepSz<uchar3>           model_color,                        // 推理出来的颜色图
        const int3                  volume_size,                        // Volume 尺寸
        const float                 voxel_scale,                        // Volume 缩放洗漱
        const CameraParameters      cam_parameters,                     // 当前图层相机内参
        const float                 truncation_distance,                // 截断距离
        const Matf33                rotation,    // 相机位姿的旋转矩阵
        const Matf31               translation)                        // 相机位姿的平移向量
    {
        // step 0 获取当前线程要处理的图像像素, 并且进行合法性检查
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        // 合法性检查: 判断是否在当前图层图像范围内
        if (x >= model_vertex.cols || y >= model_vertex.rows)
            return;

        // step 2 计算 raycast 射线, 以及应该在何处开始, 在何处结束
        // 计算 Volume 对应的空间范围
        // ! 但是我觉得, 这个范围其实和当前的线程id没有关系, 我们完全可以离线计算啊, 这里让 512*512*512 的每一个线程都计算一次是不是太浪费计算资源了
        const float3 volume_range = make_float3(volume_size.x * voxel_scale,
                                                volume_size.y * voxel_scale,
                                                volume_size.z * voxel_scale);
        // 计算当前的点和相机光心的连线, 使用的是在当前相机坐标系下的坐标; 由于后面只是为了得到方向所以这里没有乘Z
        const Matf31 pixel_position(
                (x - cam_parameters.principal_x) / cam_parameters.focal_x,      // X/Z
                (y - cam_parameters.principal_y) / cam_parameters.focal_y,      // Y/Z
                1.f);                                                           // Z/Z
        // 得到这个连线的方向(从相机指向空间中的反投影射线)在世界坐标系下的表示, 联想: P_w = R_{wc} * P_c
        Matf31 ray_direction = (rotation * pixel_position);
        ray_direction.normalize();

        // fmax: CUDA 中 float 版的 max() 函数
        // 参数 translation 应该理解为相机光心在世界坐标系下的坐标
        // 获得 raycast 的起始位置
        float ray_length = fmax(get_min_time(volume_range, translation, ray_direction), 0.f);
        // 验证是否合法: 起始位置的射线长度应该小于等于结束位置的射线长度
        if (ray_length >= get_max_time(volume_range, translation, ray_direction))
            return;

        // 在开始位置继续前进一个体素, 确保该位置已经接触到 volume
        ray_length += voxel_scale;
        Matf31 grid = (translation + (ray_direction * ray_length)) / voxel_scale;

        // 拿到 Grid 对应体素处的 TSDF 值, 这里充当当前射线的上一次的TSDF计算结果
        // 如果拿到的坐标并不在 volume 中, 那么得到的 tsdf 值无法确定, 甚至可能会触发段错误
        // __float2int_rd: 向下取整
        float tsdf = static_cast<float>(tsdf_volume.ptr(
                __float2int_rd(grid.at(2)) * volume_size.y + __float2int_rd(grid.at(1)))[__float2int_rd(grid.at(0))].x) *
                     DIVSHORTMAX;

        // 计算最大搜索长度(考虑了光线开始“投射”的时候已经走过的路程 ray_length )  
        // ! 不明白这里为什么是根号2 而不是根号3
        // ! 这里没有乘 SCALE 也应该有问题
        const float max_search_length = ray_length + volume_range.x * sqrt(2.f);
        // step 3 开始迭代搜索了, raycasting 开始. 步长为一半截断距离
        for (; ray_length < max_search_length; ray_length += truncation_distance * 0.5f) {

            // step 3.1 获取当前射线位置的 TSDF
            // 计算当前次前进后, 射线到达的体素id
            grid = ((translation + (ray_direction * (ray_length + truncation_distance * 0.5f))) / voxel_scale);

            // 合法性检查
            if (grid.x() < 1 || grid.x() >= volume_size.x - 1 || grid.y() < 1 ||
                grid.y() >= volume_size.y - 1 ||
                grid.z() < 1 || grid.z() >= volume_size.z - 1)
                continue;

            // 保存上一次的 TSDF 值, 用于进行下面的判断
            const float previous_tsdf = tsdf;
            // 计算当前 Grid 处的 TSDF 值
            tsdf = static_cast<float>(tsdf_volume.ptr(
                    __float2int_rd(grid.at(2)) * volume_size.y + __float2int_rd(grid.at(1)))[__float2int_rd(
                    grid.at(0))].x) *
                   DIVSHORTMAX;

            // step 3.2 判断是否穿过了平面
            if (previous_tsdf < 0.f && tsdf > 0.f) //Zero crossing from behind
                // 这种情况是从平面的后方穿出了
                break;
            if (previous_tsdf > 0.f && tsdf < 0.f) { //Zero crossing
                // step 3.3 确实在当前的位置穿过了平面, 计算当前射线与该平面的交点

                // 精确确定这个平面所在的位置(反映为射线的长度), 计算公式与论文中式(15)保持一致
                const float t_star =
                        ray_length - truncation_distance * 0.5f * previous_tsdf / (tsdf - previous_tsdf);
                // 计算射线和这个平面的交点. 下文简称平面顶点. vec3f 类型
                const auto vertex = translation + ray_direction * t_star;

                // 计算平面顶点在 volume 中的位置
                const Matf31 location_in_grid = (vertex / voxel_scale);
                // 然后进行合法性检查, 如果确认这个 vertex 不在我们的 Volume 中那么我们就不管它了
                if (location_in_grid.x() < 1 || location_in_grid.x() >= volume_size.x - 1 ||
                    location_in_grid.y() < 1 || location_in_grid.y() >= volume_size.y - 1 ||
                    location_in_grid.z() < 1 || location_in_grid.z() >= volume_size.z - 1)
                    break;

                // step 3.4 分x, y, z三个轴, 计算这个 Grid 点所在处的平面的法向量

                // normal  - 法向量
                // shifted - 中间变量, 用于滑动
                Matf31 normal, shifted;

                // step 3.4.1 对 x 轴方向
                shifted = location_in_grid;
                // 在平面顶点的体素位置的基础上, 哎我滑~ 如果滑出体素范围就不管了
                shifted.x() += 1;
                if (shifted.x() >= volume_size.x - 1)
                    break;
                // 这里得到的是 TSDF 值. 
                // 为什么不直接使用 shifted 对应体素的 TSDF 值而是进行三线性插值, 是因为 Volume 中只保存了体素中心点到平面的距离, 
                // 但是这里的 location_in_grid+1 也就是 shifted 是个浮点数, 为了得到相对准确的TSDF值, 需要进行三线性插值
                const float Fx1 = interpolate_trilinearly(
                    shifted,            // vertex 点在Volume的坐标滑动之后的点, Vec3fda
                    tsdf_volume,        // TSDF Volume
                    volume_size,        // Volume 的大小
                    voxel_scale);       // 尺度信息

                // 类似的操作, 不过滑动的时候换了一个方向
                shifted = location_in_grid;
                shifted.x() -= 1;
                if (shifted.x() < 1)
                    break;
                const float Fx2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

                // 由于 TSDF 值就反映了该体素中心点处到相机反投影平面的距离, 所以这里可以使用这个数据来进行表示
                // ! 但是这样基于这个点周围体素中的距离都没有被截断才比较准确, 否则可能出现一个轴上的法向量为0的情况
                normal.x() = (Fx1 - Fx2);

                // step 3.4.2 对 y 轴方向
                shifted = location_in_grid;
                shifted.y() += 1;
                if (shifted.y() >= volume_size.y - 1)
                    break;
                const float Fy1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

                shifted = location_in_grid;
                shifted.y() -= 1;
                if (shifted.y() < 1)
                    break;
                const float Fy2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

                normal.y() = (Fy1 - Fy2);

                // step 3.4.3 对 z 轴方向
                shifted = location_in_grid;
                shifted.z() += 1;
                if (shifted.z() >= volume_size.z - 1)
                    break;
                const float Fz1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

                shifted = location_in_grid;
                shifted.z() -= 1;
                if (shifted.z() < 1)
                    break;
                const float Fz2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

                normal.z() = (Fz1 - Fz2);

                // step 3.4.4 检查法向量是否计算成功, 如果成功进行归一化
                if (normal.norm() == 0)
                    break;

                // 如果法向量计算成功, 那么首先归一化
                normal.normalize();

                // step 3.5 保存平面顶点和平面法向数据
                // 然后将计算结果保存到顶点图和法向图中
                model_vertex.ptr(y)[x] = make_float3(vertex.x(), vertex.y(), vertex.z());
                model_normal.ptr(y)[x] = make_float3(normal.x(), normal.y(), normal.z());

                // step 3.6 获取该点处的彩色数据
                // 将浮点类型的这个顶点在Volume中的位置转换成为以int类型表示的
                int3 location_in_grid_int = location_in_grid.castToInt();
                // 然后就可以使用这个整数下标获取 Color Volume 中存储的彩色数据了, 将它保存到彩色图中
                model_color.ptr(y)[x] = color_volume.ptr(
                        location_in_grid_int.z * volume_size.y +
                        location_in_grid_int.y)[location_in_grid_int.x];

                break;
            }
        } // raycasting
    }

    // 执行当前帧的指定图层上的表面推理
    void SurfacePredictor::surface_prediction(
        const ModelData& model_data,                   // Global Volume
        GpuMat& model_vertex,                       // 推理得到的顶点图
        GpuMat& model_normal,                       // 推理得到的法向图
        GpuMat& model_color,                        // 推理得到的颜色
        const CameraParameters& cam_parameters,     // 当前图层的相机内参
        const float truncation_distance,            // 截断距离
        const Eigen::Matrix4f& pose)                // 当前帧的相机位姿
    {
        // step 0 数据准备: 清空顶点图\法向图\彩色图
        model_vertex.setTo(0);
        model_normal.setTo(0);
        model_color.setTo(0);

        // step 1 计算线程数量, 这和当前图层图像的大小有关
        dim3 threads(32, 32);
        dim3 blocks((model_vertex.cols + threads.x - 1) / threads.x,
                    (model_vertex.rows + threads.y - 1) / threads.y);

        Matf33 rotation = msp::eigenMatrix3fToMatf33(pose.block(0, 0, 3, 3));
        Matf31 tanslation = msp::eigenMatrix3x1ToMatf31(pose.block(0, 3, 3, 1));
        // step 2 调用核函数进行并行计算
        raycast_tsdf_kernel<<<blocks, threads>>>(
                model_data.tsdfVolume,                 // Global TSDF Volume
                model_data.colorVolume,                // Global Color Volume
                model_vertex,                       // 推理出来的顶点图
                model_normal,                       // 推理出来的法向图
                model_color,                        // 推理出来的颜色图
                model_data.volumeSize,                 // Volume 尺寸
                model_data.voxelScale,                 // Volume 缩放洗漱
                cam_parameters,                     // 当前图层相机内参
                truncation_distance,                // 截断距离
                rotation,             // 从相机位姿中提取旋转矩阵
                tanslation);            // 从相机位姿中提取平移向量

        // step 3 等待线程同步, 然后结束
        cudaThreadSynchronize();
    }


    SurfacePredictor::SurfacePredictor(CameraParameters cameraParameters,GlobalConfiguration globalConfiguration):
    cameraParameters(cameraParameters),globalConfiguration(globalConfiguration){

    }
}



namespace msp {
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

    __host__ __device__ int3 Matf31::castToInt() const {
        int3 result;
        result.x = static_cast<int>(x00);
        result.y = static_cast<int>(x10);
        result.z = static_cast<int>(x20);
        return result;
    }


}
