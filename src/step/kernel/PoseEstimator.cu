#include "../../../include/step/PoseEstimator.h"
#include "../../../include/Matrix.h"
#include "../../../include/Utils.h"
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32




using mpe::Matf31;
using mpe::Matf33;

using cv::cuda::PtrStep;
using cv::cuda::PtrStepSz;
// using Matf31da = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;

namespace step {
    namespace kernel{


        // 设备端的函数, 用于执行归约累加的操作
        template<int SIZE>
        static __device__ __forceinline__
        // volatile 关键字禁止了 nvcc 编译器优化掉这个变量, 确保每次都要读值, 避免了潜在的使用上次用剩下的指针的可能
        void reduce(volatile double* buffer)
        {
            // step 0 获取当前线程id , 每个线程对应其中的一个
            const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
            double value = buffer[thread_id];

            // step 1 归约过程开始, 之所以这样做是为了充分利用 GPU 的并行特性
            if (SIZE >= 1024) {
                if (thread_id < 512) buffer[thread_id] = value = value + buffer[thread_id + 512];
                // 一定要同步! 因为如果block规模很大的话, 其中的线程是分批次执行的, 这里就会得到错误的结果
                __syncthreads();
            }
            if (SIZE >= 512) {
                if (thread_id < 256) buffer[thread_id] = value = value + buffer[thread_id + 256];
                __syncthreads();
            }
            if (SIZE >= 256) {
                if (thread_id < 128) buffer[thread_id] = value = value + buffer[thread_id + 128];
                __syncthreads();
            }
            if (SIZE >= 128) {
                if (thread_id < 64) buffer[thread_id] = value = value + buffer[thread_id + 64];
                __syncthreads();
            }

            // step 2 随着归约过程的进行, 当最后剩下的几个线程都在一个warp中时, 就不用考虑线程间同步的问题了, 这样操作可以更快
            // 因为在 128 折半之后, 有64个数据等待加和, 此时需要使用的线程数目不会超过32个. 
            // 而一个warp,正好是32个线程, 所以如果我们使用这32个线程(或者更少的话)就不会遇到线程间同步的问题了(单指令多数据模式, 这32个线程会共享一套取指令单元, 一定是同时完成工作的)
            // 只激活低32个线程, CUDA 中底层的这32个线程一定是在一个warp上进行的.
            if (thread_id < 32) {
                if (SIZE >= 64) buffer[thread_id] = value = value + buffer[thread_id + 32];
                if (SIZE >= 32) buffer[thread_id] = value = value + buffer[thread_id + 16];
                if (SIZE >= 16) buffer[thread_id] = value = value + buffer[thread_id + 8];
                if (SIZE >= 8) buffer[thread_id] = value = value + buffer[thread_id + 4];
                if (SIZE >= 4) buffer[thread_id] = value = value + buffer[thread_id + 2];
                if (SIZE >= 2) buffer[thread_id] = value = value + buffer[thread_id + 1];
            } // 判断当前需要激活的线程是否少于32个
        }

        // 在每个 Block 已经完成累加的基础上, 进行全局的归约累加
        __global__
        void reduction_kernel(PtrStep<double> global_buffer, const int length, PtrStep<double> output)
        {
            double sum = 0.0;

            // 每个线程对应一个 block 的某项求和的结果, 获取之
            // 但是 blocks 可能很多, 这里是以512为一批进行获取, 加和处理的. 640x480只用到300个blocks.
            for (int t = threadIdx.x; t < length; t += 512)
                sum += *(global_buffer.ptr(blockIdx.x) + t);

            // 对于 GTX 1070, 每个 block 的 shared_memory 最大大小是 48KB, 足够使用了, 这里相当于只用了 1/12
            // 前面设置线程个数为这些, 也是为了避免每个 block 中的 shared memory 超标, 又能够尽可能地使用所有的 shared memory
            __shared__ double smem[512];

            // 注意超过范围的线程也能够执行到这里, 上面的循环不会执行, sum=0, 因此保存到 smem 对后面的归约过程没有影响
            smem[threadIdx.x] = sum;
            // 同时运行512个, 一个warp装不下,保险处理就是进行同步
            __syncthreads();

            // 512个线程都归约计算
            reduce<512>(smem);

            // 第0线程负责将每一项的最终求和结果进行转存
            if (threadIdx.x == 0)
                output.ptr(blockIdx.x)[0] = smem[0];
        };


        


        __global__
        void estimate_kernel(
            const Matf33 rotation_current,        // 上次迭代得到的旋转 Rwc
            const Matf31 translation_current,                                         // 上次迭代得到的平移 twc
            const PtrStep<float3> vertex_map_current,                                   // 当前帧对应图层的顶点图
            const PtrStep<float3> normal_map_current,                                   // 当前帧对应图层的法向图
            const Matf33 rotation_previous_inv,   // 上一帧相机的旋转, Rcw
            const Matf31 translation_previous,                                        // 上一帧相机的平移, twc
            const CameraParameters cam_params,                                          // 当前图层的相机内参
            const PtrStep<float3> vertex_map_previous,                                  // 上一帧相机位姿推理得到的表面顶点图
            const PtrStep<float3> normal_map_previous,                                  // 上一帧相机位姿推理得到的表面法向图
            const float distance_threshold,                                             // ICP 中关联匹配的最大距离阈值
            const float angle_threshold,                                                // ICP 中关联匹配的最大角度阈值
            const int cols,                                                             // 当前图层的图像列数
            const int rows,                                                             // 当前图层的图像行数
            PtrStep<double> global_buffer)                                              // 数据缓冲区, 暂存每个 block 中的累加和结果
        {
            // step 0 数据准备
            // 获取当前线程处理的像素坐标
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
        
            Matf31 n,         // 目标点的法向, KinectFusion中为上一帧的点云对应的法向量
                    d,         // 目标点,      KinectFusion中为上一帧的点云
                    s;         // 源点,        KinectFusion中为当前帧的点云
            // 匹配点的状态, 表示是否匹配, 初始值为 false
            bool correspondence_found = false;

            // step 1 当处理的像素位置合法时进行 // ? -- 进行投影数据关联
            if (x < cols && y < rows) {
                // step 1.1 获取当前帧点云法向量的x坐标, 判断其法向是否存在
                Matf31 normal_current;
                normal_current.x() = normal_map_current.ptr(y)[x].x;
                // 如果是个非数, 就认为这个法向是不存在的
                if (!isnan(normal_current.x())) {
                    // step 1.2 获取的点云法向量确实存在, 
                    // 获取当前帧的顶点
                    Matf31 vertex_current;
                    vertex_current.x() = vertex_map_current.ptr(y)[x].x;
                    vertex_current.y() = vertex_map_current.ptr(y)[x].y;
                    vertex_current.z() = vertex_map_current.ptr(y)[x].z;

                    // 将当前帧的顶点坐标转换到世界坐标系下 Pw = Rwc * Pc + twc
                    Matf31 vertex_current_global = rotation_current * vertex_current + translation_current;

                    // 这个顶点在上一帧相机坐标系下的坐标 Pc(k-1) = Rcw(k-1) * (Pw - twc(k-1)) 
                    // ! 这里就是为什么要对旋转求逆的原因了
                    Matf31 vertex_current_camera =
                            rotation_previous_inv * (vertex_current_global - translation_previous);

                    // 接着将该空间点投影到上一帧的图像中坐标系中
                    int point_x;
                    int point_y;
                    // __float2int_rd 向下舍入, +0.5 是为了实现"四舍五入"的效果
                    point_x = __float2int_rd(
                            vertex_current_camera.x() * cam_params.focal_x / vertex_current_camera.z() +
                            cam_params.principal_x + 0.5f);
                    point_y = __float2int_rd(
                            vertex_current_camera.y() * cam_params.focal_y / vertex_current_camera.z() +
                            cam_params.principal_y + 0.5f);

                    // 检查投影点是否在图像中
                    if (point_x >= 0 && point_y >= 0 && point_x < cols && point_y < rows &&
                        vertex_current_camera.z() >= 0) {

                        // 如果在的话, 说明数据关联有戏. 但是还需要检查两个地方
                        // 我们先获取上一帧的疑似关联点的法向
                        Matf31 normal_previous_global;
                        normal_previous_global.x() = normal_map_previous.ptr(point_y)[point_x].x;
                        // 如果它确认存在
                        if (!isnan(normal_previous_global.x())) {
                            // 获取对应顶点
                            Matf31 vertex_previous_global;
                            vertex_previous_global.x() = vertex_map_previous.ptr(point_y)[point_x].x;
                            vertex_previous_global.y() = vertex_map_previous.ptr(point_y)[point_x].y;
                            vertex_previous_global.z() = vertex_map_previous.ptr(point_y)[point_x].z;
                            // 距离检查, 如果顶点距离相差太多则认为不是正确的点
                            const float distance = (vertex_previous_global - vertex_current_global).norm();
                            if (distance <= distance_threshold) {
                                // 获取完整的当前帧该顶点的法向, 获取的过程移动到这里的主要目的也是为了避免不必要的计算
                                normal_current.y() = normal_map_current.ptr(y)[x].y;
                                normal_current.z() = normal_map_current.ptr(y)[x].z;
                                // 上面获取的法向是在当前帧相机坐标系下表示的, 这里需要转换到世界坐标系下的表示
                                Matf31 normal_current_global = rotation_current * normal_current;

                                // 同样获取完整的, 在上一帧中对应顶点的法向. 注意在平面推理阶段得到的法向就是在世界坐标系下的表示
                                // TODO 确认一下
                                normal_previous_global.y() = normal_map_previous.ptr(point_y)[point_x].y;
                                normal_previous_global.z() = normal_map_previous.ptr(point_y)[point_x].z;

                                // 通过计算叉乘得到两个向量夹角的正弦值. 由于 |axb|=|a||b|sin \alpha, 所以叉乘计算得到的向量的模就是 sin \alpha
                                const float sine = normal_current_global.cross(normal_previous_global).norm();
                                // ? 应该是夹角越大, sine 越大啊, 为什么这里是大于等于??? 
                                if (sine >= angle_threshold) {
                                    // 认为通过检查, 保存关联结果和产生的数据
                                    n = normal_previous_global;
                                    d = vertex_previous_global;
                                    s = vertex_current_global;

                                    correspondence_found = true;
                                }// 通过关联的角度检查
                            }// 通过关联的距离检查
                        }// 上一帧中的关联点有法向
                    }// 当前帧的顶点对应的空间点的对上一帧的重投影点在图像中
                }// 当前帧的顶点的法向量存在
            }// 当前线程处理的像素位置在图像范围中

            // 保存计算结果. 根据推导, 对于每个点, 对矩阵A贡献有6个元素, 对向量b贡献有一个元素
            float row[7];

            // 只有对成功匹配的点才会进行的操作. 这个判断也会滤除那些线程坐标不在图像中的线程, 这样做可以减少程序中的分支数目
            if (correspondence_found) {
                // // 前面的强制类型转换符号, 目测是为了转换成为 Eigen 中表示矩阵中浮点数元素的类型, 可以将计算结果直接一次写入到 row[0] row[1] row[2]
                // // 矩阵A中的两个主要元素
                // *(Matf31da*) &row[0] = s.cross(n);
                // *(Matf31da*) &row[3] = n;
                // // 矩阵b中当前点贡献的部分
                // row[6] = n.dot(d - s);
                Matf31 s_cross_n = s.cross(n);
                row[0] = s_cross_n.x00;
                row[1] = s_cross_n.x10;
                row[2] = s_cross_n.x20;
                row[3] = n.x00;
                row[4] = n.x10;
                row[5] = n.x20;
                // 矩阵b中当前点贡献的部分
                row[6] = n.dot(d - s);
            } else
                // 如果没有找到匹配的点, 或者说是当前线程的id不在图像区域中, 就全都给0
                // 这样反映在最后的结果中, 就是图像中的这个区域对最后的误差项没有任何贡献, 相当于不存在一样
                // 貌似这样计算量是多了,但是相比之下GPU更不适合在计算总矩阵A的时候进行多种分支的处理
                row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;

            // 存放在 shared_memory. 每一个 block 中的线程共享一个区域的shared_memory
            // smem = Shared MEMory
            __shared__ double smem[BLOCK_SIZE_X * BLOCK_SIZE_Y];
            // 计算当前线程的一维索引
            const int tid = threadIdx.y * blockDim.x + threadIdx.x;

            int shift = 0;
            for (int i = 0; i < 6; ++i) { // Rows
                for (int j = i; j < 7; ++j) { // Columns and B
                    // 同步当前线程块中的所有线程执行到这里, 避免出现竞争的情况
                    __syncthreads();
                    // 如果把向量中的每个元素都拆分出来的话, 可以发现本质上是对这27个元素累加, 如果我们拿到了最后这27项的累加和, 我们就可以构造矩阵A和向量b了
                    // 这里就是在计算其中的一项, 当前线程, 或者说当前的这个像素给的贡献
                    smem[tid] = row[i] * row[j];
                    // 再同步一次, 确保所有的线程都完成了写入操作, 避免出现"某个线程还在写数据,但是出现了另外的线程还在读数据"的情况
                    __syncthreads();

                    // Block 内对该元素归约 
                    // 调用这个函数的时候使用当前线程自己的线程id
                    // 因为我们最终的目的是对于这一项, 要将所有线程的贡献累加; 累加的过程分为两个阶段, 一个是每个block 内相加,而是对于所有的Block的和,再进行相加.
                    // 这里进行的是每个 block 中相加的一步
                    reduce<BLOCK_SIZE_X * BLOCK_SIZE_Y>(smem);

                    // 当前 block 中的线程#0 负责将归约之后的结果保存到 global_buffer 中. 
                    // shift 其实就是对应着"当前累加的和是哪一项"这一点; 当前block的结果先放在指定位置, 等全部完事之后再在每个block中的累加和已知的基础上,进行归约求和
                    if (tid == 0)
                        global_buffer.ptr(shift++)[gridDim.x * blockIdx.y + blockIdx.x] = smem[0];
                }
            }// 归约累加
        }


    
        // 使用GPU并行计算矩阵A和向量b
        void estimate_step(
            const Eigen::Matrix3f& rotation_current_eigenMf33,            // 上次迭代得到的旋转 Rwc
            const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_current_eigenMf31,                // 上次迭代得到的平移 twc
            const cv::cuda::GpuMat& vertex_map_current,         // 当前帧对应图层的的顶点图
            const cv::cuda::GpuMat& normal_map_current,         // 当前帧对应图层的的法向图
            const Eigen::Matrix3f& rotation_previous_inv_eigenMf33,       // 上一帧相机外参中的旋转的逆, Rcw
            const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_previous_eigenMf31,               // 上一帧相机的平移 twc
            const CameraParameters& cam_params,                 // 当前图层的相机内参
            const cv::cuda::GpuMat& vertex_map_previous,        // 对应图层的推理得到的平面顶点图
            const cv::cuda::GpuMat& normal_map_previous,        // 对应图层的推理得到的平面法向图
            float distance_threshold,                           // ICP迭代过程中视为外点的距离阈值
            float angle_threshold,                              // ICP迭代过程中视为外点的角度阈值(角度变正弦值)
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A,    // 计算得到的矩阵 A, 行优先
            Eigen::Matrix<double, 6, 1>& b)                     // 计算得到的向量 b
        {

            //将eigen矩阵转为自定义的矩阵，传入kernel中使用
            Matf33 rotation_current_Matf33 = mpe::eigenMatrix3fToMatf33(rotation_current_eigenMf33);
            Matf33 rotation_previous_inv_Matf33 = mpe::eigenMatrix3fToMatf33(rotation_previous_inv_eigenMf33);

            Matf31 translation_current_Matf31 = mpe::eigenMatrix3x1ToMatf31(translation_current_eigenMf31);
            Matf31 translation_previous_Matf31 = mpe::eigenMatrix3x1ToMatf31(translation_previous_eigenMf31);

            // step 0 计算需要的线程规模, 每个线程处理当前图像中的一个像素
            const int cols = vertex_map_current.cols;
            const int rows = vertex_map_current.rows;


            // 32 x 32, 但是这里相当于设置的 threads
            dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            // 这里还开了多个 Grid -- 但是这里相当于设置的 blocks
            dim3 grid(1, 1);
            grid.x = static_cast<unsigned int>(std::ceil(cols / block.x));
            grid.y = static_cast<unsigned int>(std::ceil(rows / block.y));

            cv::cuda::GpuMat global_buffer { cv::cuda::createContinuous(27, grid.x * grid.y, CV_64FC1) };
            cv::cuda::GpuMat sum_buffer { cv::cuda::createContinuous(27, 1, CV_64FC1) };

            // step 2.1 启动核函数, 对于图像上的每个像素执行: 数据关联, 计算误差贡献, 并且每个Block中累加本block的误差总贡献
            // 其实这里可以看到前面的 block 和 grid 都是相当于之前的 threads 和 blocks
            // 
            estimate_kernel<<<grid, block>>>(
                rotation_current_Matf33,                               // 上次迭代得到的旋转 Rwc
                translation_current_Matf31,                            // 上次迭代得到的平移 twc
                vertex_map_current,                             // 当前帧对应图层的顶点图
                normal_map_current,                             // 当前帧对应图层的法向图
                rotation_previous_inv_Matf33,                          // 上一帧相机外参的旋转, Rcw
                translation_previous_Matf31,                           // 上一帧相机的平移, twc
                cam_params,                                     // 对应图层的相机内参
                vertex_map_previous,                            // 上一帧位姿处推理得到的表面顶点图
                normal_map_previous,                            // 上一帧位姿处推理得到的表面法向图
                distance_threshold,                             // ICP 中关联匹配的最大距离阈值
                angle_threshold,                                // ICP 中关联匹配的最大角度阈值
                cols,                                           // 当前图层的图像列数
                rows,                                           // 当前图层的图像行数
                global_buffer);                                 // 暂存每个Block贡献和的缓冲区

            // step 2.2 在得到了每一个block累加和结果的基础上, 进行全局的归约累加
            reduction_kernel<<<27, 512>>>(                      // 27 = 项数, 512 对应blocks数目, 这里是一次批获取多少个 blocks 先前的和. 
                                                                //如果实际blocks数目超过这些, 超出的部分就类似归约形式累加, 直到累加后的blocks的数目小于512
                global_buffer,                                  // 每个Block累加的结果
                grid.x * grid.y,                                // 27项对应global_buffer中的27页,每一页中的每一个元素记录了一个block的累加结果,这里是每一页的尺寸
                sum_buffer);                                    // 输出, 结果是所有的匹配点对这27项的贡献

            // step 3 将 GPU 中计算好的矩阵A和向量b下载到CPU中，并且组装数据
            // 下载
            cv::Mat host_data { 27, 1, CV_64FC1 };
            sum_buffer.download(host_data);
            // 组装
            // 按照前面的推导, 矩阵A和向量b的最终形式使用rows[*]的下标表示分别为:
            //      | 0x0 0x1 0x2 0x3 0x4 0x5 |                 | 00 01 02 03 04 05 |
            //      | 0x1 1x1 1x2 1x3 1x4 1x5 |                 | 01 07 08 09 10 11 |
            //      | 0x2 1x2 2x2 2x3 2x4 2x5 |   按buffer下标   | 02 08 13 14 15 16 |  
            // A =  | 0x3 1x3 2x3 3x3 3x4 3x5 | =============== | 03 19 14 18 19 20 | => 斜三角对称矩阵, 只要构造上三角, 下三角对称复制就可以了
            //      | 0x4 1x4 2x4 3x4 4x4 4x5 |                 | 04 10 15 19 22 23 |
            //      | 0x5 1x5 2x5 3x5 4x5 5x5 |                 | 05 11 16 20 23 25 |
            //
            //      | 0x6 |                     | 06 |
            //      | 1x6 |                     | 12 |
            //      | 2x6 |   按buffer下标       | 17 |
            // b =  | 3x6 | ================    | 21 |  => j=6 都是 b
            //      | 4x6 |                     | 24 |
            //      | 5x6 |                     | 26 |
            //

            int shift = 0;
            for (int i = 0; i < 6; ++i) { // Rows
                for (int j = i; j < 7; ++j) { // Columns and B
                    // 获取值.[0]因为这个 host_data 就一列, []中只能填0
                    double value = host_data.ptr<double>(shift++)[0];
                    // j=6 都是 b
                    if (j == 6)
                        b.data()[i] = value;
                    else
                        A.data()[j * 6 + i] = A.data()[i * 6 + j]   // 对称赋值
                                            = value;
                }
            }
        }

    
    }
}



namespace mpe {
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
