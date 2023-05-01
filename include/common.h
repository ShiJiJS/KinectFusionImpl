//定义参数使用
#pragma once

namespace kinectfusion{
    namespace config{
        //相机参数
        constexpr int IMAGE_WIDTH = 640;        //图像宽度
        constexpr int IMAGE_HEIGHT = 480;       //图像高度
        constexpr float FOCAL_X = 525.0;        //焦距x
        constexpr float FOCAL_Y = 525.0;        //焦距y
        constexpr float PRINCIPAL_X = 319.5;    //光心x
        constexpr float PRINCIPAL_Y = 239.5;    //光心y

        //生成金字塔参数
        constexpr int NUM_LEVELS = 3;//生成金字塔的层数
        //滤波参数
        // kernel_size：5或7。较大的核可能会导致较大的计算成本，但也可能提供更好的滤波效果。
        // color_sigma：取决于深度图像的数值范围。通常，可以从范围的10%到30%开始尝试。
        // spatial_sigma：取决于图像的尺寸和噪声水平。可以从5到15的范围内尝试不同的值。  
        constexpr int KERNAL_SIZE = 5;// 双边滤波器使用的窗口（核）大小
        constexpr float DEPTH_CUTOFF = 100000.f;//截断深度
        constexpr float COLOR_SIGMA = 1.f;// 值域滤波的方差
        constexpr float SPATIAL_SIGMA = 1.f;// 空间域滤波的方差

        //ICP配准参数
        // distance_threshold：这个参数表示在 ICP 匹配过程中视为外点的距离差。匹配点对之间的距离如果大于这个阈值，它们将被视为外点并从配准过程中排除。
        //                     这有助于减小错误匹配点对对配准结果的影响。推荐取值范围通常在 0.01 到 0.1 米之间，具体取值需要根据应用场景和传感器的精度来确定。
        // angle_threshold：这个参数表示在 ICP 匹配过程中视为外点的角度差（以度为单位）。
                                // 匹配点对之间的法向量夹角如果大于这个阈值，它们将被视为外点并从配准过程中排除。这同样有助于减小错误匹配点对对配准结果的影响。
                                // 推荐取值范围通常在 1 到 30 度之间，具体取值需要根据应用场景和传感器的精度来确定。
        // icp_iterations：这个参数表示 ICP 配准过程的迭代次数。增加迭代次数可能会提高配准精度，但也会增加计算时间。
                                // 推荐的取值范围通常在 5 到 50 之间，具体取值需要根据实时性要求和计算资源来确定。
        constexpr float DISTANCE_THRESHOLD { 200.f };
        constexpr float ANGLE_THRESHOLD { 50.f };
        // constexpr std::vector<int> ICP_ITERATIONS {10, 5, 4};// 即第一层迭代10次,第二层5次,第三层4次
    }
}
