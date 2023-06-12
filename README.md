

| [English](https://github.com/ShiJiJS/KinectFusionImpl/blob/main/readme_files/README_EN.md) | 简体中文 | 

# KinectFusionImpl

一个基于CPP CUDA的Kinectfusion算法实现，参考 *Newcombe, Richard A.等人的*  **KinectFusion: Real-time dense surface mapping and tracking.** 这只是一个学习项目，因此暂未实现实时重建。更多信息可以在我们的[报告](https://github.com/ShiJiJS/KinectFusionImpl/blob/main/readme_files/%E6%B7%B1%E5%BA%A6%E4%B8%8E%E9%A2%9C%E8%89%B2%E4%BF%A1%E6%81%AF%E8%9E%8D%E5%90%88%E7%9A%84%E5%AE%9E%E6%97%B6%E4%B8%89%E7%BB%B4%E9%87%8D%E5%BB%BA%EF%BC%9A%E5%9F%BA%E4%BA%8EKinectFusion%E7%9A%84%E6%96%B9%E6%B3%95.pdf)中查看，不过暂时只有中文版本。

## Dependencies

**CUDA 11.3** 或者一些在CUDA8.0以上的低一些的版本也可以，我们是在CUDA11.3上运行的

**OpenCV 3.0** or higher. 

**Eigen3**: 用于高效的矩阵和向量操作（仅在CPU上）。我们尝试将其在GPU上运行，但似乎会出现一些问题，因此仅在CPU上使用了Eigen.

我们在Ubuntu20.04的WSL2上配置了环境，使用VS Code 来进行开发和调试，如果你想采用同样的配置方式也许可以参考[配置文档](https://github.com/ShiJiJS/KinectFusionImpl/blob/main/readme_files/%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE.pdf)，或许会有帮助。

## How to run the code?

首先你需要将代码clone到本地，并配置好环境。

采用cmake和make命令直接编译项目即可。

在数据读入方面，我们支持了TUM和ICL-NUIM两个数据集。在main.cpp中，你可以找到读入这些的逻辑，默认使用ICL-NIUM数据集，你只需要将其中一部分注释掉，并重新编译，即可切换要读入的数据集。

在使用ICL-NUIM数据集时，请使用TUM兼容的版本，并将depth，rgb两个文件夹和associations.txt放置于可执行文件位于的目录中。

在使用TUM数据集时，请将rgb,depth文件夹和depth.txt,rgb.txt放置于可执行文件的目录中。

同时在执行前也需要录入参数。请在可执行文件的目录下放置一个名为config.txt的配置文件

需要包含以下参数，请根据数据集的不同来调整

```
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FOCAL_X = 525.0
FOCAL_Y = 525.0
PRINCIPAL_X = 319.5
PRINCIPAL_Y = 239.5
NUM_LEVELS = 3
KERNAL_SIZE = 5
DEPTH_CUTOFF = 10000.f
COLOR_SIGMA = 1.f
SPATIAL_SIGMA = 1.f
DISTANCE_THRESHOLD = 200.f
ANGLE_THRESHOLD = 30.f
DETETMINANT_THRESHOLD = 1000.f
VOXEL_SCALE = 10.f
TRUNCATION_DISTANCE = 200.f
INIT_DEPTH = 1000.f
DEPTH_SCALE = 5000
```



## Results

我们在重建之后将结果导出到了meshlab中生成了网格。结果如下所示，数据集来自于cvg.cit.tum.de

![desk_mesh](https://github.com/ShiJiJS/KinectFusionImpl/blob/main/readme_files/images/desk_mesh.png)

同时对于场景重建，我们使用了ICL-NUIM数据集对算法进行了测试，将其导入到CloudCompare中分析误差。其中kt2场景的结果如下，具体结果分析可以参考我们的[报告](https://github.com/ShiJiJS/KinectFusionImpl/blob/main/readme_files/%E6%B7%B1%E5%BA%A6%E4%B8%8E%E9%A2%9C%E8%89%B2%E4%BF%A1%E6%81%AF%E8%9E%8D%E5%90%88%E7%9A%84%E5%AE%9E%E6%97%B6%E4%B8%89%E7%BB%B4%E9%87%8D%E5%BB%BA%EF%BC%9A%E5%9F%BA%E4%BA%8EKinectFusion%E7%9A%84%E6%96%B9%E6%B3%95.pdf)的测试部分。

![fullview](https://github.com/ShiJiJS/KinectFusionImpl/blob/main/readme_files/images/fullview.png)

![Histogram](https://github.com/ShiJiJS/KinectFusionImpl/blob/main/readme_files/images/Histogram.png)



## Reference

本项目是在[KinectFusionLib](https://github.com/chrdiller/KinectFusionLib)的基础上修改而来，并参照了[KinectFusionAppLib_comments](https://github.com/DreamWaterFound/KinectFusionAppLib_comments)的注解，在此非常感谢以上两个仓库的贡献者。
