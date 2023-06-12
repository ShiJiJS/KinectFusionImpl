# KinectFusionImpl

An implementation of the KinectFusion algorithm based on CPP CUDA, according to *Newcombe, Richard A., et al.* KinectFusion: Real-time dense surface mapping and tracking. This is a learning project, and thus real-time reconstruction has not been implemented yet. More information can be found in [our report](https://github.com/ShiJiJS/KinectFusionImpl/blob/main/readme_files/%E6%B7%B1%E5%BA%A6%E4%B8%8E%E9%A2%9C%E8%89%B2%E4%BF%A1%E6%81%AF%E8%9E%8D%E5%90%88%E7%9A%84%E5%AE%9E%E6%97%B6%E4%B8%89%E7%BB%B4%E9%87%8D%E5%BB%BA%EF%BC%9A%E5%9F%BA%E4%BA%8EKinectFusion%E7%9A%84%E6%96%B9%E6%B3%95.pdf), which is currently available only in Chinese.

## Dependencies

**CUDA 11.3** or lower versions above CUDA 8.0 are also feasible. We ran it on CUDA 11.3.

**OpenCV 3.0** or higher.

**Eigen3**: For efficient matrix and vector operations (only on CPU). We attempted to run it on GPU, but there seemed to be some issues, so we only used Eigen on CPU.

We set up the environment on Ubuntu20.04's WSL2 and used VS Code for development and debugging. If you wish to adopt the same configuration, you may find the [configuration documentation](https://github.com/ShiJiJS/KinectFusionImpl/blob/main/readme_files/%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE.pdf) helpful.

## How to run the code?

First, you need to clone the code to your local and set up the environment.

You can compile the project directly using the cmake and make commands.

Regarding data input, we support the TUM and ICL-NUIM datasets. In main.cpp, you can find the logic for reading these datasets. The ICL-NIUM dataset is used by default. You only need to comment out part of it and recompile to switch the dataset to be read.

When using the ICL-NUIM dataset, please use the TUM compatible version and place the depth and rgb folders and associations.txt in the directory where the executable file is located.

When using the TUM dataset, please place the rgb, depth folders and depth.txt, rgb.txt in the directory where the executable file is located.

Also, you need to input parameters before execution. Please place a configuration file named config.txt in the directory of the executable file.

The following parameters are required. Please adjust according to different datasets:

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

The meanings of these parameters can be found in Configuration.h.

## Results

After the reconstruction, we exported the results to MeshLab to generate a mesh. The results are shown below, with datasets from cvg.cit.tum.de

![desk_mesh](https://github.com/ShiJiJS/KinectFusionImpl/blob/main/readme_files/images/desk_mesh.png)

For scene reconstruction, we tested the algorithm using the ICL-NUIM dataset and imported it into CloudCompare for error analysis. The results of the kt2 scene are as follows. Specific result analysis can be found in the testing part of [our report](https://github.com/ShiJiJS/KinectFusionImpl/blob/main/readme_files/%E6%B7%B1%E5%BA%A6%E4%B8%8E%E9%A2%9C%E8%89%B2%E4%BF%A1%E6%81%AF%E8%9E%8D%E5%90%88%E7%9A%84%E5%AE%9E%E6%97%B6%E4%B8%89%E7%BB%B4%E9%87%8D%E5%BB%BA%EF%BC%9A%E5%9F%BA%E4%BA%8EKinectFusion%E7%9A%84%E6%96%B9%E6%B3%95.pdf).

![fullview](https://github.com/ShiJiJS/KinectFusionImpl/blob/main/readme_files/images/fullview.png)

![Histogram](https://github.com/ShiJiJS/KinectFusionImpl/blob/main/readme_files/images/Histogram.png)

## Reference

This project is modified based on the **[KinectFusionLib](https://github.com/chrdiller/KinectFusionLib)**, and referred to the annotations of [KinectFusionAppLib_comments](https://github.com/DreamWaterFound/KinectFusionAppLib_comments). We are very grateful to the contributors of the above two repositories.