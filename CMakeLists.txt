cmake_minimum_required(VERSION 3.0)

project(Test LANGUAGES CXX CUDA)

include_directories(include)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
find_package(PCL 1.10 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 忽略警告用
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcudafe \"--diag_suppress=code_is_unreachable\"")

add_executable(main_cmake src/surface_measurement.cpp src/cuda/surface_measurement.cu src/utils.cpp src/pipeline.cpp main.cpp)


target_link_libraries(main_cmake ${CUDA_LIBRARIES})
target_include_directories(main_cmake PRIVATE ${EIGEN3_INCLUDE_DIR})

target_link_libraries(main_cmake ${OpenCV_LIBS})
target_link_libraries(main_cmake ${PCL_LIBRARIES})