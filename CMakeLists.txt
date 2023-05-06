cmake_minimum_required(VERSION 3.0)

project(main_cmake LANGUAGES CXX CUDA)

include_directories(include)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(VTK 7.1 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.10 REQUIRED)

include(${VTK_USE_FILE})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
# find_path(GLM_INCLUDE_DIR glm/glm.hpp)
# include_directories(${GLM_INCLUDE_DIR})


# 忽略警告用
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")

# add_executable(main_cmake src/surface_measurement.cpp src/cuda/surface_measurement.cu src/utils.cpp src/pipeline.cpp main.cpp)
include_directories(include)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})