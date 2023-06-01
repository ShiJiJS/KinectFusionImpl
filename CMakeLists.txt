
cmake_minimum_required(VERSION 3.10)

project(main_cmake LANGUAGES CXX CUDA)

# Use modern C++
set(CMAKE_CXX_STANDARD 14)

# Setting output paths
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# ------------------------------------------------
#                 Dependencies
# ------------------------------------------------
## CUDA
find_package(CUDA REQUIRED)
IF (CUDA_FOUND)
    include_directories("${CUDA_INCLUDE_DIRS}")
    message(STATUS "==> main_cmake Found CUDA： " ${CUDA_INCLUDE_DIRS})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --relocatable-device-code=true")
ENDIF ()

## OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_INCLUDE_DIRS)
    include_directories("${OpenCV_INCLUDE_DIRS}")
    message(STATUS "==> main_cmake Found OpenCV： " ${OpenCV_INCLUDE_DIRS})
endif (OpenCV_INCLUDE_DIRS)

## VTK
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

## Eigen3
find_package(Eigen3 REQUIRED)
if (EIGEN3_INCLUDE_DIR)
    include_directories("${EIGEN3_INCLUDE_DIR}")
    message(STATUS "==> main_cmake Found Eigen3： " ${EIGEN3_INCLUDE_DIR})
endif (EIGEN3_INCLUDE_DIR)

## PCL
find_package(PCL 1.10 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# ------------------------------------------------
include_directories(include)


file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})


