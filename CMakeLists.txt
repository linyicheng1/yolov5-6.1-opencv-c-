cmake_minimum_required(VERSION 3.5)
project(detect_test)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -fopenmp")

#查找必要的依赖包
find_package(OpenCV REQUIRED)
set(OpenVINO_DIR /opt/intel/openvino_2022/runtime/cmake/)
find_package(OpenVINO REQUIRED)

# 编译detector.so
add_library (detector SHARED ${CMAKE_CURRENT_SOURCE_DIR}/detector.cpp)
target_include_directories(detector
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PUBLIC ${OpenCV_INCLUDE_DIR}
)
target_link_libraries(detector
        ${OpenCV_LIBS}
        openvino::runtime
)

add_executable(detect_test main.cpp utils.cpp)
target_link_libraries(detect_test
        ${OpenCV_LIBS}
        detector
)