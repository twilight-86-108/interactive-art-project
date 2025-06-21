cmake_minimum_required(VERSION 3.18)
project(AquaMirrorGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

# CUDA設定
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX4060対応

# ソースファイル
set(SOURCES
    src/gpu_processor.cpp
    src/image_processing.cpp
    cuda/resize_kernel.cu
    cuda/color_convert_kernel.cu
)

# ライブラリ作成
add_library(aqua_mirror_gpu SHARED ${SOURCES})

target_link_libraries(aqua_mirror_gpu 
    ${OpenCV_LIBS}
    cudart
    cuda
)

target_include_directories(aqua_mirror_gpu PUBLIC
    include
    ${OpenCV_INCLUDE_DIRS}
)
