cmake_minimum_required(VERSION 3.18)
project(npp_edge_cimg LANGUAGES CXX CUDA)

# -----------------------
# Language standards
# -----------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -----------------------
# CUDA Toolkit (NPP)
# -----------------------
find_package(CUDAToolkit REQUIRED)

# -----------------------
# Include directories
# -----------------------
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# -----------------------
# Source files
# -----------------------
set(SRC
    src/edge_npp.cu
)

# -----------------------
# Executable
# -----------------------
add_executable(npp_edge ${SRC})
target_compile_definitions(npp_edge PRIVATE
    cimg_display=0
    cimg_use_jpeg
)

# -----------------------
# Link libraries
# -----------------------
target_link_libraries(npp_edge
    PRIVATE CUDA::cudart
    PRIVATE CUDA::nppc
    PRIVATE CUDA::nppif
    PRIVATE CUDA::nppig
)
set(JPEG_LIBRARIES "/home/styx/Installs/envs/cuda/.pixi/envs/default/lib/libjpeg.so") # Adjust this path as necessary
set(JPEG_INCLUDE_DIR "/home/styx/Installs/envs/cuda/.pixi/envs/default/include") # Adjust this path as necessary
target_link_libraries(npp_edge PRIVATE ${JPEG_LIBRARIES})
target_include_directories(npp_edge PRIVATE ${JPEG_INCLUDE_DIR})

# -----------------------
# Enable separable compilation (recommended for CUDA)
# -----------------------
set_target_properties(npp_edge PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ----------
