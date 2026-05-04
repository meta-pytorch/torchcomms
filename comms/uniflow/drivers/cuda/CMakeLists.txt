# Copyright (c) Meta Platforms, Inc. and affiliates.

set(UNIFLOW_CUDA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/CudaApi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CudaDriverApi.cpp
)

set(UNIFLOW_CUDA_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/CudaApi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CudaDriverApi.h
)

add_library(uniflow_cuda OBJECT
    ${UNIFLOW_CUDA_SOURCES}
    ${UNIFLOW_CUDA_HEADERS}
)

target_compile_features(uniflow_cuda PUBLIC cxx_std_20)

target_compile_options(uniflow_cuda PRIVATE -fPIC)

target_include_directories(uniflow_cuda PUBLIC
    ${ROOT}
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(uniflow_cuda PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
)
