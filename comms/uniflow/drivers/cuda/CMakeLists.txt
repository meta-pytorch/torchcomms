# Copyright (c) Meta Platforms, Inc. and affiliates.

set(UNIFLOW_CUDA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/CudaApi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CudaDriverApi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CudaDeviceAdapter.cpp
)

# Every source here names cuda*/cu* types, so all of them are translated on AMD,
# together with the headers whose declarations carry those types.
uniflow_hipify(UNIFLOW_CUDA_SOURCES ${UNIFLOW_CUDA_SOURCES})
uniflow_hipify(UNIFLOW_CUDA_TRANSLATED_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/CudaApi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CudaDriverApi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CudaDevicePtr.h
)

add_library(uniflow_cuda OBJECT
    ${UNIFLOW_CUDA_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/CudaTopologyDiscovery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CudaTopologyDiscovery.h
    ${UNIFLOW_CUDA_TRANSLATED_HEADERS}
    # Carries no vendor types, so it is shared as-is.
    ${CMAKE_CURRENT_SOURCE_DIR}/CudaDeviceAdapter.h
    # Never translated: its CUDA-spelling aliases must keep their names whichever
    # tool translates the including file.
    ${CMAKE_CURRENT_SOURCE_DIR}/CudaDriverApiHipCompat.h
)

uniflow_target_hipified(uniflow_cuda)

target_compile_features(uniflow_cuda PUBLIC cxx_std_20)

target_compile_options(uniflow_cuda PRIVATE -fPIC)

target_include_directories(uniflow_cuda PUBLIC
    ${ROOT}
)

target_link_libraries(uniflow_cuda PUBLIC ${UNIFLOW_GPU_LIBRARIES})
