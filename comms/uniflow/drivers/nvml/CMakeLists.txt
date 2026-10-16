# Copyright (c) Meta Platforms, Inc. and affiliates.

# NVML is NVIDIA-only. On HIP this object library has no implementation
# sources. Extensions supplies the no-op NvmlApiStub.cpp to uniflow;
# HIP builds require UNIFLOW_BUILD_EXTENSIONS=ON.
if(UNIFLOW_GPU_PLATFORM STREQUAL CUDA)
  set(UNIFLOW_NVML_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/NvmlApi.cpp)
else()
  set(UNIFLOW_NVML_SOURCES)
endif()

set(UNIFLOW_NVML_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/NvmlApi.h
)

add_library(uniflow_nvml OBJECT
    ${UNIFLOW_NVML_SOURCES}
    ${UNIFLOW_NVML_HEADERS}
)

set_target_properties(uniflow_nvml PROPERTIES LINKER_LANGUAGE CXX)

target_compile_features(uniflow_nvml PUBLIC cxx_std_20)

target_compile_options(uniflow_nvml PRIVATE -fPIC)

target_link_libraries(uniflow_nvml PRIVATE
    uniflow_build_config
    ${CMAKE_DL_LIBS}
)
