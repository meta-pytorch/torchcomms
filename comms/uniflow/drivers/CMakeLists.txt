# Copyright (c) Meta Platforms, Inc. and affiliates.

add_subdirectory(cuda)
add_subdirectory(ibverbs)
add_subdirectory(nvml)
add_subdirectory(sysfs)

# Platform-neutral topology discovery; createDefaultDiscoveryBackend() comes
# from the platform backend (CUDA here, selected via Buck internally).
add_library(uniflow_topology_discovery OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/TopologyDiscovery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TopologyDiscovery.h
)

target_compile_features(uniflow_topology_discovery PUBLIC cxx_std_20)

target_compile_options(uniflow_topology_discovery PRIVATE -fPIC)

target_include_directories(uniflow_topology_discovery PUBLIC
    ${ROOT}
)
