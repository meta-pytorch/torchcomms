# Copyright (c) Meta Platforms, Inc. and affiliates.

add_subdirectory(cuda)
add_subdirectory(ibverbs)
add_subdirectory(nvml)
add_subdirectory(sysfs)

# The platform-neutral half of topology discovery; the vendor half lives with
# the GPU driver sources.
add_library(uniflow_drivers OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/TopologyDiscovery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TopologyDiscovery.h
)

target_compile_features(uniflow_drivers PUBLIC cxx_std_20)

target_compile_options(uniflow_drivers PRIVATE -fPIC)

target_link_libraries(uniflow_drivers PRIVATE uniflow_build_config)
