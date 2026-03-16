# Copyright (c) Meta Platforms, Inc. and affiliates.

add_library(ib_hca_parser OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/IbHcaParser.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/IbHcaParser.h
)

target_compile_features(ib_hca_parser PUBLIC cxx_std_20)
target_compile_options(ib_hca_parser PRIVATE -fPIC)
target_include_directories(ib_hca_parser PUBLIC ${ROOT})

add_library(nic_discovery OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/NicDiscovery.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/NicDiscovery.h
)

target_compile_features(nic_discovery PUBLIC cxx_std_20)
target_compile_options(nic_discovery PRIVATE -fPIC)
target_include_directories(nic_discovery PUBLIC ${ROOT})

target_link_libraries(nic_discovery PUBLIC
    ib_hca_parser
)

# CUDA Toolkit — provides cudart for GpuNicDiscovery::getCudaPciBusId().
find_package(CUDAToolkit REQUIRED)
target_link_libraries(nic_discovery PUBLIC CUDA::cudart)

# spdlog — header-only logging library used throughout NicDiscovery.
find_package(spdlog REQUIRED)
target_link_libraries(nic_discovery PUBLIC spdlog::spdlog)
