# Copyright (c) Meta Platforms, Inc. and affiliates.

add_library(uniflow_rdma_transport OBJECT
    RdmaTransport.cpp
    RdmaSlabPool.cpp
    CopyEngine.cpp
    RdmaRegistrationHandle.cpp
    RdmaResources.cpp
    RdmaTransportProvider.cpp
    RdmaTransport.h
    RdmaSlabPool.h
    CopyEngine.h
    RdmaRegistrationHandle.h
    RdmaResources.h
    RdmaTransportProvider.h
)

target_compile_features(uniflow_rdma_transport PUBLIC cxx_std_20)
target_compile_options(uniflow_rdma_transport PRIVATE -fPIC)
target_include_directories(uniflow_rdma_transport BEFORE PRIVATE
    "$<BUILD_INTERFACE:${UNIFLOW_BUILD_INCLUDE_DIR}>")
target_link_libraries(uniflow_rdma_transport PUBLIC
    uniflow_build_config
    uniflow_ibv
    uniflow_transport
    uniflow_device_adapter_access
    uniflow_nic_topology
)

if(NOT UNIFLOW_GPU_PLATFORM STREQUAL NONE)
  uniflow_hipify(UNIFLOW_RDMA_COMPATIBILITY_SOURCES
      ${CMAKE_CURRENT_SOURCE_DIR}/../../drivers/cuda/RdmaCompatibility.cpp)
  add_library(uniflow_rdma_compatibility OBJECT
      ${UNIFLOW_RDMA_COMPATIBILITY_SOURCES})
  uniflow_target_hipified(uniflow_rdma_compatibility)
  target_compile_features(uniflow_rdma_compatibility PUBLIC cxx_std_20)
  target_compile_options(uniflow_rdma_compatibility PRIVATE -fPIC)
  target_link_libraries(uniflow_rdma_compatibility PRIVATE
      uniflow_build_config
      uniflow_rdma_transport
      uniflow_cuda
      uniflow_topology_discovery
  )
endif()
