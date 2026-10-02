# Copyright (c) Meta Platforms, Inc. and affiliates.

add_library(uniflow_topology_discovery OBJECT TopologyDiscovery.cpp)
target_link_libraries(uniflow_topology_discovery PRIVATE
  uniflow_build_config
  uniflow_logging
  uniflow_transport
)

add_library(uniflow_drivers INTERFACE)
target_link_libraries(uniflow_drivers INTERFACE uniflow_topology_discovery)

if(UNIFLOW_ENABLE_CUDA)
  add_subdirectory(nvml)
  add_subdirectory(sysfs)
  if(UNIFLOW_ENABLE_RDMA)
    add_subdirectory(ibverbs)
  endif()
  add_subdirectory(cuda)
  target_link_libraries(uniflow_drivers INTERFACE
    uniflow_cuda
    uniflow_nvml
    uniflow_sysfs
  )
  if(UNIFLOW_ENABLE_RDMA)
    target_link_libraries(uniflow_drivers INTERFACE uniflow_ibv)
  endif()
else()
  target_sources(uniflow_topology_discovery PRIVATE EmptyTopologyDiscovery.cpp)
endif()
