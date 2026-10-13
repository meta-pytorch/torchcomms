# Copyright (c) Meta Platforms, Inc. and affiliates.

add_library(uniflow_nic_topology OBJECT
  PciTopology.cpp
  NicTopologyDiscovery.cpp
)
target_link_libraries(uniflow_nic_topology PRIVATE uniflow_build_config)
if(UNIFLOW_ENABLE_RDMA)
  target_link_libraries(uniflow_nic_topology PRIVATE uniflow_ibv)
endif()

add_library(uniflow_gpu_topology_cache OBJECT GpuTopologyCache.cpp)
target_link_libraries(uniflow_gpu_topology_cache PRIVATE
  uniflow_build_config
  uniflow_logging
  uniflow_registry
)
