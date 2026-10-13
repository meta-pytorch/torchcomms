# Copyright (c) Meta Platforms, Inc. and affiliates.

add_library(uniflow_topology_discovery OBJECT TopologyDiscovery.cpp)
target_link_libraries(uniflow_topology_discovery PRIVATE
  uniflow_build_config
  uniflow_logging
  uniflow_transport
)

add_subdirectory(sysfs)
if(UNIFLOW_ENABLE_RDMA)
  add_subdirectory(ibverbs)
endif()
add_subdirectory(topology)

add_library(uniflow_device_adapter_access OBJECT
  HostDeviceAdapter.cpp
  DeviceAdapterAccess.cpp
)
target_link_libraries(uniflow_device_adapter_access PRIVATE
  uniflow_build_config
  uniflow_registry
  uniflow_gpu_topology_cache
)

add_library(uniflow_drivers INTERFACE)
target_link_libraries(uniflow_drivers INTERFACE
  uniflow_topology_discovery
  uniflow_sysfs
  uniflow_nic_topology
  uniflow_gpu_topology_cache
  uniflow_device_adapter_access
)
target_sources(uniflow_drivers INTERFACE
  $<TARGET_OBJECTS:uniflow_topology_discovery>
  $<TARGET_OBJECTS:uniflow_sysfs>
  $<TARGET_OBJECTS:uniflow_nic_topology>
  $<TARGET_OBJECTS:uniflow_gpu_topology_cache>
  $<TARGET_OBJECTS:uniflow_device_adapter_access>
)
if(UNIFLOW_ENABLE_RDMA)
  target_link_libraries(uniflow_drivers INTERFACE uniflow_ibv)
  target_sources(uniflow_drivers INTERFACE $<TARGET_OBJECTS:uniflow_ibv>)
endif()

if(NOT UNIFLOW_GPU_PLATFORM STREQUAL NONE)
  add_subdirectory(nvml)
  add_subdirectory(cuda)
  target_link_libraries(uniflow_drivers INTERFACE uniflow_cuda uniflow_nvml)
  target_sources(uniflow_drivers INTERFACE
    $<TARGET_OBJECTS:uniflow_cuda>
    $<TARGET_OBJECTS:uniflow_nvml>
  )
else()
  target_sources(uniflow_topology_discovery PRIVATE EmptyTopologyDiscovery.cpp)
endif()
