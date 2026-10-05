# Copyright (c) Meta Platforms, Inc. and affiliates.

set(UNIFLOW_TRANSPORT_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/Transport.h
    ${CMAKE_CURRENT_SOURCE_DIR}/TransportType.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Topology.h
)

set(UNIFLOW_TRANSPORT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/Topology.cpp
)

add_library(uniflow_transport OBJECT
    ${UNIFLOW_TRANSPORT_HEADERS}
    ${UNIFLOW_TRANSPORT_SOURCES}
)

target_compile_features(uniflow_transport PUBLIC cxx_std_20)

target_compile_options(uniflow_transport PRIVATE -fPIC)

target_link_libraries(uniflow_transport PRIVATE uniflow_build_config)

# Sub-components. The intra-node GPU tier is vendor-split: NVLink on NVIDIA,
# P2P/XGMI on AMD. MultiTransport.cpp compiles NVLink out behind
# __HIP_PLATFORM_AMD__ and registers P2pTransportFactory in its place.
if(UNIFLOW_GPU_PLATFORM STREQUAL CUDA)
  add_subdirectory(nvlink)
else()
  add_subdirectory(p2p)
endif()
add_subdirectory(rdma)
if(UNIFLOW_ENABLE_TCP_TRANSPORT)
  add_subdirectory(tcp)
endif()
