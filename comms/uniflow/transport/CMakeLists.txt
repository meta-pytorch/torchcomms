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

target_include_directories(uniflow_transport PUBLIC
    ${ROOT}
)

# Sub-components
add_subdirectory(nvlink)
add_subdirectory(rdma)
