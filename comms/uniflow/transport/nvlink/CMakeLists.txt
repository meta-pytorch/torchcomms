# Copyright (c) Meta Platforms, Inc. and affiliates.

set(UNIFLOW_NVLINK_TRANSPORT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/NVLinkRegistrationHandle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NVLinkTopology.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NVLinkTransport.cpp
)

set(UNIFLOW_NVLINK_TRANSPORT_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/NVLinkTransport.h
)

add_library(uniflow_nvlink_transport OBJECT
    ${UNIFLOW_NVLINK_TRANSPORT_SOURCES}
    ${UNIFLOW_NVLINK_TRANSPORT_HEADERS}
)

target_compile_features(uniflow_nvlink_transport PUBLIC cxx_std_20)

target_compile_options(uniflow_nvlink_transport PRIVATE -fPIC)

target_include_directories(uniflow_nvlink_transport PUBLIC
    ${ROOT}
)
