# Copyright (c) Meta Platforms, Inc. and affiliates.

set(UNIFLOW_RDMA_TRANSPORT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/RdmaTransport.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RdmaRegistrationHandle.cpp
)

set(UNIFLOW_RDMA_TRANSPORT_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/RdmaTransport.h
)

add_library(uniflow_rdma_transport OBJECT
    ${UNIFLOW_RDMA_TRANSPORT_SOURCES}
    ${UNIFLOW_RDMA_TRANSPORT_HEADERS}
)

target_compile_features(uniflow_rdma_transport PUBLIC cxx_std_20)

target_compile_options(uniflow_rdma_transport PRIVATE -fPIC)

target_include_directories(uniflow_rdma_transport PUBLIC
    ${ROOT}
)

target_link_libraries(uniflow_rdma_transport PUBLIC uniflow_ibv_api)
