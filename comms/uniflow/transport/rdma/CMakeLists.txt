# Copyright (c) Meta Platforms, Inc. and affiliates.

# The sources that name cuda* types are translated on AMD, together with the
# headers that declare those types; RdmaRegistrationHandle.cpp stays neutral.
uniflow_hipify(UNIFLOW_RDMA_TRANSPORT_TRANSLATED
    ${CMAKE_CURRENT_SOURCE_DIR}/RdmaTransport.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RdmaSlabPool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CopyEngine.cpp
)

uniflow_hipify(UNIFLOW_RDMA_TRANSPORT_TRANSLATED_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/RdmaTransport.h
    ${CMAKE_CURRENT_SOURCE_DIR}/RdmaSlabPool.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CopyEngine.h
)

set(UNIFLOW_RDMA_TRANSPORT_SOURCES
    ${UNIFLOW_RDMA_TRANSPORT_TRANSLATED}
    ${CMAKE_CURRENT_SOURCE_DIR}/RdmaRegistrationHandle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RdmaResources.cpp
)

set(UNIFLOW_RDMA_TRANSPORT_HEADERS ${UNIFLOW_RDMA_TRANSPORT_TRANSLATED_HEADERS})

add_library(uniflow_rdma_transport OBJECT
    ${UNIFLOW_RDMA_TRANSPORT_SOURCES}
    ${UNIFLOW_RDMA_TRANSPORT_HEADERS}
)

target_compile_features(uniflow_rdma_transport PUBLIC cxx_std_20)

target_compile_options(uniflow_rdma_transport PRIVATE -fPIC)

target_link_libraries(uniflow_rdma_transport PUBLIC
    uniflow_build_config
    uniflow_cuda
    uniflow_ibv
    uniflow_transport
)

uniflow_target_hipified(uniflow_rdma_transport)
