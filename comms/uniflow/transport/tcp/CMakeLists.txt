# Copyright (c) Meta Platforms, Inc. and affiliates.

# The two sources that drive VRAM host-staging name cuda* types and are
# translated on AMD; their headers stay neutral, as does the handle source.
uniflow_hipify(UNIFLOW_TCP_TRANSPORT_TRANSLATED
    ${CMAKE_CURRENT_SOURCE_DIR}/TcpPinnedSlabPool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TcpTransport.cpp
)

set(UNIFLOW_TCP_TRANSPORT_SOURCES
    ${UNIFLOW_TCP_TRANSPORT_TRANSLATED}
    ${CMAKE_CURRENT_SOURCE_DIR}/TcpRegistrationHandle.cpp
)

set(UNIFLOW_TCP_TRANSPORT_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/TcpPinnedSlabPool.h
    ${CMAKE_CURRENT_SOURCE_DIR}/TcpRegistrationHandle.h
    ${CMAKE_CURRENT_SOURCE_DIR}/TcpTransport.h
    ${CMAKE_CURRENT_SOURCE_DIR}/TcpWireProtocol.h
)

add_library(uniflow_tcp_transport OBJECT
    ${UNIFLOW_TCP_TRANSPORT_SOURCES}
    ${UNIFLOW_TCP_TRANSPORT_HEADERS}
)

target_compile_features(uniflow_tcp_transport PUBLIC cxx_std_20)

target_compile_options(uniflow_tcp_transport PRIVATE -fPIC)

target_include_directories(uniflow_tcp_transport PUBLIC
    ${ROOT}
)

uniflow_target_hipified(uniflow_tcp_transport)
