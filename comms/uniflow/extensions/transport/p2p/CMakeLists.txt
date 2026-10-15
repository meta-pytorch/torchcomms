# Copyright (c) Meta Platforms, Inc. and affiliates.

# Intra-node GPU P2P transport (XGMI on AMD), the counterpart to NVLink on
# NVIDIA. Only the transfer source names cuda* types and gets translated; the
# header stays in neutral types (void* stream at the boundary) so MultiTransport
# includes it without being translated itself.
uniflow_hipify(UNIFLOW_P2P_TRANSPORT_TRANSLATED_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/P2pTransport.cpp
)

set(UNIFLOW_P2P_TRANSPORT_SOURCES
    ${UNIFLOW_P2P_TRANSPORT_TRANSLATED_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/P2pRegistrationHandle.cpp
)

set(UNIFLOW_P2P_TRANSPORT_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/P2pTransport.h
    ${CMAKE_CURRENT_SOURCE_DIR}/P2pRegistrationHandle.h
)

add_library(uniflow_p2p_transport OBJECT
    ${UNIFLOW_P2P_TRANSPORT_SOURCES}
    ${UNIFLOW_P2P_TRANSPORT_HEADERS}
)

target_compile_features(uniflow_p2p_transport PUBLIC cxx_std_20)

target_compile_options(uniflow_p2p_transport PRIVATE -fPIC)

target_link_libraries(uniflow_p2p_transport PRIVATE uniflow_build_config)

uniflow_target_hipified(uniflow_p2p_transport)
