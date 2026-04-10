# Copyright (c) Meta Platforms, Inc. and affiliates.

set(UNIFLOW_IBV_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/IbvApi.cpp
)

set(UNIFLOW_IBV_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/IbvApi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/IbvCore.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Mlx5Core.h
)

add_library(uniflow_ibv OBJECT
    ${UNIFLOW_IBV_SOURCES}
    ${UNIFLOW_IBV_HEADERS}
)

target_compile_features(uniflow_ibv PUBLIC cxx_std_20)

target_compile_options(uniflow_ibv PRIVATE -fPIC)

target_include_directories(uniflow_ibv PUBLIC
    ${ROOT}
)

target_link_libraries(uniflow_ibv PRIVATE
    ${CMAKE_DL_LIBS}
)
