# Copyright (c) Meta Platforms, Inc. and affiliates.

set(UNIFLOW_CONTROLLER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/TcpController.cpp
)

set(UNIFLOW_CONTROLLER_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/Controller.h
    ${CMAKE_CURRENT_SOURCE_DIR}/TcpController.h
)

add_library(uniflow_controller OBJECT
    ${UNIFLOW_CONTROLLER_SOURCES}
    ${UNIFLOW_CONTROLLER_HEADERS}
)

target_compile_features(uniflow_controller PUBLIC cxx_std_20)

target_compile_options(uniflow_controller PRIVATE -fPIC)

target_include_directories(uniflow_controller PUBLIC
    ${ROOT}
)

target_link_libraries(uniflow_controller PRIVATE
    uniflow_executor
    uniflow_logging
)
