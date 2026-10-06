# Copyright (c) Meta Platforms, Inc. and affiliates.

cmake_minimum_required(VERSION 3.20)
project(uniflow_consumer LANGUAGES CXX)

find_package(uniflow CONFIG REQUIRED)

set(_UNIFLOW_HEADER_CHECK_SOURCES)
foreach(_UNIFLOW_HEADER_PATH IN LISTS uniflow_PUBLIC_HEADERS)
  string(MAKE_C_IDENTIFIER "${_UNIFLOW_HEADER_PATH}" _UNIFLOW_HEADER_ID)
  set(_UNIFLOW_HEADER_CHECK_SOURCE
    "${CMAKE_CURRENT_BINARY_DIR}/${_UNIFLOW_HEADER_ID}.cpp")
  file(GENERATE OUTPUT "${_UNIFLOW_HEADER_CHECK_SOURCE}"
    CONTENT "#include <comms/uniflow/${_UNIFLOW_HEADER_PATH}>\n")
  list(APPEND _UNIFLOW_HEADER_CHECK_SOURCES
    "${_UNIFLOW_HEADER_CHECK_SOURCE}")
endforeach()

add_library(uniflow_public_header_checks OBJECT
  ${_UNIFLOW_HEADER_CHECK_SOURCES})
target_link_libraries(uniflow_public_header_checks PRIVATE uniflow::uniflow)

add_executable(uniflow_consumer main.cpp)
target_link_libraries(uniflow_consumer PRIVATE uniflow::uniflow)
