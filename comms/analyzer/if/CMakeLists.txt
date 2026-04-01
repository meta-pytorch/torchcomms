# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standalone build support (e.g., oss_build.sh / conda).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.10)
    project(CommsTracingService)
    find_package(wangle REQUIRED)
endif()

# When included as subdirectory, CMAKE_CURRENT_SOURCE_DIR is already
# comms/analyzer/if, so use just the filename.  Standalone builds set
# it up from the repo root.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(THRIFT_FILE "comms/analyzer/if/CommsTracingService.thrift")
else()
    set(THRIFT_FILE "CommsTracingService.thrift")
endif()
set(LIB_NAME "comms_tracing_service")

# When included as subdirectory from the root build, fbthrift's cmake modules
# are already on CMAKE_MODULE_PATH via add_subdirectory.  For standalone builds
# we need to add them explicitly.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_SOURCE_DIR}/../opensource/fbcode_builder/CMake
        ${CMAKE_CURRENT_SOURCE_DIR}/build/fbcode_builder/CMake
        ${CMAKE_MODULE_PATH})
endif()

# Include the FBThrift cmake module for helper functions.
# FBThriftConfig sets FBTHRIFT_COMPILER and FBTHRIFT_INCLUDE_DIR;
# for the subdirectory build these are already set by fbthrift.cmake.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include(FBThriftConfig)
endif()
include(FBThriftCppLibrary)

# Create the Thrift library
add_fbthrift_cpp_library(
    ${LIB_NAME}
    ${THRIFT_FILE}
    SERVICES
        CommsTracingService
    OPTIONS
        "json"
)
# fbthrift requires C++20; wangle may have set CMAKE_CXX_STANDARD to 17
# in its scope. Explicitly require C++20 for the generated thrift code.
set_target_properties(${LIB_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
