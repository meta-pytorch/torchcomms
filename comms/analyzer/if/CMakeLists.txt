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

# This CMakelists.txt file is used for oss_build.sh specifically and
# should be run under conda environment.
cmake_minimum_required(VERSION 3.10)

project(CommsTracingService)

find_package(wangle REQUIRED)

set(THRIFT_FILE "comms/analyzer/if/CommsTracingService.thrift")
set(LIB_NAME "comms_tracing_service")

set(CMAKE_MODULE_PATH
  # For in-fbsource builds:
  ${CMAKE_CURRENT_SOURCE_DIR}/../opensource/fbcode_builder/CMake
  # For shipit-transformed builds:
  ${CMAKE_CURRENT_SOURCE_DIR}/build/fbcode_builder/CMake
  ${CMAKE_MODULE_PATH})

# Include the FBThrift cmake module for helper functions
include(FBThriftConfig)
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

# Install library
install(
  TARGETS
    comms_tracing_service
  PUBLIC_HEADER DESTINATION
    "$<TARGET_PROPERTY:${LIB_NAME},HEADER_INSTALL_DIR>"
)
