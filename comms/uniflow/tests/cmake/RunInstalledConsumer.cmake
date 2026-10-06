# Copyright (c) Meta Platforms, Inc. and affiliates.

foreach(_uniflow_required_variable IN ITEMS
    UNIFLOW_BINARY_DIR
    UNIFLOW_CONSUMER_SOURCE_DIR
    UNIFLOW_CONSUMER_GENERATOR)
  if(NOT DEFINED ${_uniflow_required_variable})
    message(FATAL_ERROR "${_uniflow_required_variable} is required")
  endif()
endforeach()

set(_uniflow_test_root "${UNIFLOW_BINARY_DIR}/installed-consumer-test")
set(_uniflow_install_prefix "${_uniflow_test_root}/install")
set(_uniflow_consumer_build "${_uniflow_test_root}/build")
file(REMOVE_RECURSE "${_uniflow_test_root}")

set(_uniflow_install_command
  "${CMAKE_COMMAND}" --install "${UNIFLOW_BINARY_DIR}"
  --prefix "${_uniflow_install_prefix}")
if(UNIFLOW_CONSUMER_CONFIG)
  list(APPEND _uniflow_install_command
    --config "${UNIFLOW_CONSUMER_CONFIG}")
endif()
execute_process(
  COMMAND ${_uniflow_install_command}
  COMMAND_ERROR_IS_FATAL ANY)

set(_uniflow_configure_command
  "${CMAKE_COMMAND}"
  -S "${UNIFLOW_CONSUMER_SOURCE_DIR}"
  -B "${_uniflow_consumer_build}"
  -G "${UNIFLOW_CONSUMER_GENERATOR}"
  "-DCMAKE_PREFIX_PATH=${_uniflow_install_prefix}")
set(_uniflow_consumer_input_variables
  UNIFLOW_CONSUMER_TOOLCHAIN_FILE
  UNIFLOW_CONSUMER_CXX_COMPILER
  UNIFLOW_CONSUMER_CXX_COMPILER_TARGET
  UNIFLOW_CONSUMER_CXX_COMPILER_EXTERNAL_TOOLCHAIN
  UNIFLOW_CONSUMER_SYSROOT
  UNIFLOW_CONSUMER_SYSROOT_COMPILE
  UNIFLOW_CONSUMER_SYSROOT_LINK)
set(_uniflow_consumer_cmake_variables
  CMAKE_TOOLCHAIN_FILE
  CMAKE_CXX_COMPILER
  CMAKE_CXX_COMPILER_TARGET
  CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN
  CMAKE_SYSROOT
  CMAKE_SYSROOT_COMPILE
  CMAKE_SYSROOT_LINK)
if(UNIFLOW_CONSUMER_CROSSCOMPILING)
  list(APPEND _uniflow_consumer_input_variables
    UNIFLOW_CONSUMER_SYSTEM_NAME
    UNIFLOW_CONSUMER_SYSTEM_PROCESSOR)
  list(APPEND _uniflow_consumer_cmake_variables
    CMAKE_SYSTEM_NAME
    CMAKE_SYSTEM_PROCESSOR)
endif()
foreach(_uniflow_consumer_input_variable _uniflow_consumer_cmake_variable
    IN ZIP_LISTS
      _uniflow_consumer_input_variables
      _uniflow_consumer_cmake_variables)
  if(DEFINED ${_uniflow_consumer_input_variable} AND
     NOT "${${_uniflow_consumer_input_variable}}" STREQUAL "")
    list(APPEND _uniflow_configure_command
      "-D${_uniflow_consumer_cmake_variable}=${${_uniflow_consumer_input_variable}}")
  endif()
endforeach()
if(UNIFLOW_CONSUMER_MAKE_PROGRAM)
  list(APPEND _uniflow_configure_command
    "-DCMAKE_MAKE_PROGRAM=${UNIFLOW_CONSUMER_MAKE_PROGRAM}")
endif()
if(UNIFLOW_CONSUMER_BUILD_TYPE)
  list(APPEND _uniflow_configure_command
    "-DCMAKE_BUILD_TYPE=${UNIFLOW_CONSUMER_BUILD_TYPE}")
endif()
if(UNIFLOW_CONSUMER_CUDA_ROOT)
  list(APPEND _uniflow_configure_command
    "-DCUDAToolkit_ROOT=${UNIFLOW_CONSUMER_CUDA_ROOT}")
endif()
if(UNIFLOW_CONSUMER_FMT_DIR)
  list(APPEND _uniflow_configure_command
    "-Dfmt_DIR=${UNIFLOW_CONSUMER_FMT_DIR}")
endif()
if(UNIFLOW_CONSUMER_HIP_DIR)
  list(APPEND _uniflow_configure_command
    "-Dhip_DIR=${UNIFLOW_CONSUMER_HIP_DIR}")
endif()
if(UNIFLOW_CONSUMER_SPDLOG_DIR)
  list(APPEND _uniflow_configure_command
    "-Dspdlog_DIR=${UNIFLOW_CONSUMER_SPDLOG_DIR}")
endif()
execute_process(
  COMMAND ${_uniflow_configure_command}
  COMMAND_ERROR_IS_FATAL ANY)

set(_uniflow_build_command
  "${CMAKE_COMMAND}" --build "${_uniflow_consumer_build}")
if(UNIFLOW_CONSUMER_CONFIG)
  list(APPEND _uniflow_build_command
    --config "${UNIFLOW_CONSUMER_CONFIG}")
endif()
execute_process(
  COMMAND ${_uniflow_build_command}
  COMMAND_ERROR_IS_FATAL ANY)
