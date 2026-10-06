# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(GTest REQUIRED)

set(_UNIFLOW_TESTS_REQUIRE_GMOCK OFF)
if((UNIFLOW_ENABLE_CUDA AND UNIFLOW_GPU_PLATFORM STREQUAL CUDA) OR
   UNIFLOW_ENABLE_TCP OR
   UNIFLOW_ENABLE_RDMA OR
   (UNIFLOW_ENABLE_CUDA AND UNIFLOW_GPU_PLATFORM STREQUAL HIP) OR
   (UNIFLOW_ENABLE_NVLINK AND UNIFLOW_GPU_PLATFORM STREQUAL CUDA))
  set(_UNIFLOW_TESTS_REQUIRE_GMOCK ON)
endif()

if(_UNIFLOW_TESTS_REQUIRE_GMOCK AND NOT TARGET GTest::gmock)
  find_path(UNIFLOW_GMOCK_INCLUDE_DIR gmock/gmock.h)
  find_library(UNIFLOW_GMOCK_LIBRARY NAMES gmock)
  if(UNIFLOW_GMOCK_INCLUDE_DIR AND UNIFLOW_GMOCK_LIBRARY)
    add_library(GTest::gmock UNKNOWN IMPORTED)
    set_target_properties(GTest::gmock PROPERTIES
      IMPORTED_LOCATION "${UNIFLOW_GMOCK_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${UNIFLOW_GMOCK_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES GTest::gtest
    )
  endif()
endif()

if(_UNIFLOW_TESTS_REQUIRE_GMOCK AND NOT TARGET GTest::gmock)
  message(FATAL_ERROR
    "The enabled UniFlow hardware tests require GoogleMock, but no "
    "GTest::gmock target or standalone gmock library was found.")
endif()

function(uniflow_add_test TARGET_NAME SOURCE)
  add_executable(${TARGET_NAME} ${SOURCE})
  target_link_libraries(${TARGET_NAME} PRIVATE
    uniflow::uniflow
    uniflow_build_config
    GTest::gtest
    GTest::gtest_main
  )
  if(TARGET GTest::gmock)
    target_link_libraries(${TARGET_NAME} PRIVATE GTest::gmock)
  endif()
  add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
  set_tests_properties(${TARGET_NAME} PROPERTIES LABELS "${ARGN}")
endfunction()

function(uniflow_add_unit_test TARGET_NAME SOURCE)
  uniflow_add_test(${TARGET_NAME} ${SOURCE} unit)
endfunction()

function(uniflow_add_gpu_test TARGET_NAME SOURCE)
  uniflow_hipify(_UNIFLOW_TEST_SOURCES "${SOURCE}")
  uniflow_add_test(${TARGET_NAME} "${_UNIFLOW_TEST_SOURCES}" ${ARGN})
  uniflow_target_hipified(${TARGET_NAME})
endfunction()

if(UNIFLOW_ENABLE_CUDA AND UNIFLOW_GPU_PLATFORM STREQUAL HIP)
  uniflow_hipify(_UNIFLOW_TEST_HEADERS
    "${UNIFLOW_SOURCE_DIR}/drivers/cuda/mock/MockCudaApi.h"
    "${UNIFLOW_SOURCE_DIR}/drivers/cuda/mock/MockCudaDriverApi.h")
endif()

add_subdirectory(headers)
add_subdirectory(unit)
add_subdirectory(../controller/tests controller_tests)
add_subdirectory(../core/tests core_tests)
add_subdirectory(../executor/tests executor_tests)
add_subdirectory(../logging/tests logging_tests)
if(UNIFLOW_ENABLE_CUDA AND UNIFLOW_GPU_PLATFORM STREQUAL CUDA)
  add_subdirectory(../drivers/cuda/tests cuda_driver_tests)
  add_subdirectory(../drivers/nvml/tests nvml_driver_tests)
  add_subdirectory(../transport/tests/unit transport_unit_tests)
endif()
if(UNIFLOW_ENABLE_TCP)
  add_subdirectory(../transport/tcp/tests/unit tcp_unit_tests)
  add_subdirectory(../transport/tcp/tests/integration tcp_integration_tests)
endif()
if(UNIFLOW_ENABLE_RDMA)
  add_subdirectory(../drivers/ibverbs/tests ibverbs_driver_tests)
endif()
if(UNIFLOW_ENABLE_RDMA)
  add_subdirectory(../transport/rdma/tests rdma_tests)
endif()
if(UNIFLOW_ENABLE_CUDA AND UNIFLOW_GPU_PLATFORM STREQUAL HIP)
  add_subdirectory(../transport/p2p/tests p2p_tests)
endif()
if(UNIFLOW_ENABLE_NVLINK AND UNIFLOW_GPU_PLATFORM STREQUAL CUDA)
  add_subdirectory(../transport/nvlink/tests nvlink_tests)
endif()
if(UNIFLOW_ENABLE_RDMA AND UNIFLOW_ENABLE_NVLINK AND
   UNIFLOW_GPU_PLATFORM STREQUAL CUDA AND UNIFLOW_ENABLE_TCP)
  add_subdirectory(integration)
endif()
if(UNIFLOW_ENABLE_RDMA AND UNIFLOW_ENABLE_NVLINK AND
   UNIFLOW_GPU_PLATFORM STREQUAL CUDA)
  add_subdirectory(../transport/tests/integration topology_integration_tests)
endif()
