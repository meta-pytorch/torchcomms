# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(GTest REQUIRED)

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

add_subdirectory(headers)
add_subdirectory(unit)
add_subdirectory(../controller/tests controller_tests)
add_subdirectory(../core/tests core_tests)
add_subdirectory(../executor/tests executor_tests)
add_subdirectory(../logging/tests logging_tests)
if(NOT UNIFLOW_GPU_PLATFORM STREQUAL NONE)
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
  add_subdirectory(../transport/rdma/tests rdma_tests)
endif()
if(UNIFLOW_ENABLE_NVLINK)
  add_subdirectory(../transport/nvlink/tests nvlink_tests)
endif()
if(UNIFLOW_ENABLE_RDMA AND UNIFLOW_ENABLE_NVLINK)
  add_subdirectory(integration)
  add_subdirectory(../transport/tests/integration topology_integration_tests)
endif()
