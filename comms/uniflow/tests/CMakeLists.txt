# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(GTest REQUIRED)

function(uniflow_add_unit_test TARGET_NAME SOURCE)
  add_executable(${TARGET_NAME} ${SOURCE})
  target_link_libraries(${TARGET_NAME} PRIVATE
    uniflow::uniflow
    GTest::gtest
    GTest::gtest_main
  )
  if(TARGET GTest::gmock)
    target_link_libraries(${TARGET_NAME} PRIVATE GTest::gmock)
  endif()
  add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
  set_tests_properties(${TARGET_NAME} PROPERTIES LABELS unit)
endfunction()

add_subdirectory(headers)
add_subdirectory(unit)
add_subdirectory(../controller/tests controller_tests)
add_subdirectory(../core/tests core_tests)
add_subdirectory(../executor/tests executor_tests)
add_subdirectory(../logging/tests logging_tests)
if(UNIFLOW_ENABLE_CUDA)
  add_subdirectory(../drivers/cuda/tests cuda_driver_tests)
  add_subdirectory(../transport/tests/unit transport_unit_tests)
endif()
if(UNIFLOW_ENABLE_TCP)
  add_subdirectory(../transport/tcp/tests/unit tcp_unit_tests)
  add_subdirectory(../transport/tcp/tests/integration tcp_integration_tests)
endif()
