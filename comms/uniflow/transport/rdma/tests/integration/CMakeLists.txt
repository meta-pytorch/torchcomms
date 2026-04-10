# Copyright (c) Meta Platforms, Inc. and affiliates.

add_executable(SingleHostTest SingleHostTest.cpp)
target_link_libraries(SingleHostTest
    uniflow_rdma_transport GTest::gtest_main)
target_compile_features(SingleHostTest PUBLIC cxx_std_20)
target_include_directories(SingleHostTest PUBLIC ${ROOT})
add_test(NAME SingleHostTest COMMAND SingleHostTest)
