# Copyright (c) Meta Platforms, Inc. and affiliates.

add_executable(RdmaTransportTest RdmaTransportTest.cpp)
target_link_libraries(RdmaTransportTest
    uniflow_rdma_transport GTest::gtest_main)
target_compile_features(RdmaTransportTest PUBLIC cxx_std_20)
target_include_directories(RdmaTransportTest PUBLIC ${ROOT})
add_test(NAME RdmaTransportTest COMMAND RdmaTransportTest)
