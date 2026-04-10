# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(GTest REQUIRED)

add_executable(uniflow_result_test
    ${CMAKE_CURRENT_SOURCE_DIR}/ResultTest.cpp
)
target_compile_features(uniflow_result_test PRIVATE cxx_std_20)
target_include_directories(uniflow_result_test PRIVATE ${ROOT})
target_link_libraries(uniflow_result_test PRIVATE
    GTest::gtest
    GTest::gtest_main
)
add_test(NAME ResultTest COMMAND uniflow_result_test)

add_executable(uniflow_segment_test
    ${CMAKE_CURRENT_SOURCE_DIR}/SegmentTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Segment.cpp
)
target_compile_features(uniflow_segment_test PRIVATE cxx_std_20)
target_include_directories(uniflow_segment_test PRIVATE ${ROOT})
target_link_libraries(uniflow_segment_test PRIVATE
    GTest::gtest
    GTest::gtest_main
)
add_test(NAME SegmentTest COMMAND uniflow_segment_test)
