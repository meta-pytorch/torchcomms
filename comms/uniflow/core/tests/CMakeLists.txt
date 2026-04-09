# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(GTest REQUIRED)

add_executable(uniflow_func_test
    ${CMAKE_CURRENT_SOURCE_DIR}/FuncTest.cpp
)
target_compile_features(uniflow_func_test PRIVATE cxx_std_20)
target_include_directories(uniflow_func_test PRIVATE ${ROOT})
target_link_libraries(uniflow_func_test PRIVATE
    GTest::gtest
    GTest::gtest_main
)
add_test(NAME FuncTest COMMAND uniflow_func_test)

add_executable(uniflow_mpsc_queue_test
    ${CMAKE_CURRENT_SOURCE_DIR}/MpscQueueTest.cpp
)
target_compile_features(uniflow_mpsc_queue_test PRIVATE cxx_std_20)
target_include_directories(uniflow_mpsc_queue_test PRIVATE ${ROOT})
target_link_libraries(uniflow_mpsc_queue_test PRIVATE
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)
add_test(NAME MpscQueueTest COMMAND uniflow_mpsc_queue_test)
