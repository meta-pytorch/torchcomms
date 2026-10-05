# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(GTest REQUIRED)

add_executable(uniflow_event_base_test
    ${CMAKE_CURRENT_SOURCE_DIR}/EventBaseTest.cpp
)
target_compile_features(uniflow_event_base_test PRIVATE cxx_std_20)
target_link_libraries(uniflow_event_base_test PRIVATE
    uniflow_build_config
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    ${UNIFLOW_SPDLOG_TARGET}
    uniflow_logging
)
add_test(NAME EventBaseTest COMMAND uniflow_event_base_test)

add_executable(uniflow_scoped_event_base_thread_test
    ${CMAKE_CURRENT_SOURCE_DIR}/ScopedEventBaseThreadTest.cpp
)
target_compile_features(uniflow_scoped_event_base_thread_test PRIVATE cxx_std_20)
target_link_libraries(uniflow_scoped_event_base_thread_test PRIVATE
    uniflow_build_config
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    ${UNIFLOW_SPDLOG_TARGET}
    uniflow_logging
)
add_test(NAME ScopedEventBaseThreadTest COMMAND uniflow_scoped_event_base_thread_test)

add_executable(uniflow_stress_flood_test
    ${CMAKE_CURRENT_SOURCE_DIR}/StressFloodTest.cpp
)
target_compile_features(uniflow_stress_flood_test PRIVATE cxx_std_20)
target_link_libraries(uniflow_stress_flood_test PRIVATE
    uniflow_build_config
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    ${UNIFLOW_SPDLOG_TARGET}
    uniflow_logging
)
add_test(NAME StressFloodTest COMMAND uniflow_stress_flood_test)
