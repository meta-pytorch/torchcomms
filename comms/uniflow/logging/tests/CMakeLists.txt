# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(GTest REQUIRED)

add_executable(LoggerTest LoggerTest.cpp)
target_link_libraries(LoggerTest
    uniflow_build_config uniflow_logging ${UNIFLOW_SPDLOG_TARGET}
    GTest::gtest_main)
target_compile_features(LoggerTest PUBLIC cxx_std_20)
add_test(NAME LoggerTest COMMAND LoggerTest)

# Separate binary: this one shuts spdlog's global registry down, which no other
# test in the same process could survive logging after.
add_executable(LoggerShutdownTest LoggerShutdownTest.cpp)
target_link_libraries(LoggerShutdownTest
    uniflow_build_config uniflow_logging ${UNIFLOW_SPDLOG_TARGET}
    GTest::gtest_main)
target_compile_features(LoggerShutdownTest PUBLIC cxx_std_20)
add_test(NAME LoggerShutdownTest COMMAND LoggerShutdownTest)
