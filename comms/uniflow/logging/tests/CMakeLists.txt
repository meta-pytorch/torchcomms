# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(GTest REQUIRED)

add_executable(LoggerTest LoggerTest.cpp)
target_link_libraries(LoggerTest
    uniflow_logging spdlog::spdlog GTest::gtest_main)
target_compile_features(LoggerTest PUBLIC cxx_std_20)
target_include_directories(LoggerTest PUBLIC ${ROOT})
add_test(NAME LoggerTest COMMAND LoggerTest)

# Separate binary: this one shuts spdlog's global registry down, which no other
# test in the same process could survive logging after.
add_executable(LoggerShutdownTest LoggerShutdownTest.cpp)
target_link_libraries(LoggerShutdownTest
    uniflow_logging spdlog::spdlog GTest::gtest_main)
target_compile_features(LoggerShutdownTest PUBLIC cxx_std_20)
target_include_directories(LoggerShutdownTest PUBLIC ${ROOT})
add_test(NAME LoggerShutdownTest COMMAND LoggerShutdownTest)
