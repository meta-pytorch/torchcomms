# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(spdlog REQUIRED)

add_library(uniflow_logging OBJECT Logger.cpp)
target_compile_features(uniflow_logging PUBLIC cxx_std_20)
target_include_directories(uniflow_logging PUBLIC ${ROOT})
target_link_libraries(uniflow_logging PRIVATE spdlog::spdlog)
