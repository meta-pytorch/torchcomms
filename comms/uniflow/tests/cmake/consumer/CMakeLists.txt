# Copyright (c) Meta Platforms, Inc. and affiliates.

cmake_minimum_required(VERSION 3.20)
project(uniflow_consumer LANGUAGES CXX)

find_package(uniflow CONFIG REQUIRED)

add_executable(uniflow_consumer main.cpp)
target_link_libraries(uniflow_consumer PRIVATE uniflow::uniflow)
