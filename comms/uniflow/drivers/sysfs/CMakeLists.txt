# Copyright (c) Meta Platforms, Inc. and affiliates.

add_library(uniflow_sysfs OBJECT SysfsApi.cpp)
target_compile_features(uniflow_sysfs PUBLIC cxx_std_20)
target_link_libraries(uniflow_sysfs PRIVATE uniflow_build_config)
