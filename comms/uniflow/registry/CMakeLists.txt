# Copyright (c) Meta Platforms, Inc. and affiliates.

add_library(uniflow_registry OBJECT
  RuntimeVersion.cpp
  TransportProviderRegistry.cpp
  PlatformRegistry.cpp
)
target_link_libraries(uniflow_registry PRIVATE uniflow_build_config)
