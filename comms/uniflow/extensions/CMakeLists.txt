# Copyright (c) Meta Platforms, Inc. and affiliates.

if(UNIFLOW_GPU_PLATFORM STREQUAL HIP)
  target_sources(uniflow PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/NvmlApiStub.cpp")
  target_link_libraries(uniflow PRIVATE uniflow_build_config)
endif()
