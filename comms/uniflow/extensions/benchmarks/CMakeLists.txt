# Copyright (c) Meta Platforms, Inc. and affiliates.

if(UNIFLOW_ENABLE_TCP)
  uniflow_hipify(_UNIFLOW_TCP_BANDWIDTH_BENCH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/TcpBandwidthBenchmark.cpp)
  add_library(uniflow_tcp_bandwidth_bench OBJECT
    ${_UNIFLOW_TCP_BANDWIDTH_BENCH_SOURCES})
  target_link_libraries(uniflow_tcp_bandwidth_bench PRIVATE
    uniflow_build_config uniflow)
  uniflow_target_hipified(uniflow_tcp_bandwidth_bench)
  list(APPEND _UNIFLOW_BENCH_OBJECTS
    $<TARGET_OBJECTS:uniflow_tcp_bandwidth_bench>)
endif()

if(UNIFLOW_GPU_PLATFORM STREQUAL HIP)
  uniflow_hipify(_UNIFLOW_XGMI_BANDWIDTH_BENCH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/XgmiBandwidthBenchmark.cpp)
  add_library(uniflow_xgmi_bandwidth_bench OBJECT
    ${_UNIFLOW_XGMI_BANDWIDTH_BENCH_SOURCES})
  target_link_libraries(uniflow_xgmi_bandwidth_bench PRIVATE
    uniflow_build_config uniflow)
  uniflow_target_hipified(uniflow_xgmi_bandwidth_bench)
  list(APPEND _UNIFLOW_BENCH_OBJECTS
    $<TARGET_OBJECTS:uniflow_xgmi_bandwidth_bench>)
endif()
set(_UNIFLOW_BENCH_OBJECTS "${_UNIFLOW_BENCH_OBJECTS}" PARENT_SCOPE)
