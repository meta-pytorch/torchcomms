# Copyright (c) Meta Platforms, Inc. and affiliates.

add_library(uniflow_bench_lib OBJECT
  BenchmarkRunner.cpp
  Bootstrap.cpp
  Rendezvous.cpp
  Reporter.cpp
  Stats.cpp
)
target_link_libraries(uniflow_bench_lib PRIVATE
  uniflow_build_config
  uniflow
)

set(_UNIFLOW_BENCH_OBJECTS $<TARGET_OBJECTS:uniflow_bench_lib>)

if(UNIFLOW_ENABLE_RDMA)
  uniflow_hipify(_UNIFLOW_RDMA_BANDWIDTH_BENCH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/RdmaBandwidthBenchmark.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/SendRecvBandwidthBenchmark.cpp
  )
  add_library(uniflow_rdma_bandwidth_bench OBJECT
    ${_UNIFLOW_RDMA_BANDWIDTH_BENCH_SOURCES}
  )
  target_link_libraries(uniflow_rdma_bandwidth_bench PRIVATE
    uniflow_build_config
    uniflow
  )
  uniflow_target_hipified(uniflow_rdma_bandwidth_bench)
  list(APPEND _UNIFLOW_BENCH_OBJECTS
    $<TARGET_OBJECTS:uniflow_rdma_bandwidth_bench>)
endif()

if(UNIFLOW_ENABLE_TCP)
  uniflow_hipify(_UNIFLOW_TCP_BANDWIDTH_BENCH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/TcpBandwidthBenchmark.cpp)
  add_library(uniflow_tcp_bandwidth_bench OBJECT
    ${_UNIFLOW_TCP_BANDWIDTH_BENCH_SOURCES})
  target_link_libraries(uniflow_tcp_bandwidth_bench PRIVATE
    uniflow_build_config
    uniflow
  )
  uniflow_target_hipified(uniflow_tcp_bandwidth_bench)
  list(APPEND _UNIFLOW_BENCH_OBJECTS
    $<TARGET_OBJECTS:uniflow_tcp_bandwidth_bench>)
endif()

if(UNIFLOW_ENABLE_NVLINK AND UNIFLOW_GPU_PLATFORM STREQUAL CUDA)
  add_library(uniflow_connection_setup_bench OBJECT
    bench/ConnectionSetupBenchmark.cpp)
  add_library(uniflow_nvlink_bandwidth_bench OBJECT
    bench/NVLinkBandwidthBenchmark.cpp)
  foreach(_UNIFLOW_NVLINK_BENCH_TARGET IN ITEMS
      uniflow_connection_setup_bench
      uniflow_nvlink_bandwidth_bench)
    target_link_libraries(${_UNIFLOW_NVLINK_BENCH_TARGET} PRIVATE
      uniflow_build_config
      uniflow
    )
    list(APPEND _UNIFLOW_BENCH_OBJECTS
      $<TARGET_OBJECTS:${_UNIFLOW_NVLINK_BENCH_TARGET}>)
  endforeach()
  unset(_UNIFLOW_NVLINK_BENCH_TARGET)
endif()

if(UNIFLOW_ENABLE_CUDA AND UNIFLOW_GPU_PLATFORM STREQUAL HIP)
  uniflow_hipify(_UNIFLOW_XGMI_BANDWIDTH_BENCH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/XgmiBandwidthBenchmark.cpp)
  add_library(uniflow_xgmi_bandwidth_bench OBJECT
    ${_UNIFLOW_XGMI_BANDWIDTH_BENCH_SOURCES})
  target_link_libraries(uniflow_xgmi_bandwidth_bench PRIVATE
    uniflow_build_config
    uniflow
  )
  uniflow_target_hipified(uniflow_xgmi_bandwidth_bench)
  list(APPEND _UNIFLOW_BENCH_OBJECTS
    $<TARGET_OBJECTS:uniflow_xgmi_bandwidth_bench>)
endif()

if(UNIFLOW_ENABLE_CUDA AND UNIFLOW_GPU_PLATFORM STREQUAL CUDA)
  find_package(NCCL QUIET)
  if(NCCL_FOUND)
    add_library(uniflow_nccl_send_recv_bench OBJECT
      bench/NcclSendRecvBenchmark.cpp)
    target_link_libraries(uniflow_nccl_send_recv_bench PRIVATE
      uniflow_build_config
      uniflow
      NCCL::NCCL
    )
    list(APPEND _UNIFLOW_BENCH_OBJECTS
      $<TARGET_OBJECTS:uniflow_nccl_send_recv_bench>)
  else()
    message(STATUS "NCCL not found; skipping the NCCL benchmark")
  endif()
endif()

add_executable(uniflow_bench main.cpp ${_UNIFLOW_BENCH_OBJECTS})
if(TARGET uniflow_xgmi_bandwidth_bench)
  target_compile_definitions(uniflow_bench PRIVATE
    UNIFLOW_ENABLE_XGMI_BENCHMARK=1)
endif()
if(TARGET uniflow_nccl_send_recv_bench)
  target_compile_definitions(uniflow_bench PRIVATE
    UNIFLOW_ENABLE_NCCL_BENCHMARK=1)
  target_link_libraries(uniflow_bench PRIVATE NCCL::NCCL)
endif()
target_link_libraries(uniflow_bench PRIVATE
  uniflow_build_config
  uniflow
)

unset(_UNIFLOW_BENCH_OBJECTS)
