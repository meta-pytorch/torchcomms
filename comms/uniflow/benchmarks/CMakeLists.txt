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
  uniflow::uniflow
)

set(_UNIFLOW_BENCH_OBJECTS $<TARGET_OBJECTS:uniflow_bench_lib>)

if(UNIFLOW_ENABLE_RDMA)
  add_library(uniflow_rdma_bandwidth_bench OBJECT
    bench/RdmaBandwidthBenchmark.cpp
    bench/SendRecvBandwidthBenchmark.cpp
  )
  target_link_libraries(uniflow_rdma_bandwidth_bench PRIVATE
    uniflow_build_config
    uniflow::uniflow
  )
  list(APPEND _UNIFLOW_BENCH_OBJECTS
    $<TARGET_OBJECTS:uniflow_rdma_bandwidth_bench>)
endif()

if(UNIFLOW_ENABLE_TCP)
  add_library(uniflow_tcp_bandwidth_bench OBJECT
    bench/TcpBandwidthBenchmark.cpp)
  target_link_libraries(uniflow_tcp_bandwidth_bench PRIVATE
    uniflow_build_config
    uniflow::uniflow
  )
  list(APPEND _UNIFLOW_BENCH_OBJECTS
    $<TARGET_OBJECTS:uniflow_tcp_bandwidth_bench>)
endif()

if(UNIFLOW_ENABLE_NVLINK)
  add_library(uniflow_connection_setup_bench OBJECT
    bench/ConnectionSetupBenchmark.cpp)
  add_library(uniflow_nvlink_bandwidth_bench OBJECT
    bench/NVLinkBandwidthBenchmark.cpp)
  foreach(_UNIFLOW_NVLINK_BENCH_TARGET IN ITEMS
      uniflow_connection_setup_bench
      uniflow_nvlink_bandwidth_bench)
    target_link_libraries(${_UNIFLOW_NVLINK_BENCH_TARGET} PRIVATE
      uniflow_build_config
      uniflow::uniflow
    )
    list(APPEND _UNIFLOW_BENCH_OBJECTS
      $<TARGET_OBJECTS:${_UNIFLOW_NVLINK_BENCH_TARGET}>)
  endforeach()
  unset(_UNIFLOW_NVLINK_BENCH_TARGET)

  find_package(NCCL QUIET)
  if(NCCL_FOUND)
    add_library(uniflow_nccl_send_recv_bench OBJECT
      bench/NcclSendRecvBenchmark.cpp)
    target_link_libraries(uniflow_nccl_send_recv_bench PRIVATE
      uniflow_build_config
      uniflow::uniflow
      NCCL::NCCL
    )
    list(APPEND _UNIFLOW_BENCH_OBJECTS
      $<TARGET_OBJECTS:uniflow_nccl_send_recv_bench>)
  endif()
endif()

add_executable(uniflow_bench main.cpp ${_UNIFLOW_BENCH_OBJECTS})
if(TARGET uniflow_nccl_send_recv_bench)
  target_compile_definitions(uniflow_bench PRIVATE
    UNIFLOW_ENABLE_NCCL_BENCHMARK=1)
endif()
target_link_libraries(uniflow_bench PRIVATE
  uniflow_build_config
  uniflow::uniflow
)

unset(_UNIFLOW_BENCH_OBJECTS)
