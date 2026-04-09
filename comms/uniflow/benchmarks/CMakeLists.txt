# Copyright (c) Meta Platforms, Inc. and affiliates.

set(UNIFLOW_BENCH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/BenchmarkRunner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Bootstrap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Rendezvous.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Reporter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Stats.cpp
)

set(UNIFLOW_BENCH_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/BenchmarkResult.h
    ${CMAKE_CURRENT_SOURCE_DIR}/BenchmarkRunner.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Bootstrap.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Rendezvous.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Reporter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Stats.h
)

add_library(uniflow_bench_lib OBJECT
    ${UNIFLOW_BENCH_SOURCES}
    ${UNIFLOW_BENCH_HEADERS}
)

target_compile_features(uniflow_bench_lib PUBLIC cxx_std_20)

target_compile_options(uniflow_bench_lib PRIVATE -fPIC)

target_include_directories(uniflow_bench_lib PUBLIC
    ${ROOT}
)

target_link_libraries(uniflow_bench_lib PRIVATE
    uniflow_controller
    uniflow_logging
)

add_library(uniflow_connection_setup_bench OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/ConnectionSetupBenchmark.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/ConnectionSetupBenchmark.h
)

target_compile_features(uniflow_connection_setup_bench PUBLIC cxx_std_20)

target_compile_options(uniflow_connection_setup_bench PRIVATE -fPIC)

target_include_directories(uniflow_connection_setup_bench PUBLIC
    ${ROOT}
)

target_link_libraries(uniflow_connection_setup_bench PRIVATE
    uniflow_bench_lib
    uniflow_executor
    uniflow_nvlink_transport
    uniflow_logging
)

add_library(uniflow_rdma_bandwidth_bench OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/RdmaBandwidthBenchmark.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/RdmaBandwidthBenchmark.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SegmentHelper.h
)

target_compile_features(uniflow_rdma_bandwidth_bench PUBLIC cxx_std_20)

target_compile_options(uniflow_rdma_bandwidth_bench PRIVATE -fPIC)

target_include_directories(uniflow_rdma_bandwidth_bench PUBLIC
    ${ROOT}
)

target_link_libraries(uniflow_rdma_bandwidth_bench PRIVATE
    uniflow_bench_lib
    uniflow_executor
    uniflow_rdma_transport
    uniflow_logging
)

add_executable(uniflow_bench
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    $<TARGET_OBJECTS:uniflow_bench_lib>
    $<TARGET_OBJECTS:uniflow_connection_setup_bench>
    $<TARGET_OBJECTS:uniflow_rdma_bandwidth_bench>
)

target_compile_features(uniflow_bench PUBLIC cxx_std_20)

target_include_directories(uniflow_bench PUBLIC
    ${ROOT}
)

target_link_libraries(uniflow_bench PRIVATE
    uniflow_bench_lib
    uniflow_connection_setup_bench
    uniflow_rdma_bandwidth_bench
    uniflow_controller
    uniflow_logging
)
