// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <cstddef>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <glog/logging.h>

#include "comms/prims/benchmarks/CopyOpReduceBench.cuh"
#include "comms/testinfra/BenchUtils.h"

namespace comms::prims::benchmark {
namespace {

void runPolicy(
    unsigned int iters,
    CopyOpReducePolicy policy,
    std::size_t nbytes,
    folly::UserCounters& counters) {
  folly::BenchmarkSuspender suspender;
  suspender.dismiss();
  const auto timing =
      runCopyOpReduceBenchmark(policy, nbytes, static_cast<int>(iters));
  suspender.rehire();
  counters["deviceTimeUs"] =
      folly::UserMetric(timing.timeUs, folly::UserMetric::Type::METRIC);
  counters["payloadGBps"] =
      folly::UserMetric(timing.payloadGBps, folly::UserMetric::Type::METRIC);
  counters["memoryGBps"] =
      folly::UserMetric(timing.memoryGBps, folly::UserMetric::Type::METRIC);
}

void tileReduceStaged(
    unsigned int iters,
    std::size_t nbytes,
    folly::UserCounters& counters) {
  runPolicy(iters, CopyOpReducePolicy::TileReduceStaged, nbytes, counters);
}

void cpAsyncSmemReduce(
    unsigned int iters,
    std::size_t nbytes,
    folly::UserCounters& counters) {
  runPolicy(iters, CopyOpReducePolicy::CpAsyncSmemReduce, nbytes, counters);
}

#define REGISTER_SIZES(function)                     \
  BENCHMARK_SINGLE_PARAM_COUNTERS(function, 32768);  \
  BENCHMARK_SINGLE_PARAM_COUNTERS(function, 131072); \
  BENCHMARK_SINGLE_PARAM_COUNTERS(function, 524288); \
  BENCHMARK_SINGLE_PARAM_COUNTERS(function, 2097152)

REGISTER_SIZES(tileReduceStaged);
REGISTER_SIZES(cpAsyncSmemReduce);

} // namespace
} // namespace comms::prims::benchmark

int main(int argc, char** argv) {
  CHECK_GE(bench_utils::getNumCudaDevices(), 1);
  CHECK_EQ(cudaSetDevice(0), cudaSuccess);
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();
  CHECK_EQ(cudaDeviceReset(), cudaSuccess);
  return 0;
}
