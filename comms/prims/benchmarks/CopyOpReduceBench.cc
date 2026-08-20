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

/*
 * Cache-resident point: 4 MiB per operand, well inside B300's ~126 MB L2 and a
 * whole multiple of every tile size in the sweep. A truly L1-resident run is
 * not achievable at production tile widths -- one tile alone is 61 KiB per
 * operand at vpt=6 and 123 KiB at vpt=12, so three operands already exceed the
 * 256 KB L1 before any pipelining. The in-kernel repeat count keeps per-launch
 * overhead out of the measurement.
 */
constexpr std::size_t kL2Bytes = 4194304;
constexpr int kL2Repeats = 16;

// HBM-resident point: far outside the ~126 MB L2.
constexpr std::size_t kHbmBytes = 268435456;
constexpr int kHbmRepeats = 1;

void runPoint(
    unsigned int iters,
    CopyOpReduceShape shape,
    int threads,
    int vpt,
    std::size_t nbytes,
    int repeats,
    folly::UserCounters& counters) {
  folly::BenchmarkSuspender suspender;
  suspender.dismiss();
  const auto timing = runCopyOpReduceBenchmark(
      shape, nbytes, static_cast<int>(iters), threads, vpt, repeats);
  suspender.rehire();
  counters["deviceTimeUs"] =
      folly::UserMetric(timing.timeUs, folly::UserMetric::Type::METRIC);
  counters["memoryGBps"] =
      folly::UserMetric(timing.memoryGBps, folly::UserMetric::Type::METRIC);
  counters["payloadGBps"] =
      folly::UserMetric(timing.payloadGBps, folly::UserMetric::Type::METRIC);
  // Measured bytes/clock -- no assumed boost clock anywhere in this number.
  counters["bytesPerClk"] =
      folly::UserMetric(timing.bytesPerClock, folly::UserMetric::Type::METRIC);
}

/*
 * Names encode shape / threads / vpt / residency:
 *   unf|fus|rdo|pip  _  t<threads>  _  v<vpt>  _  l1|hbm
 * Keep the (threads, vpt) pairs in sync with COPY_OP_REDUCE_CONFIGS in
 * CopyOpReduceBench.cu -- an unlisted pair throws at launch.
 */
#define DEFINE_POINT(TAG, SHAPE, THREADS, VPT)                          \
  void TAG##_t##THREADS##_v##VPT##_l2(                                  \
      unsigned int iters, std::size_t, folly::UserCounters& counters) { \
    runPoint(                                                           \
        iters,                                                          \
        CopyOpReduceShape::SHAPE,                                       \
        THREADS,                                                        \
        VPT,                                                            \
        kL2Bytes,                                                       \
        kL2Repeats,                                                     \
        counters);                                                      \
  }                                                                     \
  void TAG##_t##THREADS##_v##VPT##_hbm(                                 \
      unsigned int iters, std::size_t, folly::UserCounters& counters) { \
    runPoint(                                                           \
        iters,                                                          \
        CopyOpReduceShape::SHAPE,                                       \
        THREADS,                                                        \
        VPT,                                                            \
        kHbmBytes,                                                      \
        kHbmRepeats,                                                    \
        counters);                                                      \
  }

#define DEFINE_ALL_SHAPES(THREADS, VPT)      \
  DEFINE_POINT(unf, Unfused, THREADS, VPT)   \
  DEFINE_POINT(fus, Fused, THREADS, VPT)     \
  DEFINE_POINT(rdo, ReadOnly, THREADS, VPT)  \
  DEFINE_POINT(wro, WriteOnly, THREADS, VPT) \
  DEFINE_POINT(cpy, Copy, THREADS, VPT)      \
  DEFINE_POINT(pip, Pipelined, THREADS, VPT) \
  DEFINE_POINT(fwd, Forward, THREADS, VPT)

DEFINE_ALL_SHAPES(640, 6)
DEFINE_ALL_SHAPES(640, 12)
DEFINE_ALL_SHAPES(1024, 6)

#define REGISTER_POINT(TAG, THREADS, VPT)                             \
  BENCHMARK_SINGLE_PARAM_COUNTERS(TAG##_t##THREADS##_v##VPT##_l2, 0); \
  BENCHMARK_SINGLE_PARAM_COUNTERS(TAG##_t##THREADS##_v##VPT##_hbm, 0)

#define REGISTER_ALL_SHAPES(THREADS, VPT) \
  REGISTER_POINT(unf, THREADS, VPT);      \
  REGISTER_POINT(fus, THREADS, VPT);      \
  REGISTER_POINT(rdo, THREADS, VPT);      \
  REGISTER_POINT(wro, THREADS, VPT);      \
  REGISTER_POINT(cpy, THREADS, VPT);      \
  REGISTER_POINT(pip, THREADS, VPT);      \
  REGISTER_POINT(fwd, THREADS, VPT)

REGISTER_ALL_SHAPES(640, 6);
REGISTER_ALL_SHAPES(640, 12);
REGISTER_ALL_SHAPES(1024, 6);

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
