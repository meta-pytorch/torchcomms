// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <vector>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/gpe/benchmarks/NoOpKernel.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/BenchUtils.h"
#include "comms/testinfra/CudaBenchBase.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

static void
GpeSubmit(uint32_t iters, int numStreams, folly::UserCounters& counters) {
  folly::BenchmarkSuspender braces;

  CUDACHECK_TEST(cudaSetDevice(0));

  auto commRAII = ctran::createDummyCtranComm();
  auto* comm = commRAII->ctranComm.get();

  CtranAlgoDeviceState* devState_d = nullptr;
  CUDACHECK_TEST(cudaMalloc(&devState_d, sizeof(CtranAlgoDeviceState)));

  auto gpe = std::make_unique<CtranGpe>(0, comm);

  std::vector<cudaStream_t> streams(numStreams);
  for (int j = 0; j < numStreams; ++j) {
    CUDACHECK_TEST(cudaStreamCreate(&streams[j]));
  }

  CudaBenchBase bench;

  // Warm up
  {
    auto config = KernelConfig(
        KernelConfig::KernelType::ALLGATHER, streams[0], "bench", 0);
    config.args.devState_d = devState_d;
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    gpe->submit(
        std::move(emptyOps),
        nullptr,
        config,
        reinterpret_cast<const void*>(NoOpKernel));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(streams[0]));

  bench.startTiming();
  braces.dismiss();

  for (uint32_t i = 0; i < iters; ++i) {
    cudaStream_t s = streams[i % numStreams];
    auto config =
        KernelConfig(KernelConfig::KernelType::ALLGATHER, s, "bench", i);
    config.args.devState_d = devState_d;
    std::vector<std::unique_ptr<struct OpElem>> emptyOps;
    gpe->submit(
        std::move(emptyOps),
        nullptr,
        config,
        reinterpret_cast<const void*>(NoOpKernel));
  }

  CUDACHECK_TEST(cudaStreamSynchronize(streams[(iters - 1) % numStreams]));
  bench.stopTiming();
  float gpuTimeMs = bench.measureTime();
  braces.rehire();

  counters["iters"] = folly::UserMetric(iters, folly::UserMetric::Type::METRIC);
  counters["numStreams"] =
      folly::UserMetric(numStreams, folly::UserMetric::Type::METRIC);
  counters["gpuTimeMs"] =
      folly::UserMetric(gpuTimeMs, folly::UserMetric::Type::METRIC);

  for (auto& s : streams) {
    CUDACHECK_TEST(cudaStreamDestroy(s));
  }
  gpe.reset();
  CUDACHECK_TEST(cudaFree(devState_d));
}

BENCHMARK_SINGLE_PARAM_COUNTERS(GpeSubmit, 1);
BENCHMARK_SINGLE_PARAM_COUNTERS(GpeSubmit, 2);
BENCHMARK_SINGLE_PARAM_COUNTERS(GpeSubmit, 4);
BENCHMARK_SINGLE_PARAM_COUNTERS(GpeSubmit, 8);

int main(int argc, char** argv) {
  CHECK_GE(bench_utils::getNumCudaDevices(), 1);

  setenv("NCCL_CTRAN_BACKENDS", "socket", 0);
  ncclCvarInit();
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();

  cudaSetDevice(0);
  cudaDeviceReset();

  return 0;
}
