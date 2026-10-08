// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Bulk-times CollTrace construction (the COLLTRACE_INIT scope of communicator
// init) and prints cold-vs-warm stats. Single-rank, .cc-only: no nvcc is
// needed to build this target, device code comes from the prebuilt colltrace
// library.
//
// Usage:
//   buck2 run //comms/utils/colltrace/benchmarks:colltrace_init_bench -- ITERS
//   CUDA_VISIBLE_DEVICES=3 buck2 run ...   # pick a GPU
//
// Mirrors production McclComm::init ordering: the driver and primary context
// are warmed before iteration 0 (as prims does), so iter 0 isolates the
// colltrace cold cost (first calibration + ring) rather than driver init.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <vector>

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <fmt/core.h>
#include <folly/init/Init.h>

#include "comms/utils/checks.h"
#include "comms/utils/colltrace/CollTrace.h"
#include "comms/utils/colltrace/plugins/CommDumpPlugin.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace {

constexpr int kDefaultIters = 20;
constexpr int kDeviceId = 0;

struct IterTiming {
  int64_t ctorUs;
  int64_t totalUs;
};

std::shared_ptr<meta::comms::colltrace::CollTrace> makeCollTrace() {
  auto plugins =
      std::vector<std::unique_ptr<meta::comms::colltrace::ICollTracePlugin>>{};
  plugins.push_back(std::make_unique<meta::comms::colltrace::CommDumpPlugin>());
  return std::make_shared<meta::comms::colltrace::CollTrace>(
      meta::comms::colltrace::CollTraceConfig{},
      ::CommLogData{},
      []() -> meta::comms::CommsMaybeVoid {
        CUDA_CHECK_EXPECTED(cudaSetDevice(kDeviceId));
        return folly::unit;
      },
      std::move(plugins));
}

void printStats(const char* scope, const std::vector<IterTiming>& samples) {
  std::vector<int64_t> ctor, total;
  ctor.reserve(samples.size());
  total.reserve(samples.size());
  for (const auto& s : samples) {
    ctor.push_back(s.ctorUs);
    total.push_back(s.totalUs);
  }
  std::sort(ctor.begin(), ctor.end());
  std::sort(total.begin(), total.end());
  const auto n = samples.size();
  const auto mean = [](const std::vector<int64_t>& v) {
    return std::accumulate(v.begin(), v.end(), int64_t{0}) /
        static_cast<int64_t>(v.size());
  };
  fmt::print(
      "[colltrace_init] scope={} n={} ctor_min={}us ctor_max={}us "
      "ctor_mean={}us ctor_p50={}us total_min={}us total_max={}us "
      "total_mean={}us total_p50={}us\n",
      scope,
      n,
      ctor.at(0),
      ctor.at(n - 1),
      mean(ctor),
      ctor.at(n / 2),
      total.at(0),
      total.at(n - 1),
      mean(total),
      total.at(n / 2));
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  ncclCvarInit();
  NCCL_COLLTRACE_TRACE_CUDA_GRAPH = true;

  int iters = kDefaultIters;
  if (argc > 1) {
    iters = std::max(1, std::atoi(argv[1]));
  }

  if (cudaSetDevice(kDeviceId) != cudaSuccess) {
    std::fprintf(
        stderr,
        "colltrace_init_bench: no usable GPU (cudaSetDevice failed). "
        "Dev-mode asan builds need ASAN_OPTIONS=protect_shadow_gap=0.\n");
    return 1;
  }
  const bool graphSupported =
      meta::comms::colltrace::graphColltraceSupported("bench");
  fmt::print(
      "[colltrace_init] iters={} graph_supported={}\n",
      iters,
      graphSupported ? 1 : 0);

  // Pre-warm driver + primary context outside the timer (production: prims
  // runs first), so iter 0 measures colltrace-cold, not process-cold.
  void* warm = nullptr;
  if (cudaMalloc(&warm, 1024) != cudaSuccess) {
    std::fprintf(stderr, "colltrace_init_bench: cudaMalloc failed\n");
    return 1;
  }
  if (warm != nullptr && cudaFree(warm) != cudaSuccess) {
    std::fprintf(stderr, "colltrace_init_bench: cudaFree failed\n");
    return 1;
  }

  std::vector<IterTiming> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    auto trace = makeCollTrace();
    const auto t1 = std::chrono::steady_clock::now();
    // Destruction joins the warmup thread, so total covers full bring-up.
    trace.reset();
    const auto t2 = std::chrono::steady_clock::now();
    IterTiming sample{
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(),
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count()};
    samples.push_back(sample);
    fmt::print(
        "[colltrace_init] iter={} ctor={}us total={}us\n",
        i,
        sample.ctorUs,
        sample.totalUs);
  }

  printStats("cold", {samples.at(0)});
  if (samples.size() > 1) {
    printStats(
        "warm", std::vector<IterTiming>(samples.begin() + 1, samples.end()));
  }
  return 0;
}
