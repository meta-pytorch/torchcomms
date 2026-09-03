// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "comms/utils/collstats/CollStatsDeviceBlock.h"
#include "comms/utils/collstats/CollStatsKeyRegistry.h"
#include "comms/utils/collstats/CollStatsReader.h"
#include "comms/utils/collstats/CollStatsSpan.cuh"
#include "comms/utils/collstats/CollStatsTypes.h"

// Sanity gate (TL review): the device %globaltimer span the collective records
// must agree with an independent host-side cudaEvent measurement of the same
// kernel. A gross disagreement (wrong units, bad tick->ns, wrong field) is
// caught here without needing a live multi-rank run.

namespace meta::comms::collstats {
namespace {

// Span entry -> busy-wait a known duration on the reference clock -> finalize,
// mirroring a real collective's kernel but with a controlled span length.
__global__ void timedSpanKernel(
    CollStatsDeviceBlock* block,
    uint32_t keyId,
    uint64_t logicalBytes,
    uint64_t busyNs) {
  collStatsSpanEntry(block, /*slot=*/0);
  // Gated on the same arch as the clock itself. Below sm90
  // collStatsGlobaltimerNs() is compiled to `return 0`, so the difference is
  // always 0 and the loop condition never goes false -- the kernel would spin
  // forever and hang the device rather than fail the test. The test skips such
  // devices too; this makes the kernel safe even if that check is ever lost.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const uint64_t t0 = collStatsGlobaltimerNs();
  while (collStatsGlobaltimerNs() - t0 < busyNs) {
    // spin
  }
#else
  (void)busyNs;
#endif
  __syncthreads();
  if (threadIdx.x == 0) {
    collStatsSpanFinalizeElectedById(block, /*slot=*/0, keyId, logicalBytes);
  }
}

// RAII timer: records the start event on construction and, on destruction,
// records the stop event, synchronizes, and writes the elapsed time (ns) to the
// caller's slot. Its scope brackets the timed region.
// Every CUDA call is checked into `*outStatus`, which holds the first failure.
// Unchecked, a failed event call leaves `ms` at 0, `eventNs` at 0, and the test
// reports a timing disagreement -- pointing at the span logic when the real
// fault was the reference measurement.
class ScopedCudaTimer {
 public:
  ScopedCudaTimer(cudaStream_t stream, double* outNs, cudaError_t* outStatus)
      : stream_(stream), outNs_(outNs), outStatus_(outStatus) {
    record(cudaEventCreate(&start_));
    record(cudaEventCreate(&stop_));
    record(cudaEventRecord(start_, stream_));
  }
  ~ScopedCudaTimer() {
    record(cudaEventRecord(stop_, stream_));
    record(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    record(cudaEventElapsedTime(&ms, start_, stop_));
    *outNs_ = static_cast<double>(ms) * 1e6;
    record(cudaEventDestroy(start_));
    record(cudaEventDestroy(stop_));
    *outStatus_ = status_;
  }
  ScopedCudaTimer(const ScopedCudaTimer&) = delete;
  ScopedCudaTimer& operator=(const ScopedCudaTimer&) = delete;
  ScopedCudaTimer(ScopedCudaTimer&&) = delete;
  ScopedCudaTimer& operator=(ScopedCudaTimer&&) = delete;

 private:
  // Keeps the first failure: a later call failing because an earlier one did is
  // the less informative of the two.
  void record(cudaError_t e) {
    if (status_ == cudaSuccess) {
      status_ = e;
    }
  }

  cudaStream_t stream_;
  double* outNs_;
  cudaError_t* outStatus_;
  cudaError_t status_{cudaSuccess};
  cudaEvent_t start_{nullptr};
  cudaEvent_t stop_{nullptr};
};

uint64_t maxDurNs(const CollStatSnapshot& snap) {
  uint64_t m = 0;
  for (const auto& v : snap.values) {
    m = v.durMaxNs > m ? v.durMaxNs : m;
  }
  return m;
}

TEST(CollStatsCrossCheckGpuTest, GlobaltimerSpanAgreesWithCudaEvent) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "no CUDA device";
  }
  // The span clock (%globaltimer) only exists from sm90; below it the kernel's
  // busy-wait has no clock to wait on and the span is never recorded, so there
  // is nothing to cross-check.
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  int major = 0;
  ASSERT_EQ(
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device),
      cudaSuccess);
  if (major < 9) {
    GTEST_SKIP() << "collective span requires sm90+, device is sm" << major
                 << "x";
  }

  const uint32_t capacity = 64;
  CollStatsDeviceBlockHandle h =
      collStatsAllocDeviceBlock(capacity, /*numSlots=*/4);
  ASSERT_NE(h.dev, nullptr);

  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  CollStatsKeyRegistry keys(capacity);
  const uint32_t id = keys.resolve(
      CollStatKey{
          CollStatOp::AllReduce,
          CollStatAlgo::Direct,
          CollStatProto::Unknown,
          7u,
          3u});
  const uint64_t busyNs =
      1'000'000; // 1 ms — well above launch/scheduling noise

  double eventNs = 0.0;
  cudaError_t timerStatus = cudaSuccess;
  cudaError_t launchStatus = cudaSuccess;
  collStatsEnqueuePreReset(h, /*slot=*/0, stream);
  {
    ScopedCudaTimer timer(stream, &eventNs, &timerStatus);
    timedSpanKernel<<<8, 64, 0, stream>>>(h.dev, id, 4096, busyNs);
    // Checked here rather than left to surface as a zero span: a launch that
    // never ran would otherwise be reported as a timing disagreement.
    launchStatus = cudaGetLastError();
  } // timer dtor records stop, syncs, fills eventNs
  ASSERT_EQ(launchStatus, cudaSuccess) << "timedSpanKernel launch failed";
  ASSERT_EQ(timerStatus, cudaSuccess) << "cudaEvent reference timing failed";

  const CollStatSnapshot snap = collStatsReadWindow(h, stream, keys);
  const uint64_t spanNs = maxDurNs(snap);

  // Both measurements must see roughly the busy-wait duration...
  EXPECT_GE(spanNs, busyNs * 8 / 10); // span >= ~0.8x the busy-wait
  EXPECT_GE(eventNs, static_cast<double>(busyNs) * 0.8);
  // ...and, crucially, agree with each other within 25% (catches unit/scale
  // bugs in the on-device tick handling). The cudaEvent brackets the kernel so
  // it is the wider of the two (adds launch/scheduling overhead).
  // Asserted, not expected: with both measurements at zero the ratio below is
  // 0.0/0.0, and the test would fail reporting `nan` instead of the fact that
  // neither clock produced anything.
  const double denom = std::max(static_cast<double>(spanNs), eventNs);
  ASSERT_GT(denom, 0.0) << "neither the span nor the cudaEvent measured "
                           "anything; spanNs="
                        << spanNs << " eventNs=" << eventNs;
  const double rel = std::abs(static_cast<double>(spanNs) - eventNs) / denom;
  EXPECT_LT(rel, 0.25) << "spanNs=" << spanNs << " eventNs=" << eventNs;

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  cudaStreamDestroy(stream);
  collStatsFreeDeviceBlock(h);
}

} // namespace
} // namespace meta::comms::collstats
