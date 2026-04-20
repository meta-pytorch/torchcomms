// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "comms/ctran/tests/cudagraph/CtranCudaGraphTestBase.h"
#include "comms/ctran/tests/cudagraph/CudaGraphTestBuilder.h"
#include "comms/ctran/tests/cudagraph/DeviceVerify.h"

enum class GraphPattern {
  Basic,
};

inline const char* patternToString(GraphPattern pattern) {
  switch (pattern) {
    case GraphPattern::Basic:
      return "Basic";
  }
  return "Unknown";
}

// Base replay count per pattern. Multiplied by the replay multiplier parameter.
inline int baseReplays(GraphPattern pattern) {
  switch (pattern) {
    case GraphPattern::Basic:
      return 5;
  }
  return 3;
}

struct AlgoDescriptor {
  struct Buffers {
    virtual ~Buffers() = default;
    virtual void* sendbuf() = 0;
    virtual void* recvbuf() = 0;
    virtual size_t recvBytes() = 0;
  };

  std::string name;

  std::function<bool(CtranComm*, size_t count)> expectsHostNodes =
      [](CtranComm*, size_t) { return true; };

  std::function<bool(CtranComm*, size_t count, int numRanks)> isSupported;

  std::function<std::shared_ptr<Buffers>(size_t count, int rank, int nRanks)>
      makeBuffers;

  std::function<void(Buffers*, size_t count, ctran::testing::CaptureContext&)>
      capture;
};

// Run the collective eagerly to compute expected output.
// Uses the same comm as graph capture — the eager run warms up connections
// which is needed for some algorithms on certain topologies.
inline void computeExpected(
    AlgoDescriptor& desc,
    AlgoDescriptor::Buffers* bufs,
    size_t count,
    CtranComm* comm,
    int rank,
    int nRanks) {
  meta::comms::CudaStream stream(cudaStreamNonBlocking);
  ctran::testing::CaptureContext ctx{comm, stream.get(), rank, nRanks};
  desc.capture(bufs, count, ctx);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  cudaDeviceSynchronize();
}

// Launch device-side buffer comparison kernel. Async on stream, no host sync.
inline void deviceVerifyAgainstExpected(
    AlgoDescriptor::Buffers* actual,
    AlgoDescriptor::Buffers* expected,
    unsigned int* mismatchCount_d,
    cudaStream_t stream) {
  ctran::testing::launchCompareBuffers(
      actual->recvbuf(),
      expected->recvbuf(),
      actual->recvBytes(),
      mismatchCount_d,
      stream);
}

inline void runBasicPattern(
    CtranComm* comm,
    int rank,
    int nRanks,
    size_t count,
    int numReplays,
    AlgoDescriptor& desc) {
  auto bufs = desc.makeBuffers(count, rank, nRanks);
  auto expected = desc.makeBuffers(count, rank, nRanks);
  computeExpected(desc, expected.get(), count, comm, rank, nRanks);

  ctran::testing::CtranGraphTestBuilder(comm, rank, nRanks)
      .withNumReplays(numReplays)
      .addCapture([&](ctran::testing::CaptureContext& ctx) {
        desc.capture(bufs.get(), count, ctx);
      })
      .withReset([&](cudaStream_t stream) {
        cudaMemsetAsync(bufs->recvbuf(), 0, bufs->recvBytes(), stream);
      })
      .withDeviceVerify([&](cudaStream_t stream, unsigned int* mc) {
        deviceVerifyAgainstExpected(bufs.get(), expected.get(), mc, stream);
      })
      .withGraphAssertions(
          CtranCudaGraphTestBase::expectGraphNodes(
              desc.expectsHostNodes(comm, count) ? 1 : 0))
      .run();
}

// GraphTestParam: (algo, pattern, count, replayMultiplier)
// Actual replays = baseReplays(pattern) * replayMultiplier
using GraphTestParam = std::tuple<AlgoDescriptor, GraphPattern, size_t, int>;

#ifndef CUDAGRAPH_TEST_PATTERN
static_assert(false && "CUDAGRAPH_TEST_PATTERN must be defined");
#endif

inline void runPattern(
    GraphPattern pattern,
    CtranComm* comm,
    int rank,
    int nRanks,
    size_t count,
    int numReplays,
    AlgoDescriptor& desc) {
  switch (pattern) {
    case GraphPattern::Basic:
      runBasicPattern(comm, rank, nRanks, count, numReplays, desc);
      break;
  }
}

#define DEFINE_CUDAGRAPH_PARAM_TEST(SuiteName, ...)                          \
  class SuiteName : public CtranCudaGraphTestBase,                           \
                    public ::testing::WithParamInterface<GraphTestParam> {}; \
                                                                             \
  TEST_P(SuiteName, CudaGraphOp) {                                           \
    auto [desc, pattern, count, replayMult] = GetParam();                    \
    int numReplays = baseReplays(pattern) * replayMult;                      \
    auto comm = makeCtranComm();                                             \
    ASSERT_NE(comm, nullptr);                                                \
    if (!desc.isSupported(comm.get(), count, numRanks)) {                    \
      GTEST_SKIP() << desc.name << " not supported";                         \
    }                                                                        \
    runPattern(                                                              \
        pattern, comm.get(), globalRank, numRanks, count, numReplays, desc); \
  }                                                                          \
                                                                             \
  std::string SuiteName##TestName(                                           \
      const ::testing::TestParamInfo<GraphTestParam>& info) {                \
    auto& [desc, pattern, count, replayMult] = info.param;                   \
    return desc.name + "_" + patternToString(pattern) + "_" +                \
        std::to_string(count) + "_x" + std::to_string(replayMult);           \
  }                                                                          \
                                                                             \
  INSTANTIATE_TEST_SUITE_P(                                                  \
      SuiteName##Tests,                                                      \
      SuiteName,                                                             \
      ::testing::Combine(                                                    \
          ::testing::Values(__VA_ARGS__),                                    \
          ::testing::Values(CUDAGRAPH_TEST_PATTERN),                         \
          ::testing::Values(1024UL, 8192UL),                                 \
          ::testing::Values(1)),                                             \
      SuiteName##TestName)

// Stress variant: reuses the same test class from DEFINE_CUDAGRAPH_PARAM_TEST
// but with a higher replay multiplier. Must be in a separate .cc / BUCK target
// with a longer re_timeout.
#define DEFINE_CUDAGRAPH_STRESS_TEST(SuiteName, Multiplier, ...) \
  INSTANTIATE_TEST_SUITE_P(                                      \
      SuiteName##StressTests,                                    \
      SuiteName,                                                 \
      ::testing::Combine(                                        \
          ::testing::Values(__VA_ARGS__),                        \
          ::testing::Values(                                     \
              GraphPattern::Basic,                               \
              GraphPattern::MultipleSequential,                  \
              GraphPattern::MultiStream),                        \
          ::testing::Values(8192UL),                             \
          ::testing::Values(Multiplier)),                        \
      SuiteName##TestName)
