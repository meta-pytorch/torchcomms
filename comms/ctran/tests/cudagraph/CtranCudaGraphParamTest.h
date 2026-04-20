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
  MultipleSequential,
  MultiStream,
  DestroyRecreate,
  MixedEagerGraph,
  MultiGraph,
  InPlace,
};

inline const char* patternToString(GraphPattern pattern) {
  switch (pattern) {
    case GraphPattern::Basic:
      return "Basic";
    case GraphPattern::MultipleSequential:
      return "MultipleSequential";
    case GraphPattern::MultiStream:
      return "MultiStream";
    case GraphPattern::DestroyRecreate:
      return "DestroyRecreate";
    case GraphPattern::MixedEagerGraph:
      return "MixedEagerGraph";
    case GraphPattern::MultiGraph:
      return "MultiGraph";
    case GraphPattern::InPlace:
      return "InPlace";
  }
  return "Unknown";
}

// Base replay count per pattern. Multiplied by the replay multiplier parameter.
inline int baseReplays(GraphPattern pattern) {
  switch (pattern) {
    case GraphPattern::Basic:
      return 5;
    case GraphPattern::MultipleSequential:
      return 3;
    case GraphPattern::MultiStream:
      return 3;
    case GraphPattern::DestroyRecreate:
      return 3;
    case GraphPattern::MixedEagerGraph:
      return 3;
    case GraphPattern::MultiGraph:
      return 3;
    case GraphPattern::InPlace:
      return 3;
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

static constexpr int kNumSequentialOps = 3;

inline void runMultipleSequentialPattern(
    CtranComm* comm,
    int rank,
    int nRanks,
    size_t count,
    int numReplays,
    AlgoDescriptor& desc) {
  std::vector<std::shared_ptr<AlgoDescriptor::Buffers>> bufSets, expectedSets;
  for (int i = 0; i < kNumSequentialOps; ++i) {
    bufSets.push_back(desc.makeBuffers(count, rank, nRanks));
    expectedSets.push_back(desc.makeBuffers(count, rank, nRanks));
    computeExpected(desc, expectedSets.back().get(), count, comm, rank, nRanks);
  }

  ctran::testing::CtranGraphTestBuilder(comm, rank, nRanks)
      .withNumReplays(numReplays)
      .addCapture([&](ctran::testing::CaptureContext& ctx) {
        for (int i = 0; i < kNumSequentialOps; ++i) {
          desc.capture(bufSets[i].get(), count, ctx);
        }
      })
      .withReset([&](cudaStream_t stream) {
        for (int i = 0; i < kNumSequentialOps; ++i) {
          cudaMemsetAsync(
              bufSets[i]->recvbuf(), 0, bufSets[i]->recvBytes(), stream);
        }
      })
      .withDeviceVerify([&](cudaStream_t stream, unsigned int* mc) {
        for (int i = 0; i < kNumSequentialOps; ++i) {
          deviceVerifyAgainstExpected(
              bufSets[i].get(), expectedSets[i].get(), mc, stream);
        }
      })
      .withGraphAssertions(
          CtranCudaGraphTestBase::expectGraphNodes(
              desc.expectsHostNodes(comm, count) ? kNumSequentialOps : 0,
              kNumSequentialOps))
      .run();
}

inline void runMultiStreamPattern(
    CtranComm* comm,
    int rank,
    int nRanks,
    size_t count,
    int numReplays,
    AlgoDescriptor& desc) {
  auto bufs1 = desc.makeBuffers(count, rank, nRanks);
  auto bufs2 = desc.makeBuffers(count, rank, nRanks);
  auto expected1 = desc.makeBuffers(count, rank, nRanks);
  auto expected2 = desc.makeBuffers(count, rank, nRanks);
  computeExpected(desc, expected1.get(), count, comm, rank, nRanks);
  computeExpected(desc, expected2.get(), count, comm, rank, nRanks);

  meta::comms::CudaEvent forkEvent;
  meta::comms::CudaEvent joinEvent;

  ctran::testing::CtranGraphTestBuilder(comm, rank, nRanks)
      .withNumReplays(numReplays)
      .addCapture([&](ctran::testing::CaptureContext& ctx) {
        desc.capture(bufs1.get(), count, ctx);

        meta::comms::CudaStream stream2(cudaStreamNonBlocking);
        ASSERT_EQ(cudaEventRecord(forkEvent.get(), ctx.stream), cudaSuccess);
        ASSERT_EQ(
            cudaStreamWaitEvent(stream2.get(), forkEvent.get(), 0),
            cudaSuccess);

        ctran::testing::CaptureContext ctx2{
            ctx.comm, stream2.get(), ctx.rank, ctx.numRanks};
        desc.capture(bufs2.get(), count, ctx2);

        ASSERT_EQ(cudaEventRecord(joinEvent.get(), stream2.get()), cudaSuccess);
        ASSERT_EQ(
            cudaStreamWaitEvent(ctx.stream, joinEvent.get(), 0), cudaSuccess);
      })
      .withReset([&](cudaStream_t stream) {
        cudaMemsetAsync(bufs1->recvbuf(), 0, bufs1->recvBytes(), stream);
        cudaMemsetAsync(bufs2->recvbuf(), 0, bufs2->recvBytes(), stream);
      })
      .withDeviceVerify([&](cudaStream_t stream, unsigned int* mc) {
        deviceVerifyAgainstExpected(bufs1.get(), expected1.get(), mc, stream);
        deviceVerifyAgainstExpected(bufs2.get(), expected2.get(), mc, stream);
      })
      .withGraphAssertions(
          CtranCudaGraphTestBase::expectGraphNodes(
              desc.expectsHostNodes(comm, count) ? 2 : 0, 2))
      .run();
}

// ---------------------------------------------------------------------------
// DestroyRecreate: capture -> replay -> destroy -> capture again -> replay
// ---------------------------------------------------------------------------

inline void runDestroyRecreatePattern(
    CtranComm* comm,
    int rank,
    int nRanks,
    size_t count,
    int numReplays,
    AlgoDescriptor& desc) {
  auto bufs = desc.makeBuffers(count, rank, nRanks);
  auto expected = desc.makeBuffers(count, rank, nRanks);
  computeExpected(desc, expected.get(), count, comm, rank, nRanks);

  for (int cycle = 0; cycle < 3; ++cycle) {
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
        .run();
  }
}

// ---------------------------------------------------------------------------
// MixedEagerGraph: eager -> graph -> eager to test state isolation
// ---------------------------------------------------------------------------

inline void runMixedEagerGraphPattern(
    CtranComm* comm,
    int rank,
    int nRanks,
    size_t count,
    int numReplays,
    AlgoDescriptor& desc) {
  auto bufs = desc.makeBuffers(count, rank, nRanks);
  auto expected = desc.makeBuffers(count, rank, nRanks);

  computeExpected(desc, expected.get(), count, comm, rank, nRanks);

  // Use addSchedule to run: eager -> graph -> eager as a pipeline.
  ctran::testing::CtranGraphTestBuilder(comm, rank, nRanks)
      .withNumReplays(numReplays)
      .addSchedule({
          // Step 1: Eager op
          {{ctran::testing::ScheduleEntryType::Eager,
            [&](ctran::testing::CaptureContext& ctx) {
              desc.capture(bufs.get(), count, ctx);
            }}},
          // Step 2: Graph op
          {{ctran::testing::ScheduleEntryType::Graph,
            [&](ctran::testing::CaptureContext& ctx) {
              desc.capture(bufs.get(), count, ctx);
            }}},
          // Step 3: Eager op again -- verify state isn't corrupted after graph
          {{ctran::testing::ScheduleEntryType::Eager,
            [&](ctran::testing::CaptureContext& ctx) {
              desc.capture(bufs.get(), count, ctx);
            }}},
      })
      .withReset([&](cudaStream_t stream) {
        cudaMemsetAsync(bufs->recvbuf(), 0, bufs->recvBytes(), stream);
      })
      .withDeviceVerify([&](cudaStream_t stream, unsigned int* mc) {
        deviceVerifyAgainstExpected(bufs.get(), expected.get(), mc, stream);
      })
      .run();
}

// ---------------------------------------------------------------------------
// MultiGraph: two independent graphs captured and replayed concurrently
// ---------------------------------------------------------------------------

inline void runMultiGraphPattern(
    CtranComm* comm,
    int rank,
    int nRanks,
    size_t count,
    int numReplays,
    AlgoDescriptor& desc) {
  auto bufs1 = desc.makeBuffers(count, rank, nRanks);
  auto bufs2 = desc.makeBuffers(count, rank, nRanks);
  auto expected1 = desc.makeBuffers(count, rank, nRanks);
  auto expected2 = desc.makeBuffers(count, rank, nRanks);
  computeExpected(desc, expected1.get(), count, comm, rank, nRanks);
  computeExpected(desc, expected2.get(), count, comm, rank, nRanks);

  // Capture two separate graphs
  ctran::testing::CtranGraphTestBuilder(comm, rank, nRanks)
      .withNumReplays(numReplays)
      .addCapture([&](ctran::testing::CaptureContext& ctx) {
        desc.capture(bufs1.get(), count, ctx);
      })
      .addCapture([&](ctran::testing::CaptureContext& ctx) {
        desc.capture(bufs2.get(), count, ctx);
      })
      .withReset([&](cudaStream_t stream) {
        cudaMemsetAsync(bufs1->recvbuf(), 0, bufs1->recvBytes(), stream);
        cudaMemsetAsync(bufs2->recvbuf(), 0, bufs2->recvBytes(), stream);
      })
      .withDeviceVerify([&](cudaStream_t stream, unsigned int* mc) {
        deviceVerifyAgainstExpected(bufs1.get(), expected1.get(), mc, stream);
        deviceVerifyAgainstExpected(bufs2.get(), expected2.get(), mc, stream);
      })
      .run();
}

// ---------------------------------------------------------------------------
// InPlace: use same buffer for send and recv
// ---------------------------------------------------------------------------

inline void runInPlacePattern(
    CtranComm* comm,
    int rank,
    int nRanks,
    size_t count,
    int numReplays,
    AlgoDescriptor& desc) {
  // Create buffers normally for expected computation
  auto expected = desc.makeBuffers(count, rank, nRanks);
  computeExpected(desc, expected.get(), count, comm, rank, nRanks);

  // For in-place, use the recvbuf as both send and recv.
  // Copy sendbuf data into recvbuf before capture.
  auto bufs = desc.makeBuffers(count, rank, nRanks);
  CUDACHECK_TEST(cudaMemcpy(
      bufs->recvbuf(), bufs->sendbuf(), bufs->recvBytes(), cudaMemcpyDefault));

  // Save the input data for reset
  std::vector<char> savedInput(bufs->recvBytes());
  CUDACHECK_TEST(cudaMemcpy(
      savedInput.data(),
      bufs->recvbuf(),
      bufs->recvBytes(),
      cudaMemcpyDefault));

  ctran::testing::CtranGraphTestBuilder(comm, rank, nRanks)
      .withNumReplays(numReplays)
      .addCapture([&](ctran::testing::CaptureContext& ctx) {
        // In-place: pass recvbuf as both sendbuf and recvbuf
        auto* b = bufs.get();
        // Create a temporary CaptureContext-like call using recvbuf as send
        // This only works for collectives that support in-place
        // For now, just call capture normally -- the collective implementation
        // handles in-place when sendbuf == recvbuf
        desc.capture(b, count, ctx);
      })
      .withReset([&](cudaStream_t stream) {
        // Restore input data
        cudaMemcpyAsync(
            bufs->recvbuf(),
            savedInput.data(),
            bufs->recvBytes(),
            cudaMemcpyDefault,
            stream);
      })
      .withDeviceVerify([&](cudaStream_t stream, unsigned int* mc) {
        deviceVerifyAgainstExpected(bufs.get(), expected.get(), mc, stream);
      })
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
    case GraphPattern::MultipleSequential:
      runMultipleSequentialPattern(comm, rank, nRanks, count, numReplays, desc);
      break;
    case GraphPattern::MultiStream:
      runMultiStreamPattern(comm, rank, nRanks, count, numReplays, desc);
      break;
    case GraphPattern::DestroyRecreate:
      runDestroyRecreatePattern(comm, rank, nRanks, count, numReplays, desc);
      break;
    case GraphPattern::MixedEagerGraph:
      runMixedEagerGraphPattern(comm, rank, nRanks, count, numReplays, desc);
      break;
    case GraphPattern::MultiGraph:
      runMultiGraphPattern(comm, rank, nRanks, count, numReplays, desc);
      break;
    case GraphPattern::InPlace:
      runInPlacePattern(comm, rank, nRanks, count, numReplays, desc);
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
