// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "comms/utils/colltrace/common/ColltraceDeviceEventScope.cuh"
#include "comms/utils/colltrace/v2/CollTraceV2.h"

using meta::comms::colltrace::ColltraceDeviceHandle;
using meta::comms::colltracev2::COLLTRACE_INVALID_COLLID;
using meta::comms::colltracev2::CollTraceV2;
using meta::comms::colltracev2::Event;
using meta::comms::colltracev2::EventKind;

namespace {

constexpr std::string_view kComm = "test_comm";

__global__ void collectiveKernel(ColltraceDeviceHandle hdr) {
  meta::comms::colltrace::ColltraceDeviceEventScope scope(hdr);
}

using Shape = std::tuple<uint64_t, EventKind>;

// The first comm registered with a fresh CollTraceV2 gets comm index 1.
constexpr uint64_t id(uint32_t local) {
  return (uint64_t{1} << 48) | local;
}

constexpr auto kStart = EventKind::kCollStartEvent;
constexpr auto kEnd = EventKind::kCollEndEvent;

std::vector<Shape> shapeOf(const std::vector<Event>& events) {
  std::vector<Shape> out;
  out.reserve(events.size());
  for (const auto& e : events) {
    out.emplace_back(e.collId, e.kind);
  }
  return out;
}

} // namespace

class CollTraceV2Test : public ::testing::Test {
 protected:
  void SetUp() override {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    int major = 0;
    ASSERT_EQ(
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0),
        cudaSuccess);
    if (major < 9) {
      GTEST_SKIP() << "in-kernel ring write requires sm_90+";
    }
    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    // CollTraceV2 reads these at construction; tests drain through
    // getCollEvents() unless they say otherwise.
    setenv("COLLTRACE_RING_ENTRIES", "65536", 1);
    setenv("COLLTRACE_MAX_BUFFERED_EVENTS", "1048576", 1);
    setenv("COLLTRACE_COLL_HISTORY", "65536", 1);
    setenv("COLLTRACE_DRAIN_INTERVAL_MS", "3600000", 1);
  }

  void TearDown() override {
    if (stream_ != nullptr) {
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaStreamDestroy(stream_);
    }
  }

  void launch(const ColltraceDeviceHandle& hdr) {
    collectiveKernel<<<1, 32, 0, stream_>>>(hdr);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }

  void sync() {
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  }

  void runCollective(CollTraceV2& ct, int kernels = 1) {
    const auto collId = ct.newCollId(kComm);
    for (int i = 0; i < kernels; ++i) {
      launch(ct.getKernelHandle(collId));
    }
  }

  cudaStream_t stream_{nullptr};
};

TEST_F(CollTraceV2Test, UnregisteredCommIsInert) {
  CollTraceV2 ct;
  EXPECT_EQ(ct.newCollId(kComm), COLLTRACE_INVALID_COLLID);
  EXPECT_EQ(ct.getNumKernels(COLLTRACE_INVALID_COLLID), -1);
  EXPECT_FALSE(ct.getKernelHandle(COLLTRACE_INVALID_COLLID).valid());
  EXPECT_EQ(ct.getLatestCollId(kComm), COLLTRACE_INVALID_COLLID);
  EXPECT_TRUE(ct.getCollEvents(kComm).empty());
}

TEST_F(CollTraceV2Test, RegisterTwiceFails) {
  CollTraceV2 ct;
  EXPECT_TRUE(ct.registerComm(kComm));
  EXPECT_FALSE(ct.registerComm(kComm));
}

TEST_F(CollTraceV2Test, SingleKernelProducesStartAndEnd) {
  CollTraceV2 ct;
  ASSERT_TRUE(ct.registerComm(kComm));

  runCollective(ct);
  sync();

  auto events = ct.getCollEvents(kComm);
  const std::vector<Shape> expected{{id(0), kStart}, {id(0), kEnd}};
  EXPECT_EQ(shapeOf(events), expected);
  ASSERT_EQ(events.size(), 2u);
  EXPECT_LE(events[0].ts, events[1].ts);
  const auto now = std::chrono::system_clock::now();
  EXPECT_LT(now - events[1].ts, std::chrono::minutes{1});
}

TEST_F(CollTraceV2Test, GetCollEventsEmptiesTheBuffer) {
  CollTraceV2 ct;
  ASSERT_TRUE(ct.registerComm(kComm));

  runCollective(ct);
  sync();

  EXPECT_EQ(ct.getCollEvents(kComm).size(), 2u);
  EXPECT_TRUE(ct.getCollEvents(kComm).empty());
}

TEST_F(CollTraceV2Test, ZeroKernelCollectiveGetsAnId) {
  CollTraceV2 ct;
  ASSERT_TRUE(ct.registerComm(kComm));

  const auto collId = ct.newCollId(kComm);
  EXPECT_EQ(collId, id(0));
  EXPECT_EQ(ct.getLatestCollId(kComm), collId);
  EXPECT_EQ(ct.getNumKernels(collId), 0);
  EXPECT_TRUE(ct.getCollEvents(kComm).empty());
}

TEST_F(CollTraceV2Test, MultiKernelCollectiveSharesOneId) {
  CollTraceV2 ct;
  ASSERT_TRUE(ct.registerComm(kComm));

  runCollective(ct, /*kernels=*/3);
  sync();
  EXPECT_EQ(ct.getNumKernels(id(0)), 3);

  const std::vector<Shape> expected{
      {id(0), kStart},
      {id(0), kEnd},
      {id(0), kStart},
      {id(0), kEnd},
      {id(0), kStart},
      {id(0), kEnd},
  };
  EXPECT_EQ(shapeOf(ct.getCollEvents(kComm)), expected);
}

TEST_F(CollTraceV2Test, LatestCollIdAdvancesPerCollective) {
  CollTraceV2 ct;
  ASSERT_TRUE(ct.registerComm(kComm));

  EXPECT_EQ(ct.getLatestCollId(kComm), COLLTRACE_INVALID_COLLID);
  runCollective(ct);
  EXPECT_EQ(ct.getLatestCollId(kComm), id(0));
  runCollective(ct, /*kernels=*/2);
  EXPECT_EQ(ct.getLatestCollId(kComm), id(1));
  sync();
}

TEST_F(CollTraceV2Test, GraphReplayRepeatsTheCapturedIds) {
  CollTraceV2 ct;
  ASSERT_TRUE(ct.registerComm(kComm));

  ASSERT_EQ(
      cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
      cudaSuccess);
  runCollective(ct);
  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(stream_, &graph), cudaSuccess);
  cudaGraphExec_t exec = nullptr;
  ASSERT_EQ(
      cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), cudaSuccess);

  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(cudaGraphLaunch(exec, stream_), cudaSuccess);
  }
  sync();

  // Armed once at capture; every replay writes the same id.
  EXPECT_EQ(ct.getNumKernels(id(0)), 1);
  const std::vector<Shape> expected{
      {id(0), kStart},
      {id(0), kEnd},
      {id(0), kStart},
      {id(0), kEnd},
      {id(0), kStart},
      {id(0), kEnd},
  };
  EXPECT_EQ(shapeOf(ct.getCollEvents(kComm)), expected);

  ASSERT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
}

TEST_F(CollTraceV2Test, KernelCountAgesOutOfTheHistory) {
  setenv("COLLTRACE_COLL_HISTORY", "2", 1);
  CollTraceV2 ct;
  ASSERT_TRUE(ct.registerComm(kComm));

  runCollective(ct, /*kernels=*/2);
  runCollective(ct);
  runCollective(ct);
  EXPECT_EQ(ct.getNumKernels(id(0)), -1) << "aged out of the 2-deep window";
  EXPECT_EQ(ct.getNumKernels(id(1)), 1);
  EXPECT_EQ(ct.getNumKernels(id(2)), 1);
  EXPECT_EQ(ct.getNumKernels(id(3)), -1) << "never issued";
  sync();
}

TEST_F(CollTraceV2Test, DropsOldestPastMaxBufferedEvents) {
  setenv("COLLTRACE_MAX_BUFFERED_EVENTS", "4", 1);
  CollTraceV2 ct;
  ASSERT_TRUE(ct.registerComm(kComm));

  for (int i = 0; i < 4; ++i) {
    runCollective(ct);
  }
  sync();

  const std::vector<Shape> expected{
      {id(2), kStart},
      {id(2), kEnd},
      {id(3), kStart},
      {id(3), kEnd},
  };
  EXPECT_EQ(shapeOf(ct.getCollEvents(kComm)), expected);
}

TEST_F(CollTraceV2Test, DeregisteredCommIsInert) {
  CollTraceV2 ct;
  ASSERT_TRUE(ct.registerComm(kComm));
  runCollective(ct);
  sync();

  const auto stale = ct.getLatestCollId(kComm);
  ct.deregisterComm(kComm);
  EXPECT_EQ(ct.newCollId(kComm), COLLTRACE_INVALID_COLLID);
  EXPECT_FALSE(ct.getKernelHandle(stale).valid());
  EXPECT_TRUE(ct.getCollEvents(kComm).empty());
}

TEST_F(CollTraceV2Test, KernelsMayLaunchLongAfterEnqueueAndOutOfOrder) {
  CollTraceV2 ct;
  ASSERT_TRUE(ct.registerComm(kComm));

  // Two collectives enqueued back to back (as inside a group), kernels
  // launched afterwards in reverse order: the ids still land on the right
  // kernels because the id, not the call order, carries the association.
  const auto first = ct.newCollId(kComm);
  const auto second = ct.newCollId(kComm);
  launch(ct.getKernelHandle(second));
  launch(ct.getKernelHandle(first));
  sync();

  const std::vector<Shape> expected{
      {id(1), kStart},
      {id(1), kEnd},
      {id(0), kStart},
      {id(0), kEnd},
  };
  EXPECT_EQ(shapeOf(ct.getCollEvents(kComm)), expected);
  EXPECT_EQ(ct.getLatestCollId(kComm), second);
}

TEST_F(CollTraceV2Test, TwoCommsGetDistinctIdSpaces) {
  CollTraceV2 ct;
  ASSERT_TRUE(ct.registerComm("a"));
  ASSERT_TRUE(ct.registerComm("b"));

  const auto a = ct.newCollId("a");
  const auto b = ct.newCollId("b");
  EXPECT_NE(a, b);
  launch(ct.getKernelHandle(a));
  launch(ct.getKernelHandle(b));
  sync();

  const std::vector<Shape> expectedA{{a, kStart}, {a, kEnd}};
  const std::vector<Shape> expectedB{{b, kStart}, {b, kEnd}};
  EXPECT_EQ(shapeOf(ct.getCollEvents("a")), expectedA);
  EXPECT_EQ(shapeOf(ct.getCollEvents("b")), expectedB);
}

TEST_F(CollTraceV2Test, TimerThreadDrainsWithoutGetCollEvents) {
  setenv("COLLTRACE_DRAIN_INTERVAL_MS", "20", 1);
  CollTraceV2 ct;
  ASSERT_TRUE(ct.registerComm(kComm));

  runCollective(ct);
  sync();
  std::this_thread::sleep_for(std::chrono::milliseconds{200});

  // The events were already pulled off the ring by the timer thread and are
  // still handed out by the next getCollEvents().
  EXPECT_EQ(ct.getCollEvents(kComm).size(), 2u);
}

TEST_F(CollTraceV2Test, RingLossIsSurvivedAndLaterEventsStillFlow) {
  setenv("COLLTRACE_RING_ENTRIES", "16", 1);
  CollTraceV2 ct;
  ASSERT_TRUE(ct.registerComm(kComm));

  // 20 kernels write 40 entries into a 16-entry ring; only the newest 16
  // survive and the rest are reported lost.
  for (int i = 0; i < 20; ++i) {
    runCollective(ct);
  }
  sync();
  const auto afterLoss = ct.getCollEvents(kComm);
  EXPECT_EQ(afterLoss.size(), 16u);
  EXPECT_EQ(afterLoss.back().collId, id(19));
  EXPECT_EQ(afterLoss.back().kind, kEnd);

  runCollective(ct);
  sync();
  const std::vector<Shape> expected{{id(20), kStart}, {id(20), kEnd}};
  EXPECT_EQ(shapeOf(ct.getCollEvents(kComm)), expected);
}
