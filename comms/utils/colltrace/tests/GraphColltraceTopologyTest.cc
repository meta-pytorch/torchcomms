// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Validates graph colltrace replay behavior: per-replay record accounting via
// the flush path, replay timing consistency, and that no telemetry is produced
// when cudagraph tracing is disabled. Collectives publish their start/end into
// the shared ring in-kernel; these tests drive that ring via the device handle
// (the same write a collective kernel does) so the captured graph replays it.

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/colltrace/CollTrace.h"
#include "comms/utils/colltrace/CollTraceHandle.h"
#include "comms/utils/colltrace/ColltraceDeviceHandle.h"
#include "comms/utils/colltrace/GraphCollTraceEvent.h"
#include "comms/utils/colltrace/GraphCudaWaitEvent.h"
#include "comms/utils/colltrace/plugins/CommDumpPlugin.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/hrdw_ring_buffer/GpuClockCalibration.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"
#include "comms/utils/test_utils/CudaGraphTestUtils.h"

using meta::comms::colltrace::CollTrace;
using meta::comms::colltrace::CollTraceConfig;
using meta::comms::colltrace::ColltraceDeviceHandle;
using meta::comms::colltrace::CollTraceHandleTriggerState;
using meta::comms::colltrace::CommDumpPlugin;
using meta::comms::colltrace::GraphCollTraceEvent;
using meta::comms::colltrace::GraphCollTracePhase;
using meta::comms::colltrace::GraphCudaWaitEvent;
using meta::comms::colltrace::ICollMetadata;
using meta::comms::colltrace::ICollTracePlugin;

namespace {

class BenchMetadata : public ICollMetadata {
 public:
  std::size_t hash() const override {
    return 0;
  }
  bool equals(const ICollMetadata&) const noexcept override {
    return true;
  }
  std::string_view getMetadataType() const noexcept override {
    return "bench";
  }
  folly::dynamic toDynamic() const noexcept override {
    return folly::dynamic::object("type", "bench");
  }
  void fromDynamic(const folly::dynamic&) noexcept override {}
};

} // namespace

class GraphColltraceReplayTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int deviceCount = 0;
    auto err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaSetDevice(0);
    if (!meta::comms::colltrace::graphColltraceSupported(
            "GraphColltraceReplayTest")) {
      GTEST_SKIP()
          << "graph colltrace unsupported on this device (needs sm_90+)";
    }
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamCreate(&stream_);

    cvarGuard_.emplace(NCCL_COLLTRACE_TRACE_CUDA_GRAPH, true);
    hrdw_ring_buffer::GlobaltimerCalibration::get();

    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaMalloc(&buf1_, kWorkBytes);
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaMalloc(&buf2_, kWorkBytes);

    auto dumpPlugin = std::make_unique<CommDumpPlugin>(
        meta::comms::colltrace::CommDumpConfig{.pastCollSize = 100});
    dumpPlugin_ = dumpPlugin.get();

    auto plugins = std::vector<std::unique_ptr<ICollTracePlugin>>{};
    plugins.push_back(std::move(dumpPlugin));
    CommLogData logData{};
    colltrace_ = std::make_shared<CollTrace>(
        CollTraceConfig{
            // Long interval so the poll thread sleeps between cycles; ring
            // entries sit unread until an explicit flush wakes the thread.
            .maxCheckCancelInterval = std::chrono::milliseconds{100000}},
        logData,
        []() -> meta::comms::CommsMaybeVoid {
          // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
          cudaSetDevice(0);
          auto mode = cudaStreamCaptureModeThreadLocal;
          // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
          cudaThreadExchangeStreamCaptureMode(&mode);
          return folly::unit;
        },
        std::move(plugins));
  }

  void TearDown() override {
    colltrace_.reset();
    if (buf2_) {
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaFree(buf2_);
    }
    if (buf1_) {
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaFree(buf1_);
    }
    if (stream_) {
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaStreamDestroy(stream_);
    }
  }

  void launchWork() {
    cudaMemcpyAsync(
        buf2_, buf1_, kWorkBytes, cudaMemcpyDeviceToDevice, stream_);
  }

  // Write a colltrace event into the ring via the device handle colltrace hands
  // to a kernel — the same write a collective kernel does in-kernel. Enqueued
  // on stream_ so it is captured/replayed like the real kernel write.
  void writeColltraceRing(
      const ColltraceDeviceHandle& devHandle,
      GraphCollTracePhase phase) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    hrdw_ring_buffer::launchRingBufferWrite<GraphCollTraceEvent>(
        stream_,
        devHandle.ring.ring,
        devHandle.ring.writeIndex,
        devHandle.ring.mask,
        devHandle.ring.shift,
        GraphCollTraceEvent{devHandle.collId, phase});
  }

  // Capture N serial collectives, each emitting its start/end into the ring
  // in-kernel (via the device handle) around the work — the graph replay then
  // reproduces those writes exactly as a real collective kernel would.
  cudaGraph_t captureSerial(uint32_t numColls) {
    cudaGraph_t graph;
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
    for (uint32_t c = 0; c < numColls; ++c) {
      auto metadata = std::make_unique<BenchMetadata>();
      auto waitEvent = std::make_unique<GraphCudaWaitEvent>(stream_);
      auto handle =
          colltrace_
              ->recordCollective(std::move(metadata), std::move(waitEvent))
              .value();
      auto devHandle = handle->getColltraceDeviceHandle();
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
      writeColltraceRing(devHandle, GraphCollTracePhase::kStart);
      launchWork();
      writeColltraceRing(devHandle, GraphCollTracePhase::kEnd);
      handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
    }
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamEndCapture(stream_, &graph);
    return graph;
  }

  static constexpr size_t kWorkBytes = 64 * 1024;
  cudaStream_t stream_{nullptr};
  float* buf1_{nullptr};
  float* buf2_{nullptr};
  std::optional<EnvRAII<bool>> cvarGuard_;
  std::shared_ptr<CollTrace> colltrace_;
  CommDumpPlugin* dumpPlugin_{nullptr};
};

// Without an explicit flush, ring buffer entries written during replay sit
// unread (the poll thread sleeps on the long interval), so the dump is empty.
TEST_F(GraphColltraceReplayTest, DumpWithoutFlushMissesEvents) {
  constexpr uint32_t kNumColls = 5;
  auto graph = captureSerial(kNumColls);
  ASSERT_NE(graph, nullptr);

  cudaGraphExec_t instance;
  ASSERT_EQ(
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0),
      cudaSuccess);

  ASSERT_EQ(cudaGraphLaunch(instance, stream_), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  auto dumpResult = dumpPlugin_->dump();
  ASSERT_TRUE(dumpResult.hasValue());
  EXPECT_EQ(static_cast<int>(dumpResult.value().pastColls.size()), 0)
      << "Without flush, ring buffer entries should not be processed yet";

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphExecDestroy(instance);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphDestroy(graph);
}

// Flush wakes the poll thread, which drains the ring and processes every graph
// replay event through the plugin pipeline.
TEST_F(GraphColltraceReplayTest, FlushDrainsRingBuffer) {
  constexpr uint32_t kNumColls = 5;
  auto graph = captureSerial(kNumColls);
  ASSERT_NE(graph, nullptr);

  cudaGraphExec_t instance;
  ASSERT_EQ(
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0),
      cudaSuccess);

  ASSERT_EQ(cudaGraphLaunch(instance, stream_), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  colltrace_->waitFlush(colltrace_->requestFlush());

  auto dumpResult = dumpPlugin_->dump();
  ASSERT_TRUE(dumpResult.hasValue());
  EXPECT_EQ(static_cast<int>(dumpResult.value().pastColls.size()), kNumColls)
      << "After flush, all graph replay events should appear in pastColls";

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphExecDestroy(instance);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphDestroy(graph);
}

// Replay a graph multiple times and verify each replay produces valid timing
// (non-zero duration, enqueueTs == startTs for graph clones).
TEST_F(GraphColltraceReplayTest, ReplayTimingValid) {
  constexpr uint32_t kNumColls = 2;
  constexpr int kNumReplays = 3;

  auto graph = captureSerial(kNumColls);
  ASSERT_NE(graph, nullptr);

  cudaGraphExec_t instance;
  ASSERT_EQ(
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0),
      cudaSuccess);

  for (int r = 0; r < kNumReplays; ++r) {
    ASSERT_EQ(cudaGraphLaunch(instance, stream_), cudaSuccess);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  colltrace_->waitFlush(colltrace_->requestFlush());

  auto dump = dumpPlugin_->dump();
  ASSERT_TRUE(dump.hasValue());
  EXPECT_GE(
      static_cast<int>(dump.value().pastColls.size()),
      static_cast<int>(kNumColls * kNumReplays));

  for (const auto& coll : dump.value().pastColls) {
    auto startTs = coll->getTimingInfo().getCollStartTs();
    auto endTs = coll->getTimingInfo().getCollEndTs();
    auto enqueueTs = coll->getTimingInfo().getCollEnqueueTs();
    auto dur =
        std::chrono::duration_cast<std::chrono::microseconds>(endTs - startTs)
            .count();
    EXPECT_GT(dur, 0) << "collId=" << coll->getCollId()
                      << " has non-positive duration";
    EXPECT_GE(endTs, startTs);
    EXPECT_EQ(enqueueTs, startTs)
        << "Graph clone enqueueTs should equal startTs";
  }

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphExecDestroy(instance);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphDestroy(graph);
}

// Reproduces the production bug where pastColls entries had corrupted timing
// because the clone was shared between pastColls and the next replay's
// collEntry. Run multiple replays without flushing between them (the production
// pattern), then flush once and verify that all retained pastColls entries have
// enqueueTs == startTs. With the old shared-record approach, the next replay's
// start event would overwrite startTs while leaving enqueueTs stale.
TEST_F(GraphColltraceReplayTest, ReplayTimingConsistency) {
  constexpr uint32_t kNumColls = 5;
  constexpr int kNumReplays = 10;

  auto graph = captureSerial(kNumColls);
  ASSERT_NE(graph, nullptr);

  cudaGraphExec_t instance;
  ASSERT_EQ(
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0),
      cudaSuccess);

  // Run all replays back-to-back WITHOUT flushing — this is the production
  // pattern where comm_dump_all is called once per step but collectives from
  // the previous step are still in pastColls.
  for (int step = 0; step < kNumReplays; ++step) {
    ASSERT_EQ(cudaGraphLaunch(instance, stream_), cudaSuccess);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  colltrace_->waitFlush(colltrace_->requestFlush());

  auto dump = dumpPlugin_->dump();
  ASSERT_TRUE(dump.hasValue());

  int checked = 0;
  for (const auto& coll : dump.value().pastColls) {
    auto startTs = coll->getTimingInfo().getCollStartTs();
    auto endTs = coll->getTimingInfo().getCollEndTs();
    auto enqueueTs = coll->getTimingInfo().getCollEnqueueTs();
    auto dur =
        std::chrono::duration_cast<std::chrono::microseconds>(endTs - startTs)
            .count();
    EXPECT_GT(dur, 0) << "collId=" << coll->getCollId()
                      << " has non-positive duration " << dur << "us";
    EXPECT_GE(endTs, startTs)
        << "collId=" << coll->getCollId() << ": endTs < startTs";
    EXPECT_EQ(enqueueTs, startTs)
        << "collId=" << coll->getCollId()
        << ": enqueueTs != startTs — pastColls entry was mutated"
           " by a subsequent replay's start event";

    checked++;
  }

  EXPECT_GT(checked, 0) << "pastColls should not be empty after " << kNumReplays
                        << " replays";

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphExecDestroy(instance);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphDestroy(graph);
}

// When NCCL_COLLTRACE_TRACE_CUDA_GRAPH is disabled, recordCollective should
// fail for graph-captured collectives and the resulting graph should contain no
// telemetry kernel nodes.
TEST(GraphColltraceDisabledTest, NoTelemetryNodesWhenDisabled) {
  int deviceCount = 0;
  auto err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaSetDevice(0);

  // Disable the cvar.
  EnvRAII<bool> cvarGuard(NCCL_COLLTRACE_TRACE_CUDA_GRAPH, false);

  // Create CollTrace with the cvar disabled — ring buffer should not
  // be allocated.
  auto plugins = std::vector<std::unique_ptr<ICollTracePlugin>>{};
  plugins.push_back(std::make_unique<CommDumpPlugin>());
  CommLogData logData{};
  auto colltrace = std::make_shared<CollTrace>(
      CollTraceConfig{.maxCheckCancelInterval = std::chrono::milliseconds{10}},
      logData,
      []() -> meta::comms::CommsMaybeVoid {
        // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
        cudaSetDevice(0);
        auto mode = cudaStreamCaptureModeThreadLocal;
        // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
        cudaThreadExchangeStreamCaptureMode(&mode);
        return folly::unit;
      },
      std::move(plugins));

  cudaStream_t stream;
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaStreamCreate(&stream);

  constexpr size_t kWorkBytes = 64 * 1024;
  float* buf1 = nullptr;
  float* buf2 = nullptr;
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaMalloc(&buf1, kWorkBytes);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaMalloc(&buf2, kWorkBytes);

  // Eagerly initialize globaltimer calibration before capture starts —
  // it does cudaHostAlloc which is illegal during stream capture.
  hrdw_ring_buffer::GlobaltimerCalibration::get();

  // Capture a graph — recordCollective should fail for graph-captured
  // collectives when the ring buffer is not allocated.
  cudaGraph_t graph;
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  for (int i = 0; i < 3; ++i) {
    auto metadata = std::make_unique<BenchMetadata>();
    auto waitEvent = std::make_unique<GraphCudaWaitEvent>(stream);
    auto result =
        colltrace->recordCollective(std::move(metadata), std::move(waitEvent));
    EXPECT_TRUE(result.hasError())
        << "recordCollective should fail when cudagraph tracing is disabled";

    cudaMemcpyAsync(buf2, buf1, kWorkBytes, cudaMemcpyDeviceToDevice, stream);
  }

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaStreamEndCapture(stream, &graph);
  ASSERT_NE(graph, nullptr);

  auto topo = getGraphTopology(graph);

  // Only memcpy (work) nodes should exist — no kernel (telemetry) nodes.
  auto& memcpyNodes = topo.nodesOfType(cudaGraphNodeTypeMemcpy);
  auto& kernelNodes = topo.nodesOfType(cudaGraphNodeTypeKernel);
  EXPECT_EQ(memcpyNodes.size(), 3u);
  EXPECT_EQ(kernelNodes.size(), 0u)
      << "Expected no telemetry kernel nodes when "
         "NCCL_COLLTRACE_TRACE_CUDA_GRAPH is disabled";

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphDestroy(graph);
  colltrace.reset();
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaFree(buf2);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaFree(buf1);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaStreamDestroy(stream);
}
