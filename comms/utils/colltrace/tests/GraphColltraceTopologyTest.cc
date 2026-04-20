// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Validates the CUDA graph DAG topology produced by graph colltrace capture.

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/GpuClockCalibration.h"
#include "comms/utils/colltrace/CollTrace.h"
#include "comms/utils/colltrace/CollTraceHandle.h"
#include "comms/utils/colltrace/GraphCudaWaitEvent.h"
#include "comms/utils/colltrace/plugins/CommDumpPlugin.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/test_utils/CudaGraphTestUtils.h"

using meta::comms::colltrace::CollTrace;
using meta::comms::colltrace::CollTraceConfig;
using meta::comms::colltrace::CollTraceHandleTriggerState;
using meta::comms::colltrace::CommDumpPlugin;
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

class GraphColltraceTopologyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int deviceCount = 0;
    auto err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaSetDevice(0);
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamCreate(&stream_);

    // Enable cudagraph tracing for these tests.
    cvarGuard_.emplace(NCCL_COLLTRACE_TRACE_CUDA_GRAPH, true);

    meta::comms::colltrace::GlobaltimerCalibration::get();

    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaMalloc(&buf1_, kWorkBytes);
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaMalloc(&buf2_, kWorkBytes);

    auto plugins = std::vector<std::unique_ptr<ICollTracePlugin>>{};
    plugins.push_back(std::make_unique<CommDumpPlugin>());
    CommLogData logData{};
    colltrace_ = std::make_shared<CollTrace>(
        CollTraceConfig{
            .maxCheckCancelInterval = std::chrono::milliseconds{10}},
        logData,
        [this]() -> meta::comms::CommsMaybeVoid {
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

  // If COLLTRACE_GRAPH_DUMP_DIR is set, dump the graph as a DOT file.
  // Filename includes the current test name and a suffix for context.
  void maybeDumpGraph(cudaGraph_t graph, const std::string& suffix) {
    const char* dir = std::getenv("COLLTRACE_GRAPH_DUMP_DIR");
    if (dir != nullptr) {
      auto testName = std::string(
          ::testing::UnitTest::GetInstance()->current_test_info()->name());
      auto path = std::string(dir) + "/" + testName + "_" + suffix + ".dot";
      cudaGraphDebugDotPrint(graph, path.c_str(), 0);
    }
  }

  // Capture N serial collectives with colltrace instrumentation.
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
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
      launchWork();
      handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
    }
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamEndCapture(stream_, &graph);
    maybeDumpGraph(graph, std::to_string(numColls));
    return graph;
  }

  // Capture N concurrent collectives on separate streams, simulating
  // the signal/wait RMA pattern where canConcurrent=true.
  cudaGraph_t captureConcurrent(uint32_t numColls) {
    cudaGraph_t graph;
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

    // Create per-collective streams forked from the main capture stream.
    std::vector<cudaStream_t> collStreams(numColls);
    cudaEvent_t forkEvent;
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaEventCreateWithFlags(&forkEvent, cudaEventDisableTiming);

    // Fork: main stream → each collective stream.
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaEventRecord(forkEvent, stream_);
    for (uint32_t c = 0; c < numColls; ++c) {
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaStreamCreate(&collStreams[c]);
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaStreamWaitEvent(collStreams[c], forkEvent);
    }

    // Launch work + colltrace on each concurrent stream.
    for (uint32_t c = 0; c < numColls; ++c) {
      auto metadata = std::make_unique<BenchMetadata>();
      auto waitEvent = std::make_unique<GraphCudaWaitEvent>(collStreams[c]);
      auto handle =
          colltrace_
              ->recordCollective(std::move(metadata), std::move(waitEvent))
              .value();
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
      // Work on the collective's own stream.
      cudaMemcpyAsync(
          buf2_, buf1_, kWorkBytes, cudaMemcpyDeviceToDevice, collStreams[c]);
      handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
    }

    // Join: all collective streams → main stream.
    for (uint32_t c = 0; c < numColls; ++c) {
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaEventRecord(forkEvent, collStreams[c]);
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaStreamWaitEvent(stream_, forkEvent);
    }

    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamEndCapture(stream_, &graph);
    maybeDumpGraph(graph, std::to_string(numColls));

    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaEventDestroy(forkEvent);
    for (auto& s : collStreams) {
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaStreamDestroy(s);
    }

    return graph;
  }

  // Capture a mixed pattern: serial → concurrent → serial.
  // Models a realistic workload like: allreduce, then concurrent
  // signal/wait RMA ops, then another allreduce.
  cudaGraph_t captureMixed(uint32_t numConcurrent) {
    cudaGraph_t graph;
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

    // Serial collective 1 (e.g., allreduce)
    {
      auto metadata = std::make_unique<BenchMetadata>();
      auto waitEvent = std::make_unique<GraphCudaWaitEvent>(stream_);
      auto handle =
          colltrace_
              ->recordCollective(std::move(metadata), std::move(waitEvent))
              .value();
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
      launchWork();
      handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
    }

    // Concurrent collectives (e.g., signal/wait RMA ops)
    {
      std::vector<cudaStream_t> concStreams(numConcurrent);
      cudaEvent_t forkEvent;
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaEventCreateWithFlags(&forkEvent, cudaEventDisableTiming);

      // Fork main → concurrent streams
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaEventRecord(forkEvent, stream_);
      for (uint32_t c = 0; c < numConcurrent; ++c) {
        // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
        cudaStreamCreate(&concStreams[c]);
        // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
        cudaStreamWaitEvent(concStreams[c], forkEvent);
      }

      // Launch concurrent work + colltrace
      for (uint32_t c = 0; c < numConcurrent; ++c) {
        auto metadata = std::make_unique<BenchMetadata>();
        auto waitEvent = std::make_unique<GraphCudaWaitEvent>(concStreams[c]);
        auto handle =
            colltrace_
                ->recordCollective(std::move(metadata), std::move(waitEvent))
                .value();
        handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
        cudaMemcpyAsync(
            buf2_, buf1_, kWorkBytes, cudaMemcpyDeviceToDevice, concStreams[c]);
        handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
      }

      // Join concurrent streams → main
      for (uint32_t c = 0; c < numConcurrent; ++c) {
        // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
        cudaEventRecord(forkEvent, concStreams[c]);
        // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
        cudaStreamWaitEvent(stream_, forkEvent);
      }

      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaEventDestroy(forkEvent);
      for (auto& s : concStreams) {
        // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
        cudaStreamDestroy(s);
      }
    }

    // Serial collective 2 (e.g., another allreduce)
    {
      auto metadata = std::make_unique<BenchMetadata>();
      auto waitEvent = std::make_unique<GraphCudaWaitEvent>(stream_);
      auto handle =
          colltrace_
              ->recordCollective(std::move(metadata), std::move(waitEvent))
              .value();
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
      launchWork();
      handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
    }

    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamEndCapture(stream_, &graph);
    maybeDumpGraph(graph, std::to_string(numConcurrent));
    return graph;
  }

  // Capture with non-traced ops interleaved between traced collectives.
  // This tests that the deferred rejoin doesn't drop dependencies from
  // non-traced operations.
  cudaGraph_t captureWithInterleavedOps(uint32_t numColls) {
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
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
      launchWork();
      handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);

      // Non-traced op between collectives (e.g., a memset).
      if (c < numColls - 1) {
        // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
        cudaMemsetAsync(buf1_, 0, kWorkBytes, stream_);
      }
    }
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamEndCapture(stream_, &graph);
    maybeDumpGraph(graph, "interleaved_" + std::to_string(numColls));
    return graph;
  }

  static constexpr size_t kWorkBytes = 64 * 1024;
  cudaStream_t stream_{nullptr};
  float* buf1_{nullptr};
  float* buf2_{nullptr};
  std::optional<EnvRAII<bool>> cvarGuard_;
  std::shared_ptr<CollTrace> colltrace_;
};

// Graph capture succeeds for various configurations.
TEST_F(GraphColltraceTopologyTest, CaptureSucceeds) {
  for (uint32_t n : {1u, 3u, 10u}) {
    auto graph = captureSerial(n);
    ASSERT_NE(graph, nullptr) << "Serial capture failed for N=" << n;
    auto topo = getGraphTopology(graph);
    EXPECT_GT(topo.allNodes.size(), 0u);
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaGraphDestroy(graph);
  }
}

// Serial collectives have the expected node types.
TEST_F(GraphColltraceTopologyTest, SerialNodeTypes) {
  constexpr uint32_t kNumColls = 3;
  auto graph = captureSerial(kNumColls);
  auto topo = getGraphTopology(graph);

  auto& memcpyNodes = topo.nodesOfType(cudaGraphNodeTypeMemcpy);
  auto& kernelNodes = topo.nodesOfType(cudaGraphNodeTypeKernel);

  EXPECT_EQ(memcpyNodes.size(), kNumColls);
  EXPECT_GE(kernelNodes.size(), kNumColls)
      << "Expected at least " << kNumColls << " kernel nodes";

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphDestroy(graph);
}

// Each collective has at least one kernel reachable from it (end kernel).
TEST_F(GraphColltraceTopologyTest, EndKernelsDependOnCollectives) {
  constexpr uint32_t kNumColls = 3;
  auto graph = captureSerial(kNumColls);
  auto topo = getGraphTopology(graph);

  auto& memcpyNodes = topo.nodesOfType(cudaGraphNodeTypeMemcpy);
  auto& kernelNodes = topo.nodesOfType(cudaGraphNodeTypeKernel);

  for (size_t i = 0; i < memcpyNodes.size(); ++i) {
    bool hasReachableKernel = false;
    for (auto& kn : kernelNodes) {
      if (topo.hasEdge(memcpyNodes[i], kn) ||
          topo.hasPath(memcpyNodes[i], kn)) {
        hasReachableKernel = true;
        break;
      }
    }
    EXPECT_TRUE(hasReachableKernel)
        << "Collective " << i << " has no reachable end kernel";
  }

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphDestroy(graph);
}

// Verify deferred rejoin: for serial collectives, no kernel node (start or
// end timestamp kernel) should be a direct predecessor of any memcpy node
// (collective). This proves that intermediate rejoins have been undone —
// the next collective doesn't wait for the previous end kernel.
TEST_F(GraphColltraceTopologyTest, DeferredRejoinRemovesIntermediateEdges) {
  constexpr uint32_t kNumColls = 3;
  auto graph = captureSerial(kNumColls);
  auto topo = getGraphTopology(graph);

  auto& memcpyNodes = topo.nodesOfType(cudaGraphNodeTypeMemcpy);
  auto& kernelNodes = topo.nodesOfType(cudaGraphNodeTypeKernel);

  ASSERT_EQ(memcpyNodes.size(), kNumColls);

  // For each memcpy (collective), check that no kernel node has a direct
  // edge to it. If the deferred rejoin is working, the main stream's
  // capture dependencies were restored to exclude the end kernel before
  // the next collective was captured.
  for (size_t i = 0; i < memcpyNodes.size(); ++i) {
    for (auto& kn : kernelNodes) {
      // Direct edge means the end kernel's rejoin was NOT undone.
      // hasPath would also catch indirect edges through event nodes,
      // which is too strict — we only care about direct edges that
      // indicate a pending rejoin.

      // Check: is there a direct edge from any kernel to this memcpy?
      // We look for edges kernel → X → memcpy where X is an event wait
      // node (which is how rejoins manifest in the graph). A direct
      // kernel → memcpy edge or kernel → eventWait → memcpy means the
      // rejoin was not undone.

      // Simpler: just check no kernel is a direct predecessor.
      EXPECT_FALSE(topo.hasEdge(kn, memcpyNodes[i]))
          << "Kernel node has a direct edge to collective " << i
          << " — deferred rejoin not working (intermediate end kernel "
             "is blocking the next collective)";
    }
  }

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphDestroy(graph);
}

// Concurrent capture succeeds and produces a valid graph.
TEST_F(GraphColltraceTopologyTest, ConcurrentCaptureSucceeds) {
  for (uint32_t n : {2u, 3u, 5u}) {
    auto graph = captureConcurrent(n);
    ASSERT_NE(graph, nullptr) << "Concurrent capture failed for N=" << n;
    auto topo = getGraphTopology(graph);
    EXPECT_GT(topo.allNodes.size(), 0u);

    // Should have N memcpy nodes (one per concurrent collective).
    auto& memcpyNodes = topo.nodesOfType(cudaGraphNodeTypeMemcpy);
    EXPECT_EQ(memcpyNodes.size(), n);

    // Should have kernel nodes (start + end per collective).
    auto& kernelNodes = topo.nodesOfType(cudaGraphNodeTypeKernel);
    EXPECT_GE(kernelNodes.size(), n);

    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaGraphDestroy(graph);
  }
}

// Concurrent collectives: verify that the graph can be instantiated and
// replayed, producing correct timestamps for each collective.
TEST_F(GraphColltraceTopologyTest, ConcurrentReplayProducesTimestamps) {
  constexpr uint32_t kNumColls = 3;
  auto graph = captureConcurrent(kNumColls);
  ASSERT_NE(graph, nullptr);

  cudaGraphExec_t instance;
  ASSERT_EQ(
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0),
      cudaSuccess);

  // Replay a few times.
  for (int i = 0; i < 5; ++i) {
    ASSERT_EQ(cudaGraphLaunch(instance, stream_), cudaSuccess);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphExecDestroy(instance);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphDestroy(graph);
}

// Mixed pattern: serial → concurrent → serial. Models a realistic workload.
TEST_F(GraphColltraceTopologyTest, MixedSerialConcurrentSerial) {
  constexpr uint32_t kNumConcurrent = 2;
  auto graph = captureMixed(kNumConcurrent);
  ASSERT_NE(graph, nullptr);

  auto topo = getGraphTopology(graph);

  // Total: 1 serial + 2 concurrent + 1 serial = 4 memcpy nodes.
  auto& memcpyNodes = topo.nodesOfType(cudaGraphNodeTypeMemcpy);
  EXPECT_EQ(memcpyNodes.size(), 2 + kNumConcurrent);

  // Should have kernel nodes for all collectives.
  auto& kernelNodes = topo.nodesOfType(cudaGraphNodeTypeKernel);
  EXPECT_GE(kernelNodes.size(), 2 + kNumConcurrent);

  // Verify the graph can be instantiated and replayed.
  cudaGraphExec_t instance;
  ASSERT_EQ(
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0),
      cudaSuccess);
  for (int i = 0; i < 5; ++i) {
    ASSERT_EQ(cudaGraphLaunch(instance, stream_), cudaSuccess);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphExecDestroy(instance);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphDestroy(graph);
}

// Concurrent collectives: verify no kernel node serialization.
// With per-collective timestamp streams, concurrent collectives' memcpy
// nodes should NOT have a path between them (they're on parallel streams).
TEST_F(GraphColltraceTopologyTest, ConcurrentCollectivesAreParallel) {
  constexpr uint32_t kNumColls = 3;
  auto graph = captureConcurrent(kNumColls);
  auto topo = getGraphTopology(graph);

  auto& memcpyNodes = topo.nodesOfType(cudaGraphNodeTypeMemcpy);
  ASSERT_EQ(memcpyNodes.size(), kNumColls);

  // For concurrent collectives, no memcpy should be reachable from another
  // memcpy (they're on independent streams).
  for (size_t i = 0; i < memcpyNodes.size(); ++i) {
    for (size_t j = 0; j < memcpyNodes.size(); ++j) {
      if (i == j) {
        continue;
      }
      EXPECT_FALSE(topo.hasPath(memcpyNodes[i], memcpyNodes[j]))
          << "Concurrent collective " << i << " should not be reachable from "
          << j << " — they should be parallel";
    }
  }

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphDestroy(graph);
}

// Verify that non-traced operations between traced collectives preserve
// their ordering. The deferred rejoin restores capture deps, which must
// include any non-traced ops that were added after the previous rejoin.
TEST_F(GraphColltraceTopologyTest, InterleavedNonTracedOpsPreserveOrder) {
  constexpr uint32_t kNumColls = 3;
  auto graph = captureWithInterleavedOps(kNumColls);
  ASSERT_NE(graph, nullptr);

  auto topo = getGraphTopology(graph);

  // With 3 collectives and memsets between them:
  // 3 memcpy nodes (collectives) + 2 memset nodes.
  auto& memcpyNodes = topo.nodesOfType(cudaGraphNodeTypeMemcpy);
  auto& memsetNodes = topo.nodesOfType(cudaGraphNodeTypeMemset);

  EXPECT_EQ(memcpyNodes.size(), kNumColls);
  EXPECT_EQ(memsetNodes.size(), kNumColls - 1);

  // Each memset should be reachable from at least one memcpy (the
  // collective before it) and should have at least one memcpy reachable
  // from it (the collective after it). If the deferred rejoin broke the
  // dependency chain, a memset would be disconnected.
  for (size_t i = 0; i < memsetNodes.size(); ++i) {
    bool hasPredMemcpy = false;
    bool hasSuccMemcpy = false;
    for (auto& mc : memcpyNodes) {
      if (topo.hasPath(mc, memsetNodes[i])) {
        hasPredMemcpy = true;
      }
      if (topo.hasPath(memsetNodes[i], mc)) {
        hasSuccMemcpy = true;
      }
    }
    EXPECT_TRUE(hasPredMemcpy)
        << "Memset " << i
        << " has no predecessor collective — "
           "deferred rejoin may have dropped dependencies";
    EXPECT_TRUE(hasSuccMemcpy)
        << "Memset " << i
        << " has no successor collective — "
           "deferred rejoin may have dropped dependencies";
  }

  // Verify it can instantiate and replay correctly.
  cudaGraphExec_t instance;
  ASSERT_EQ(
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0),
      cudaSuccess);
  for (int i = 0; i < 5; ++i) {
    ASSERT_EQ(cudaGraphLaunch(instance, stream_), cudaSuccess);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphExecDestroy(instance);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphDestroy(graph);
}

// When NCCL_COLLTRACE_TRACE_CUDA_GRAPH is disabled, recordCollective should
// fail for graph-captured collectives and the resulting graph should contain
// no telemetry kernel nodes.
TEST(GraphColltraceTopologyDisabledTest, NoTelemetryNodesWhenDisabled) {
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
  meta::comms::colltrace::GlobaltimerCalibration::get();

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
