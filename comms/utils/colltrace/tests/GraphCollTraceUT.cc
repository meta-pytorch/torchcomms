// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Validates per-collective start detection for graph collectives.
//
// Uses a host sleep node as the "work" between the start and end timestamp
// kernels. While the host callback sleeps, only the start entry has been
// written to the ring buffer and the poll thread should fire
// collEventProgressing for that specific collective.

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <set>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/GpuClockCalibration.h"
#include "comms/utils/colltrace/CollTrace.h"
#include "comms/utils/colltrace/CollTraceHandle.h"
#include "comms/utils/colltrace/CudaWaitEvent.h"
#include "comms/utils/colltrace/GraphCudaWaitEvent.h"
#include "comms/utils/cvars/nccl_cvars.h"

using meta::comms::colltrace::CollTrace;
using meta::comms::colltrace::CollTraceConfig;
using meta::comms::colltrace::CollTraceEvent;
using meta::comms::colltrace::CollTraceHandleTriggerState;
using meta::comms::colltrace::GraphCudaWaitEvent;
using meta::comms::colltrace::ICollMetadata;
using meta::comms::colltrace::ICollTracePlugin;

namespace {

class SimpleMetadata : public ICollMetadata {
 public:
  std::size_t hash() const override {
    return 0;
  }
  bool equals(const ICollMetadata&) const noexcept override {
    return true;
  }
  std::string_view getMetadataType() const noexcept override {
    return "test";
  }
  folly::dynamic toDynamic() const noexcept override {
    return folly::dynamic::object("type", "test");
  }
  void fromDynamic(const folly::dynamic&) noexcept override {}
};

// Plugin that tracks which collIds had collEventProgressing fired.
class ProgressTrackingPlugin : public ICollTracePlugin {
 public:
  std::string_view getName() const noexcept override {
    return "ProgressTrackingPlugin";
  }

  meta::comms::CommsMaybeVoid beforeCollKernelScheduled(
      CollTraceEvent&) noexcept override {
    return folly::unit;
  }

  meta::comms::CommsMaybeVoid afterCollKernelScheduled(
      CollTraceEvent&) noexcept override {
    return folly::unit;
  }

  meta::comms::CommsMaybeVoid afterCollKernelStart(
      CollTraceEvent& event) noexcept override {
    if (event.collRecord) {
      std::lock_guard<std::mutex> lock(mu_);
      startedCollIds_.insert(event.collRecord->getCollId());
    }
    return folly::unit;
  }

  meta::comms::CommsMaybeVoid collEventProgressing(
      CollTraceEvent& event) noexcept override {
    if (event.collRecord) {
      std::lock_guard<std::mutex> lock(mu_);
      progressedCollIds_.insert(event.collRecord->getCollId());
    }
    progressCount_.fetch_add(1);
    return folly::unit;
  }

  meta::comms::CommsMaybeVoid afterCollKernelEnd(
      CollTraceEvent& event) noexcept override {
    if (event.collRecord) {
      std::lock_guard<std::mutex> lock(mu_);
      completedCollIds_.insert(event.collRecord->getCollId());
    }
    return folly::unit;
  }

  std::set<int64_t> getProgressedCollIds() const {
    std::lock_guard<std::mutex> lock(mu_);
    return progressedCollIds_;
  }

  std::set<int64_t> getStartedCollIds() const {
    std::lock_guard<std::mutex> lock(mu_);
    return startedCollIds_;
  }

  std::set<int64_t> getCompletedCollIds() const {
    std::lock_guard<std::mutex> lock(mu_);
    return completedCollIds_;
  }

  int progressCount() const {
    return progressCount_.load();
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mu_);
    progressedCollIds_.clear();
    startedCollIds_.clear();
    completedCollIds_.clear();
    progressCount_.store(0);
  }

 private:
  mutable std::mutex mu_;
  std::set<int64_t> progressedCollIds_;
  std::set<int64_t> startedCollIds_;
  std::set<int64_t> completedCollIds_;
  std::atomic<int> progressCount_{0};
};

// Host callback that sleeps for a fixed duration.
void CUDART_CB hostSleepCallback(void* userData) {
  auto durationMs = reinterpret_cast<uintptr_t>(userData);
  // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
  std::this_thread::sleep_for(std::chrono::milliseconds(durationMs));
}

} // namespace

class GraphColltraceProgressingTest : public ::testing::Test {
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

    auto progressPlugin = std::make_unique<ProgressTrackingPlugin>();
    progressPlugin_ = progressPlugin.get();

    auto plugins = std::vector<std::unique_ptr<ICollTracePlugin>>{};
    plugins.push_back(std::move(progressPlugin));
    CommLogData logData{};
    colltrace_ = std::make_shared<CollTrace>(
        CollTraceConfig{.maxCheckCancelInterval = std::chrono::milliseconds{1}},
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
    if (stream_) {
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaStreamDestroy(stream_);
    }
  }

  // Launch a host callback that sleeps for sleepMs on the capture stream.
  void launchHostSleep(uint64_t sleepMs) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaLaunchHostFunc(
        stream_,
        hostSleepCallback,
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        reinterpret_cast<void*>(static_cast<uintptr_t>(sleepMs)));
  }

  struct CapturedGraph {
    cudaGraph_t graph{nullptr};
    cudaGraphExec_t instance{nullptr};
    std::vector<int64_t> collIds;

    ~CapturedGraph() {
      if (instance) {
        // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
        cudaGraphExecDestroy(instance);
      }
      if (graph) {
        // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
        cudaGraphDestroy(graph);
      }
    }
    CapturedGraph() = default;
    CapturedGraph(CapturedGraph&& o) noexcept
        : graph(std::exchange(o.graph, nullptr)),
          instance(std::exchange(o.instance, nullptr)),
          collIds(std::move(o.collIds)) {}
    CapturedGraph& operator=(CapturedGraph&&) = delete;
    CapturedGraph(const CapturedGraph&) = delete;
    CapturedGraph& operator=(const CapturedGraph&) = delete;
  };

  // Capture N serial collectives, each doing a host sleep as "work".
  CapturedGraph captureSerial(uint32_t numColls, uint64_t sleepMs) {
    CapturedGraph cg;
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
    for (uint32_t c = 0; c < numColls; ++c) {
      auto metadata = std::make_unique<SimpleMetadata>();
      auto waitEvent = std::make_unique<GraphCudaWaitEvent>(stream_);
      auto handle =
          colltrace_
              ->recordCollective(std::move(metadata), std::move(waitEvent))
              .value();
      auto collRecord = handle->getCollRecord();
      if (collRecord.hasValue() && collRecord.value() != nullptr) {
        cg.collIds.push_back(collRecord.value()->getCollId());
      }
      handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
      launchHostSleep(sleepMs);
      handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
    }
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamEndCapture(stream_, &cg.graph);
    EXPECT_NE(cg.graph, nullptr);
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaGraphInstantiate(&cg.instance, cg.graph, nullptr, nullptr, 0);
    return cg;
  }

  cudaStream_t stream_{nullptr};
  std::optional<EnvRAII<bool>> cvarGuard_;
  std::shared_ptr<CollTrace> colltrace_;
  ProgressTrackingPlugin* progressPlugin_{nullptr};
};

// Each collective sleeps 50ms. With a 1ms poll interval, the poll thread
// has many chances to observe the start entry (before the end entry is
// written) and fire collEventProgressing for the specific in-flight
// collective.
TEST_F(GraphColltraceProgressingTest, DetectsInFlightCollective) {
  constexpr uint32_t kNumColls = 3;
  constexpr uint64_t kSleepMs = 50;
  auto cg = captureSerial(kNumColls, kSleepMs);
  ASSERT_NE(cg.instance, nullptr);
  ASSERT_EQ(cg.collIds.size(), kNumColls);

  std::set<int64_t> validCollIds(cg.collIds.begin(), cg.collIds.end());

  // Replay once — 150ms total (3 × 50ms), giving the poll thread plenty
  // of time to observe in-flight entries.
  ASSERT_EQ(cudaGraphLaunch(cg.instance, stream_), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  // Give the poll thread a moment to process remaining entries.
  // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  auto progressedIds = progressPlugin_->getProgressedCollIds();

  // With 50ms per collective and 1ms poll interval, the poll thread
  // should observe at least one in-flight collective.
  EXPECT_FALSE(progressedIds.empty())
      << "Expected collEventProgressing to fire for at least one collective";

  // Every reported collId must be from our captured set.
  for (auto id : progressedIds) {
    EXPECT_TRUE(validCollIds.count(id) > 0)
        << "collEventProgressing reported unknown collId " << id;
  }

  EXPECT_GT(progressPlugin_->progressCount(), 0);

  // After sync, all collectives should have completed.
  auto completedIds = progressPlugin_->getCompletedCollIds();
  EXPECT_EQ(completedIds, validCollIds)
      << "All collectives should be marked completed after stream sync";
}

// Capture concurrent collectives on separate streams so multiple start
// kernels fire before any end kernel. The poll thread should observe
// multiple start entries (without corresponding end entries) simultaneously
// and fire collEventProgressing for each one individually.
TEST_F(GraphColltraceProgressingTest, DetectsMultipleInFlightCollectives) {
  constexpr uint32_t kNumColls = 3;
  constexpr uint64_t kSleepMs = 100;

  // Create per-collective streams forked from the main capture stream.
  std::vector<cudaStream_t> collStreams(kNumColls);
  for (auto& s : collStreams) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamCreate(&s);
  }
  cudaEvent_t forkEvent;
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaEventCreateWithFlags(&forkEvent, cudaEventDisableTiming);

  CapturedGraph cg;
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

  // Fork: main stream → each collective stream.
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaEventRecord(forkEvent, stream_);
  for (uint32_t c = 0; c < kNumColls; ++c) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamWaitEvent(collStreams[c], forkEvent);
  }

  // Launch each collective on its own stream with a host sleep.
  // All start kernels fire concurrently before any end kernel runs.
  std::vector<std::shared_ptr<meta::comms::colltrace::ICollTraceHandle>>
      handles;
  for (uint32_t c = 0; c < kNumColls; ++c) {
    auto metadata = std::make_unique<SimpleMetadata>();
    auto waitEvent = std::make_unique<GraphCudaWaitEvent>(collStreams[c]);
    auto handle =
        colltrace_->recordCollective(std::move(metadata), std::move(waitEvent))
            .value();
    auto collRecord = handle->getCollRecord();
    if (collRecord.hasValue() && collRecord.value() != nullptr) {
      cg.collIds.push_back(collRecord.value()->getCollId());
    }
    handle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
    // Host sleep on the collective's stream — all sleeps run concurrently.
    cudaLaunchHostFunc(
        collStreams[c],
        hostSleepCallback,
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        reinterpret_cast<void*>(static_cast<uintptr_t>(kSleepMs)));
    handle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
    handles.push_back(std::move(handle));
  }

  // Join: all collective streams → main stream.
  for (uint32_t c = 0; c < kNumColls; ++c) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaEventRecord(forkEvent, collStreams[c]);
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamWaitEvent(stream_, forkEvent);
  }

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaStreamEndCapture(stream_, &cg.graph);
  ASSERT_NE(cg.graph, nullptr);
  ASSERT_EQ(
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaGraphInstantiate(&cg.instance, cg.graph, nullptr, nullptr, 0),
      cudaSuccess);
  ASSERT_EQ(cg.collIds.size(), kNumColls);

  std::set<int64_t> validCollIds(cg.collIds.begin(), cg.collIds.end());

  // Replay — all 3 collectives run concurrently with 100ms sleeps.
  // The poll thread (1ms interval) has ~100ms to observe all 3 as
  // simultaneously in-flight.
  ASSERT_EQ(cudaGraphLaunch(cg.instance, stream_), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  auto progressedIds = progressPlugin_->getProgressedCollIds();

  // All 3 collectives should have been observed as in-flight since
  // they run concurrently for 100ms.
  EXPECT_EQ(progressedIds.size(), kNumColls)
      << "Expected all " << kNumColls
      << " concurrent collectives to be detected as in-flight";

  for (auto id : progressedIds) {
    EXPECT_TRUE(validCollIds.count(id) > 0)
        << "collEventProgressing reported unknown collId " << id;
  }

  EXPECT_GT(progressPlugin_->progressCount(), 0);

  // After sync, all collectives should have completed.
  auto completedIds = progressPlugin_->getCompletedCollIds();
  EXPECT_EQ(completedIds, validCollIds)
      << "All concurrent collectives should be marked completed after stream sync";

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaEventDestroy(forkEvent);
  for (auto& s : collStreams) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamDestroy(s);
  }
}

// ---------------------------------------------------------------------------
// Integration test: verify that eager and graph events are merged by
// timestamp in processCompletedEvents. Records one graph collective and
// one eager collective, lets both complete, and verifies the plugin sees
// events in timestamp order.
// ---------------------------------------------------------------------------

TEST_F(GraphColltraceProgressingTest, EagerAndGraphMergedByTimestamp) {
  // Record a graph collective (captured with a host sleep so it takes time).
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t instance = nullptr;

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
  auto graphMetadata = std::make_unique<SimpleMetadata>();
  auto graphWaitEvent = std::make_unique<GraphCudaWaitEvent>(stream_);
  auto graphHandle =
      colltrace_
          ->recordCollective(
              std::move(graphMetadata), std::move(graphWaitEvent))
          .value();
  graphHandle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
  // Short host sleep as the "work".
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaLaunchHostFunc(
      stream_,
      hostSleepCallback,
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      reinterpret_cast<void*>(static_cast<uintptr_t>(10)));
  graphHandle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaStreamEndCapture(stream_, &graph);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

  // Launch the graph — this fires the start+end ring buffer writes.
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphLaunch(instance, stream_);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaStreamSynchronize(stream_);

  // Now record an eager collective on the same stream.
  auto eagerMetadata = std::make_unique<SimpleMetadata>();
  auto eagerWaitEvent =
      std::make_unique<meta::comms::colltrace::CudaWaitEvent>(stream_);
  auto eagerHandle =
      colltrace_
          ->recordCollective(
              std::move(eagerMetadata), std::move(eagerWaitEvent))
          .value();
  eagerHandle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);
  // Minimal work — just a host func.
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaLaunchHostFunc(
      stream_,
      hostSleepCallback,
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      reinterpret_cast<void*>(static_cast<uintptr_t>(10)));
  eagerHandle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
  eagerHandle->trigger(CollTraceHandleTriggerState::KernelStarted);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaStreamSynchronize(stream_);
  eagerHandle->trigger(CollTraceHandleTriggerState::KernelFinished);

  // Wait for the poll thread to process everything.
  // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Verify both collectives completed.
  auto completedIds = progressPlugin_->getCompletedCollIds();
  EXPECT_EQ(completedIds.size(), 2u)
      << "Both graph and eager collectives should complete";

  // The graph collective ran first (lower timestamps), so its events
  // should appear before the eager collective's events in the merged
  // timeline. The exact ordering depends on timestamp comparison, but
  // both should be present.
  auto startedIds = progressPlugin_->getStartedCollIds();
  EXPECT_EQ(startedIds.size(), 2u) << "Both collectives should have started";

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphExecDestroy(instance);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphDestroy(graph);
}
