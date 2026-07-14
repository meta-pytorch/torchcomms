// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "comms/prims/tests/PipesTraceTest.cuh"
#include "comms/prims/trace/PipesTrace.h"
#include "comms/prims/trace/PipesTraceTypes.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBufferReader.h"

using comms::prims::PipesTrace;
using comms::prims::PipesTraceEntry;
using comms::prims::PipesTraceEvent;
using comms::prims::PipesTraceHandle;

namespace {

constexpr auto kPollInterval = std::chrono::milliseconds{1};
constexpr auto kWaitTimeout = std::chrono::seconds{10};

template <typename DeviceHandle>
PipesTraceHandle toPipesTraceHandle(const DeviceHandle& handle) {
  return PipesTraceHandle{
      .ring = reinterpret_cast<PipesTraceEntry*>(handle.ring),
      .writeIndex = handle.writeIndex,
      .mask = handle.mask,
      .shift = handle.shift};
}

class MappedSpinFlag {
 public:
  explicit MappedSpinFlag(cudaStream_t stream) : stream_(stream) {}

  ~MappedSpinFlag() {
    release();
  }

  cudaError_t init() {
    auto err = cudaHostAlloc(
        reinterpret_cast<void**>(&hostFlag_), sizeof(int), cudaHostAllocMapped);
    if (err != cudaSuccess) {
      return err;
    }
    if (hostFlag_ == nullptr) {
      return cudaErrorMemoryAllocation;
    }
    *hostFlag_ = 0;

    err = cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&deviceFlag_), hostFlag_, 0);
    if (err != cudaSuccess) {
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaFreeHost(hostFlag_);
      hostFlag_ = nullptr;
      deviceFlag_ = nullptr;
    }
    return err;
  }

  volatile int* device() const {
    return deviceFlag_;
  }

  void release() {
    if (hostFlag_ == nullptr) {
      return;
    }
    *hostFlag_ = 1;
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamSynchronize(stream_);
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaFreeHost(hostFlag_);
    hostFlag_ = nullptr;
    deviceFlag_ = nullptr;
  }

 private:
  cudaStream_t stream_{nullptr};
  int* hostFlag_{nullptr};
  int* deviceFlag_{nullptr};
};

class PipesTraceCudaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int deviceCount = 0;
    auto err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    int major = 0;
    ASSERT_EQ(
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0),
        cudaSuccess);
    if (major < 9) {
      GTEST_SKIP() << "HRDW ring buffer requires sm_90+";
    }

    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  }

  void TearDown() override {
    if (stream_ != nullptr) {
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaStreamDestroy(stream_);
    }
  }

  cudaStream_t stream_{nullptr};
};

void waitForReceivedEvents(
    std::mutex& mu,
    const std::vector<PipesTraceEvent>& received,
    size_t expected,
    const char* timeoutMessage) {
  auto deadline = std::chrono::steady_clock::now() + kWaitTimeout;
  while (true) {
    {
      std::lock_guard<std::mutex> lock(mu);
      if (received.size() >= expected) {
        return;
      }
    }
    ASSERT_LT(std::chrono::steady_clock::now(), deadline) << timeoutMessage;
    // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
    std::this_thread::sleep_for(kPollInterval);
  }
}

} // namespace

TEST(PipesTraceTest, NormalizeRingSizeZero) {
  EXPECT_EQ(PipesTrace::normalizeRingSize(0), 0u);
}

TEST(PipesTraceTest, NormalizeRingSizeClampsLarge) {
  EXPECT_EQ(
      PipesTrace::normalizeRingSize(1ULL << 32),
      static_cast<uint32_t>(1ULL << 31));
}

TEST(PipesTraceTest, NormalizeRingSizePassthrough) {
  EXPECT_EQ(PipesTrace::normalizeRingSize(1024), 1024u);
}

TEST_F(PipesTraceCudaTest, EnsureCreatesBuffer) {
  PipesTrace trace;
  trace.ensure(64, kPollInterval);
  auto handle = trace.deviceHandle();
  EXPECT_NE(handle.ring, nullptr);
  EXPECT_NE(handle.writeIndex, nullptr);
}

TEST_F(PipesTraceCudaTest, EnsureZeroSizeIsNoOp) {
  PipesTrace trace;
  trace.ensure(0, kPollInterval);
  auto handle = trace.deviceHandle();
  EXPECT_EQ(handle.ring, nullptr);
}

TEST_F(PipesTraceCudaTest, DeviceHandleNullBeforeEnsure) {
  PipesTrace trace;
  auto handle = trace.deviceHandle();
  EXPECT_EQ(handle.ring, nullptr);
}

TEST_F(PipesTraceCudaTest, DestructionAfterEnsure) {
  auto trace = std::make_unique<PipesTrace>();
  trace->ensure(64, kPollInterval);
  auto handle = trace->deviceHandle();
  EXPECT_NE(handle.ring, nullptr);
  trace.reset();
}

TEST_F(PipesTraceCudaTest, EnsureIdempotent) {
  PipesTrace trace;
  trace.ensure(64, kPollInterval);
  auto handle1 = trace.deviceHandle();

  trace.ensure(64, kPollInterval);
  auto handle2 = trace.deviceHandle();

  EXPECT_EQ(handle1.ring, handle2.ring);
  EXPECT_EQ(handle1.writeIndex, handle2.writeIndex);
}

TEST_F(PipesTraceCudaTest, WriteAndReadEvents) {
  using Buffer = hrdw_ring_buffer::HRDWRingBuffer<PipesTraceEvent>;
  using Reader = hrdw_ring_buffer::HRDWRingBufferReader<PipesTraceEvent>;

  constexpr int kNumEvents = 10;
  Buffer buf(64);
  ASSERT_TRUE(buf.valid());
  auto handle = toPipesTraceHandle(buf.deviceHandle());

  comms::prims::test::launchWriteEvents(handle, kNumEvents, stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  Reader reader(buf);
  uint64_t eventsRead = 0;
  auto result = reader.poll([&](const auto& entry, uint64_t) {
    EXPECT_NE(entry.timestamp, 0u);
    EXPECT_EQ(
        entry.data.type,
        static_cast<uint8_t>(
            comms::prims::PipesTraceEventType::kHierAgNvlTaskDone));
    ++eventsRead;
  });

  EXPECT_EQ(result.entriesRead, kNumEvents);
  EXPECT_EQ(eventsRead, kNumEvents);
  EXPECT_EQ(result.entriesLost, 0u);
}

TEST_F(PipesTraceCudaTest, EventsVisibleBeforeKernelCompletes) {
  using Buffer = hrdw_ring_buffer::HRDWRingBuffer<PipesTraceEvent>;
  using Reader = hrdw_ring_buffer::HRDWRingBufferReader<PipesTraceEvent>;

  constexpr int kNumEvents = 10;
  Buffer buf(64);
  ASSERT_TRUE(buf.valid());
  auto handle = toPipesTraceHandle(buf.deviceHandle());

  MappedSpinFlag releaseFlag(stream_);
  ASSERT_EQ(releaseFlag.init(), cudaSuccess);

  comms::prims::test::launchWriteAndSpin(
      handle, releaseFlag.device(), kNumEvents, stream_);

  Reader reader(buf);
  uint64_t totalRead = 0;
  auto deadline = std::chrono::steady_clock::now() + kWaitTimeout;
  while (totalRead < static_cast<uint64_t>(kNumEvents)) {
    reader.poll([&](const auto& entry, uint64_t) {
      EXPECT_NE(entry.timestamp, 0u);
      ++totalRead;
    });

    ASSERT_LT(std::chrono::steady_clock::now(), deadline)
        << "Timed out waiting for events while kernel is still running. Got "
        << totalRead << "/" << kNumEvents << " events.";
    // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
    std::this_thread::sleep_for(kPollInterval);
  }

  EXPECT_EQ(totalRead, static_cast<uint64_t>(kNumEvents));
  EXPECT_EQ(cudaStreamQuery(stream_), cudaErrorNotReady)
      << "Kernel should still be spinning on the flag";

  releaseFlag.release();
}

TEST_F(PipesTraceCudaTest, PollThreadPicksUpEventsBeforeKernelEnds) {
  constexpr int kNumEvents = 10;

  std::mutex mu;
  std::vector<PipesTraceEvent> received;

  PipesTrace trace;
  trace.ensure(64, kPollInterval, [&](const PipesTraceEvent& event, uint64_t) {
    std::lock_guard<std::mutex> lock(mu);
    received.push_back(event);
  });
  auto handle = trace.deviceHandle();
  ASSERT_NE(handle.ring, nullptr);

  MappedSpinFlag releaseFlag(stream_);
  ASSERT_EQ(releaseFlag.init(), cudaSuccess);

  comms::prims::test::launchWriteAndSpin(
      handle, releaseFlag.device(), kNumEvents, stream_);

  waitForReceivedEvents(
      mu,
      received,
      kNumEvents,
      "Timed out waiting for poll thread to deliver events");

  EXPECT_EQ(cudaStreamQuery(stream_), cudaErrorNotReady)
      << "Kernel should still be spinning on the flag";

  releaseFlag.release();

  std::lock_guard<std::mutex> lock(mu);
  ASSERT_EQ(received.size(), static_cast<size_t>(kNumEvents));
  for (int i = 0; i < kNumEvents; i++) {
    EXPECT_EQ(received[i].step, static_cast<uint32_t>(i));
    EXPECT_EQ(received[i].detail, static_cast<uint16_t>(i));
    EXPECT_EQ(
        received[i].type,
        static_cast<uint8_t>(
            comms::prims::PipesTraceEventType::kHierAgNvlTaskDone));
  }
}

TEST_F(PipesTraceCudaTest, PollThreadConsumesEagerEvents) {
  constexpr int kNumEvents = 20;

  std::mutex mu;
  std::vector<PipesTraceEvent> received;

  PipesTrace trace;
  trace.ensure(64, kPollInterval, [&](const PipesTraceEvent& event, uint64_t) {
    std::lock_guard<std::mutex> lock(mu);
    received.push_back(event);
  });
  auto handle = trace.deviceHandle();
  ASSERT_NE(handle.ring, nullptr);

  comms::prims::test::launchWriteEvents(handle, kNumEvents, stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  waitForReceivedEvents(
      mu,
      received,
      kNumEvents,
      "Timed out waiting for poll thread to deliver events");

  std::lock_guard<std::mutex> lock(mu);
  ASSERT_EQ(received.size(), static_cast<size_t>(kNumEvents));
  for (int i = 0; i < kNumEvents; i++) {
    EXPECT_EQ(received[i].step, static_cast<uint32_t>(i));
    EXPECT_EQ(received[i].detail, static_cast<uint16_t>(i));
    EXPECT_EQ(
        received[i].type,
        static_cast<uint8_t>(
            comms::prims::PipesTraceEventType::kHierAgNvlTaskDone));
  }
}

TEST_F(PipesTraceCudaTest, MultipleKernelLaunches) {
  using Buffer = hrdw_ring_buffer::HRDWRingBuffer<PipesTraceEvent>;
  using Reader = hrdw_ring_buffer::HRDWRingBufferReader<PipesTraceEvent>;

  Buffer buf(256);
  ASSERT_TRUE(buf.valid());
  auto handle = toPipesTraceHandle(buf.deviceHandle());

  constexpr int kLaunches = 5;
  constexpr int kEventsPerLaunch = 10;

  for (int i = 0; i < kLaunches; i++) {
    comms::prims::test::launchWriteEvents(handle, kEventsPerLaunch, stream_);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  Reader reader(buf);
  uint64_t totalRead = 0;
  auto result = reader.poll([&](const auto& entry, uint64_t) {
    EXPECT_NE(entry.timestamp, 0u);
    ++totalRead;
  });

  EXPECT_EQ(totalRead, static_cast<uint64_t>(kLaunches * kEventsPerLaunch));
  EXPECT_EQ(result.entriesLost, 0u);
}

TEST_F(PipesTraceCudaTest, GraphCaptureAndReplay) {
  using Buffer = hrdw_ring_buffer::HRDWRingBuffer<PipesTraceEvent>;
  using Reader = hrdw_ring_buffer::HRDWRingBufferReader<PipesTraceEvent>;

  constexpr int kNumEvents = 10;
  Buffer buf(256);
  ASSERT_TRUE(buf.valid());
  auto handle = toPipesTraceHandle(buf.deviceHandle());

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t instance = nullptr;

  ASSERT_EQ(
      cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal),
      cudaSuccess);
  comms::prims::test::launchWriteEvents(handle, kNumEvents, stream_);
  ASSERT_EQ(cudaStreamEndCapture(stream_, &graph), cudaSuccess);
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(
      cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0), cudaSuccess);

  Reader reader(buf);
  auto captureResult = reader.poll([](const auto&, uint64_t) {});
  EXPECT_EQ(captureResult.entriesRead, 0u);

  ASSERT_EQ(cudaGraphLaunch(instance, stream_), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  uint64_t totalRead = 0;
  auto replayResult = reader.poll([&](const auto& entry, uint64_t) {
    EXPECT_NE(entry.timestamp, 0u);
    EXPECT_EQ(
        entry.data.type,
        static_cast<uint8_t>(
            comms::prims::PipesTraceEventType::kHierAgNvlTaskDone));
    ++totalRead;
  });

  EXPECT_EQ(totalRead, static_cast<uint64_t>(kNumEvents));
  EXPECT_EQ(replayResult.entriesLost, 0u);

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphExecDestroy(instance);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphDestroy(graph);
}

TEST_F(PipesTraceCudaTest, GraphMultipleReplays) {
  using Buffer = hrdw_ring_buffer::HRDWRingBuffer<PipesTraceEvent>;
  using Reader = hrdw_ring_buffer::HRDWRingBufferReader<PipesTraceEvent>;

  constexpr int kNumEvents = 5;
  constexpr int kReplays = 10;
  Buffer buf(256);
  ASSERT_TRUE(buf.valid());
  auto handle = toPipesTraceHandle(buf.deviceHandle());

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t instance = nullptr;

  ASSERT_EQ(
      cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal),
      cudaSuccess);
  comms::prims::test::launchWriteEvents(handle, kNumEvents, stream_);
  ASSERT_EQ(cudaStreamEndCapture(stream_, &graph), cudaSuccess);
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(
      cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0), cudaSuccess);

  for (int i = 0; i < kReplays; i++) {
    ASSERT_EQ(cudaGraphLaunch(instance, stream_), cudaSuccess);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  Reader reader(buf);
  uint64_t totalRead = 0;
  auto result = reader.poll([&](const auto& entry, uint64_t) {
    EXPECT_NE(entry.timestamp, 0u);
    ++totalRead;
  });

  EXPECT_EQ(
      totalRead + result.entriesLost,
      static_cast<uint64_t>(kNumEvents * kReplays));

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphExecDestroy(instance);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphDestroy(graph);
}

TEST_F(PipesTraceCudaTest, PollThreadWithGraphReplay) {
  constexpr int kNumEvents = 10;
  constexpr int kReplays = 3;
  constexpr uint64_t kTotalEvents = kNumEvents * kReplays;

  std::mutex mu;
  std::vector<PipesTraceEvent> received;

  PipesTrace trace;
  trace.ensure(256, kPollInterval, [&](const PipesTraceEvent& event, uint64_t) {
    std::lock_guard<std::mutex> lock(mu);
    received.push_back(event);
  });
  auto handle = trace.deviceHandle();
  ASSERT_NE(handle.ring, nullptr);

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t instance = nullptr;

  ASSERT_EQ(
      cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal),
      cudaSuccess);
  comms::prims::test::launchWriteEvents(handle, kNumEvents, stream_);
  ASSERT_EQ(cudaStreamEndCapture(stream_, &graph), cudaSuccess);
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(
      cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0), cudaSuccess);

  {
    std::lock_guard<std::mutex> lock(mu);
    EXPECT_EQ(received.size(), 0u);
  }

  for (int i = 0; i < kReplays; i++) {
    ASSERT_EQ(cudaGraphLaunch(instance, stream_), cudaSuccess);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  waitForReceivedEvents(
      mu,
      received,
      kTotalEvents,
      "Timed out waiting for poll thread to deliver graph replay events");

  {
    std::lock_guard<std::mutex> lock(mu);
    ASSERT_EQ(received.size(), static_cast<size_t>(kTotalEvents));
    for (size_t i = 0; i < received.size(); i++) {
      EXPECT_EQ(received[i].step, static_cast<uint32_t>(i % kNumEvents));
      EXPECT_EQ(
          received[i].type,
          static_cast<uint8_t>(
              comms::prims::PipesTraceEventType::kHierAgNvlTaskDone));
    }
  }

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphExecDestroy(instance);
  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaGraphDestroy(graph);
}

TEST_F(PipesTraceCudaTest, MultiBlockWriteAndRead) {
  using Buffer = hrdw_ring_buffer::HRDWRingBuffer<PipesTraceEvent>;
  using Reader = hrdw_ring_buffer::HRDWRingBufferReader<PipesTraceEvent>;

  constexpr int kNumBlocks = 8;
  constexpr int kEventsPerBlock = 10;
  constexpr uint64_t kTotalEvents = kNumBlocks * kEventsPerBlock;

  Buffer buf(256);
  ASSERT_TRUE(buf.valid());
  auto handle = toPipesTraceHandle(buf.deviceHandle());

  comms::prims::test::launchWriteEventsMultiBlock(
      handle, kNumBlocks, kEventsPerBlock, stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  Reader reader(buf);
  uint64_t totalRead = 0;
  auto result = reader.poll([&](const auto& entry, uint64_t) {
    EXPECT_NE(entry.timestamp, 0u);
    EXPECT_EQ(
        entry.data.type,
        static_cast<uint8_t>(
            comms::prims::PipesTraceEventType::kHierAgNvlTaskDone));
    EXPECT_LT(entry.data.rank, kNumBlocks);
    ++totalRead;
  });

  EXPECT_EQ(totalRead + result.entriesLost, kTotalEvents);
  EXPECT_EQ(totalRead, result.entriesRead);
}

TEST_F(PipesTraceCudaTest, PollThreadMultiBlockEvents) {
  constexpr int kNumBlocks = 8;
  constexpr int kEventsPerBlock = 10;
  constexpr uint64_t kTotalEvents = kNumBlocks * kEventsPerBlock;

  std::mutex mu;
  std::vector<PipesTraceEvent> received;

  PipesTrace trace;
  trace.ensure(256, kPollInterval, [&](const PipesTraceEvent& event, uint64_t) {
    std::lock_guard<std::mutex> lock(mu);
    received.push_back(event);
  });
  auto handle = trace.deviceHandle();
  ASSERT_NE(handle.ring, nullptr);

  comms::prims::test::launchWriteEventsMultiBlock(
      handle, kNumBlocks, kEventsPerBlock, stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  waitForReceivedEvents(
      mu,
      received,
      kTotalEvents,
      "Timed out waiting for poll thread to deliver multi-block events");

  std::lock_guard<std::mutex> lock(mu);
  ASSERT_EQ(received.size(), static_cast<size_t>(kTotalEvents));

  std::set<uint8_t> seenBlocks;
  for (const auto& event : received) {
    EXPECT_EQ(
        event.type,
        static_cast<uint8_t>(
            comms::prims::PipesTraceEventType::kHierAgNvlTaskDone));
    EXPECT_LT(event.rank, kNumBlocks);
    seenBlocks.insert(event.rank);
  }
  EXPECT_EQ(seenBlocks.size(), static_cast<size_t>(kNumBlocks));
}
