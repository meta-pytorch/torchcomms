// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "comms/utils/HRDWRingBuffer.h"
#include "comms/utils/HRDWRingBufferDeviceHandle.cuh"
#include "comms/utils/HRDWRingBufferReader.h"
#include "comms/utils/tests/HRDWRingBufferTestTypes.h"

using meta::comms::colltrace::HRDWRingBuffer;
using meta::comms::colltrace::HRDWRingBufferDeviceHandle;
using meta::comms::colltrace::HRDWRingBufferReader;

// kernel to write count entries to the buffer (one per thread)
__global__ void inlineWriteKernel(
    HRDWRingBufferDeviceHandle<TestEvent> rb,
    TestEvent* dataArray,
    int count) {
  if (unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < count) {
    rb.write(dataArray[tid]);
  }
}

// one thread writes a "start" event, does some busy work, then writes an "end"
// event. this is the common pattern for collective telemetry.
enum class EventPhase : uint8_t {
  kStart = 0,
  kEnd = 1,
};

struct TelemetryEvent {
  EventPhase phase;
  uint32_t opId;
};

__global__ void startEndKernel(
    HRDWRingBufferDeviceHandle<TelemetryEvent> rb,
    int numOps) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numOps) {
    rb.write({EventPhase::kStart, static_cast<uint32_t>(tid)});

    // busy loop so the GPU actually spends some time. volatile to prevent the
    // compiler from optimizing it away.
    volatile int sink = 0;
    for (int i = 0; i < 1000; ++i) {
      sink += i;
    }

    rb.write({EventPhase::kEnd, static_cast<uint32_t>(tid)});
  }
}

class DeviceHandleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int deviceCount = 0;
    auto err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaSetDevice(0);
    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  }

  void TearDown() override {
    if (stream_) {
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaStreamDestroy(stream_);
    }
  }

  cudaStream_t stream_{nullptr};
};

TEST_F(DeviceHandleTest, BasicInlineWrite) {
  constexpr int kCount = 32;
  HRDWRingBuffer<TestEvent> buf(64);
  ASSERT_TRUE(buf.valid());

  auto handle = buf.deviceHandle();

  TestEvent* d_dataArray = nullptr;
  ASSERT_EQ(cudaMalloc(&d_dataArray, sizeof(TestEvent) * kCount), cudaSuccess);

  std::vector<TestEvent> h_dataArray(kCount);
  for (int i = 0; i < kCount; ++i) {
    h_dataArray[i] = TestEvent{static_cast<uint32_t>(i + 1)};
  }
  ASSERT_EQ(
      cudaMemcpy(
          d_dataArray,
          h_dataArray.data(),
          sizeof(TestEvent) * kCount,
          cudaMemcpyHostToDevice),
      cudaSuccess);

  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  inlineWriteKernel<<<1, kCount, 0, stream_>>>(handle, d_dataArray, kCount);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  HRDWRingBufferReader<TestEvent> reader(buf);
  std::vector<uint32_t> tags;
  auto result = reader.poll([&](const auto& e, uint64_t) {
    tags.push_back(e.data.tag);
    EXPECT_NE(e.timestamp_ns, 0u) << "GPU timestamp should be non-zero";
  });

  EXPECT_EQ(result.entriesRead, kCount);
  EXPECT_EQ(result.entriesLost, 0u);

  std::sort(tags.begin(), tags.end());

  for (int i = 0; i < kCount; ++i) {
    EXPECT_EQ(tags[i], static_cast<uint32_t>(i + 1));
  }

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaFree(d_dataArray);
}

TEST_F(DeviceHandleTest, MultiBlockConcurrentWrite) {
  constexpr int kThreadsPerBlock = 64;
  constexpr int kNumBlocks = 8;
  constexpr int kTotal = kThreadsPerBlock * kNumBlocks;

  HRDWRingBuffer<TestEvent> buf(1024);
  ASSERT_TRUE(buf.valid());
  auto handle = buf.deviceHandle();

  TestEvent* d_dataArray = nullptr;
  ASSERT_EQ(cudaMalloc(&d_dataArray, sizeof(TestEvent) * kTotal), cudaSuccess);

  std::vector<TestEvent> h_dataArray(kTotal);
  for (int i = 0; i < kTotal; ++i) {
    h_dataArray[i] = TestEvent{static_cast<uint32_t>(i)};
  }
  ASSERT_EQ(
      cudaMemcpy(
          d_dataArray,
          h_dataArray.data(),
          sizeof(TestEvent) * kTotal,
          cudaMemcpyHostToDevice),
      cudaSuccess);

  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  inlineWriteKernel<<<kNumBlocks, kThreadsPerBlock, 0, stream_>>>(
      handle, d_dataArray, kTotal);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  HRDWRingBufferReader<TestEvent> reader(buf);
  std::vector<uint32_t> tags;
  auto result = reader.poll([&](const auto& e, uint64_t) {
    EXPECT_NE(e.timestamp_ns, 0u);
    tags.push_back(e.data.tag);
  });

  EXPECT_EQ(result.entriesRead, kTotal);
  EXPECT_EQ(result.entriesLost, 0u);

  std::sort(tags.begin(), tags.end());
  for (int i = 0; i < kTotal; ++i) {
    EXPECT_EQ(tags[i], static_cast<uint32_t>(i));
  }

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaFree(d_dataArray);
}

// verify paired events with monotonic timestamps.
TEST_F(DeviceHandleTest, StartEndPairing) {
  constexpr int kNumOps = 16;

  HRDWRingBuffer<TelemetryEvent> buf(256);
  ASSERT_TRUE(buf.valid());
  auto handle = buf.deviceHandle();

  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  startEndKernel<<<1, kNumOps, 0, stream_>>>(handle, kNumOps);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  struct OpTiming {
    uint64_t startTs{0};
    uint64_t endTs{0};
  };
  std::vector<OpTiming> timings(kNumOps);

  HRDWRingBufferReader<TelemetryEvent> reader(buf);
  auto result = reader.poll([&](const auto& e, uint64_t) {
    const auto& evt = e.data;
    ASSERT_LT(evt.opId, kNumOps);
    switch (evt.phase) {
      case EventPhase::kStart: {
        timings[evt.opId].startTs = e.timestamp_ns;
        break;
      }
      case EventPhase::kEnd: {
        timings[evt.opId].endTs = e.timestamp_ns;
        break;
      }
    }
  });

  EXPECT_EQ(result.entriesRead, kNumOps * 2);
  EXPECT_EQ(result.entriesLost, 0u);

  for (int i = 0; i < kNumOps; ++i) {
    EXPECT_NE(timings[i].startTs, 0u) << "op " << i << " missing start";
    EXPECT_NE(timings[i].endTs, 0u) << "op " << i << " missing end";
    EXPECT_LE(timings[i].startTs, timings[i].endTs)
        << "op " << i << " end before start";
  }
}

// mix host-side buf.write() kernel launches with inline
// device-side handle.write() on the same buffer.
TEST_F(DeviceHandleTest, InterleavedWithHostWrites) {
  constexpr int kInlineCount = 16;
  constexpr int kHostCount = 8;

  HRDWRingBuffer<TestEvent> buf(64);
  ASSERT_TRUE(buf.valid());
  auto handle = buf.deviceHandle();

  for (int i = 0; i < kHostCount; ++i) {
    ASSERT_EQ(
        buf.write(stream_, TestEvent{static_cast<uint32_t>(1000 + i)}),
        cudaSuccess);
  }

  TestEvent* d_dataArray = nullptr;
  ASSERT_EQ(
      cudaMalloc(&d_dataArray, sizeof(TestEvent) * kInlineCount), cudaSuccess);

  std::vector<TestEvent> h_dataArray(kInlineCount);
  for (int i = 0; i < kInlineCount; ++i) {
    h_dataArray[i] = TestEvent{static_cast<uint32_t>(2000 + i)};
  }
  ASSERT_EQ(
      cudaMemcpy(
          d_dataArray,
          h_dataArray.data(),
          sizeof(TestEvent) * kInlineCount,
          cudaMemcpyHostToDevice),
      cudaSuccess);

  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  inlineWriteKernel<<<1, kInlineCount, 0, stream_>>>(
      handle, d_dataArray, kInlineCount);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  HRDWRingBufferReader<TestEvent> reader(buf);
  uint32_t hostEntries = 0;
  uint32_t inlineEntries = 0;
  auto result = reader.poll([&](const auto& e, uint64_t) {
    auto tag = e.data.tag;
    EXPECT_NE(e.timestamp_ns, 0u);
    if (tag >= 2000) {
      ++inlineEntries;
    } else if (tag >= 1000) {
      ++hostEntries;
    }
  });

  EXPECT_EQ(result.entriesRead, kHostCount + kInlineCount);
  EXPECT_EQ(result.entriesLost, 0u);
  EXPECT_EQ(hostEntries, kHostCount);
  EXPECT_EQ(inlineEntries, kInlineCount);

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaFree(d_dataArray);
}
