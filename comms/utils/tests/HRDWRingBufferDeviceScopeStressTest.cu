// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// GPU stress tests for HRDWRingBuffer with MemoryCoherenceScope::Device.
// Exercises multi-stream and multi-block concurrent writes, timestamp
// ordering, wrap-around under lapping pressure, and drain correctness.

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <algorithm>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBufferDeviceHandle.cuh"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBufferReader.h"

using hrdw_ring_buffer::HRDWRingBuffer;
using hrdw_ring_buffer::HRDWRingBufferDeviceHandle;
using hrdw_ring_buffer::HRDWRingBufferReader;
using hrdw_ring_buffer::MemoryCoherenceScope;

struct StressEvent {
  uint32_t tag;
};

using DeviceBuffer = HRDWRingBuffer<StressEvent, MemoryCoherenceScope::Device>;
using DeviceReader =
    HRDWRingBufferReader<StressEvent, MemoryCoherenceScope::Device>;
using DeviceHandle =
    HRDWRingBufferDeviceHandle<StressEvent, MemoryCoherenceScope::Device>;

__global__ void
stressWriteKernel(DeviceHandle handle, int writesPerThread, uint32_t tagBase) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < writesPerThread; ++i) {
    handle.write(StressEvent{tagBase + tid});
  }
}

struct StressConfig {
  uint32_t ringSize;
  int numStreams;
  int numWrites;

  int totalWrites() const {
    return numStreams * numWrites;
  }
};

static const StressConfig kStressConfigs[] = {
    {64, 4, 500},
    {256, 8, 200},
    {4096, 1, 2000},
    {4096, 10, 200},
    {16, 1, 1000},
};

class DeviceScopeStressTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int deviceCount = 0;
    auto err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaSetDevice(0);
  }
};

TEST_F(DeviceScopeStressTest, MultiStreamWriteAndDrain) {
  for (const auto& cfg : kStressConfigs) {
    SCOPED_TRACE(
        "ring=" + std::to_string(cfg.ringSize) +
        " streams=" + std::to_string(cfg.numStreams) +
        " writes=" + std::to_string(cfg.numWrites));

    DeviceBuffer buf(cfg.ringSize);
    ASSERT_TRUE(buf.valid());
    DeviceReader reader(buf);

    std::vector<cudaStream_t> streams(cfg.numStreams);
    for (auto& s : streams) {
      ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    }

    for (int i = 0; i < cfg.numWrites; ++i) {
      auto& s = streams[i % cfg.numStreams];
      ASSERT_EQ(buf.write(s, StressEvent{0}), cudaSuccess);
    }

    for (auto& s : streams) {
      ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    }

    uint64_t badEntries = 0;
    auto result = reader.poll(streams[0], [&](const auto& e, uint64_t) {
      if (e.timestamp == 0) {
        ++badEntries;
      }
    });
    ASSERT_EQ(result.error, cudaSuccess);
    EXPECT_EQ(badEntries, 0u);
    EXPECT_EQ(
        result.entriesRead + result.entriesLost,
        static_cast<uint64_t>(cfg.numWrites));

    for (auto& s : streams) {
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaStreamDestroy(s);
    }
  }
}

TEST_F(DeviceScopeStressTest, SingleStreamMonotonicTimestamps) {
  for (const auto& cfg : kStressConfigs) {
    if (cfg.numStreams != 1) {
      continue;
    }
    SCOPED_TRACE(
        "ring=" + std::to_string(cfg.ringSize) +
        " writes=" + std::to_string(cfg.numWrites));

    DeviceBuffer buf(cfg.ringSize);
    ASSERT_TRUE(buf.valid());
    DeviceReader reader(buf);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    for (int i = 0; i < cfg.numWrites; ++i) {
      ASSERT_EQ(buf.write(stream, StressEvent{0}), cudaSuccess);
    }

    uint64_t badEntries = 0;
    uint64_t prevTimestamp = 0;
    auto result = reader.poll(stream, [&](const auto& e, uint64_t) {
      if (e.timestamp == 0) {
        ++badEntries;
      }
      if (prevTimestamp > 0 && e.timestamp < prevTimestamp) {
        ++badEntries;
      }
      prevTimestamp = e.timestamp;
    });
    ASSERT_EQ(result.error, cudaSuccess);
    EXPECT_EQ(badEntries, 0u);
    EXPECT_EQ(
        result.entriesRead + result.entriesLost,
        static_cast<uint64_t>(cfg.numWrites));

    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamDestroy(stream);
  }
}

TEST_F(DeviceScopeStressTest, MultiBlockConcurrentInlineWrites) {
  constexpr int kNumBlocks = 32;
  constexpr int kThreadsPerBlock = 128;
  constexpr int kWritesPerThread = 10;
  constexpr int kTotal = kNumBlocks * kThreadsPerBlock * kWritesPerThread;

  DeviceBuffer buf(65536);
  ASSERT_TRUE(buf.valid());
  DeviceReader reader(buf);

  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  auto handle = buf.deviceHandle();
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  stressWriteKernel<<<kNumBlocks, kThreadsPerBlock, 0, stream>>>(
      handle, kWritesPerThread, 0);

  uint64_t badTimestamps = 0;
  auto result = reader.poll(stream, [&](const auto& e, uint64_t) {
    if (e.timestamp == 0) {
      ++badTimestamps;
    }
  });
  ASSERT_EQ(result.error, cudaSuccess);
  EXPECT_EQ(badTimestamps, 0u);
  EXPECT_EQ(result.entriesRead, kTotal);
  EXPECT_EQ(result.entriesLost, 0u);

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaStreamDestroy(stream);
}

TEST_F(DeviceScopeStressTest, LappingPressureDrainAccounting) {
  constexpr uint32_t kRingSize = 16;
  constexpr int kTotalWrites = 1000;

  DeviceBuffer buf(kRingSize);
  ASSERT_TRUE(buf.valid());
  DeviceReader reader(buf);

  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  for (int i = 0; i < kTotalWrites; ++i) {
    ASSERT_EQ(
        buf.write(stream, StressEvent{static_cast<uint32_t>(i)}), cudaSuccess);
  }

  auto result = reader.poll(stream, [](const auto&, uint64_t) {});
  ASSERT_EQ(result.error, cudaSuccess);
  EXPECT_EQ(result.entriesRead + result.entriesLost, kTotalWrites);
  EXPECT_EQ(result.entriesRead, kRingSize);
  EXPECT_EQ(
      result.entriesLost, static_cast<uint64_t>(kTotalWrites - kRingSize));

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaStreamDestroy(stream);
}

TEST_F(DeviceScopeStressTest, RepeatedDrainResetCycles) {
  constexpr int kCycles = 20;
  constexpr int kWritesPerCycle = 100;

  DeviceBuffer buf(256);
  ASSERT_TRUE(buf.valid());
  DeviceReader reader(buf);

  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  auto handle = buf.deviceHandle();

  for (int cycle = 0; cycle < kCycles; ++cycle) {
    // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
    stressWriteKernel<<<1, kWritesPerCycle, 0, stream>>>(
        handle, 1, static_cast<uint32_t>(cycle * 1000));

    uint64_t readCount = 0;
    auto result =
        reader.poll(stream, [&](const auto&, uint64_t) { ++readCount; });
    ASSERT_EQ(result.error, cudaSuccess);
    EXPECT_EQ(readCount, kWritesPerCycle);
    EXPECT_EQ(result.entriesLost, 0u);
  }

  // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
  cudaStreamDestroy(stream);
}

TEST_F(DeviceScopeStressTest, MultiStreamInlineWriteNeverBlocks) {
  constexpr int kNumStreams = 4;
  constexpr int kWritesPerStream = 500;
  constexpr uint32_t kRingSize = 32;

  DeviceBuffer buf(kRingSize);
  ASSERT_TRUE(buf.valid());
  DeviceReader reader(buf);

  auto handle = buf.deviceHandle();

  cudaStream_t streams[kNumStreams];
  for (auto& s : streams) {
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
  }

  for (int w = 0; w < kWritesPerStream; ++w) {
    for (int si = 0; si < kNumStreams; ++si) {
      // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
      stressWriteKernel<<<1, 1, 0, streams[si]>>>(
          handle, 1, static_cast<uint32_t>(si * 10000 + w));
    }
  }

  for (auto& s : streams) {
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
  }

  uint64_t badTimestamps = 0;
  auto result = reader.poll(streams[0], [&](const auto& e, uint64_t) {
    if (e.timestamp == 0) {
      ++badTimestamps;
    }
  });
  ASSERT_EQ(result.error, cudaSuccess);
  EXPECT_EQ(badTimestamps, 0u);

  uint64_t totalSlots = static_cast<uint64_t>(kNumStreams) * kWritesPerStream;
  EXPECT_EQ(result.entriesRead + result.entriesLost, totalSlots);

  for (auto& s : streams) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaStreamDestroy(s);
  }
}
