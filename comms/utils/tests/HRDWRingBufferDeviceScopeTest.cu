// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <cstdint>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBufferDeviceHandle.cuh"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBufferReader.h"

using hrdw_ring_buffer::HRDWRingBuffer;
using hrdw_ring_buffer::HRDWRingBufferDeviceHandle;
using hrdw_ring_buffer::HRDWRingBufferReader;
using hrdw_ring_buffer::MemoryCoherenceScope;

struct DeviceTestEvent {
  uint32_t tag;
};

using DeviceBuffer =
    HRDWRingBuffer<DeviceTestEvent, MemoryCoherenceScope::Device>;
using DeviceReader =
    HRDWRingBufferReader<DeviceTestEvent, MemoryCoherenceScope::Device>;
using DeviceHandle =
    HRDWRingBufferDeviceHandle<DeviceTestEvent, MemoryCoherenceScope::Device>;

__global__ void deviceScopeWriteKernel(DeviceHandle rb, int count) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < count) {
    rb.write(DeviceTestEvent{static_cast<uint32_t>(tid)});
  }
}

class HRDWRingBufferDeviceScopeTest : public ::testing::Test {
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

TEST_F(HRDWRingBufferDeviceScopeTest, ConstructionAndValid) {
  DeviceBuffer buf(64);
  ASSERT_TRUE(buf.valid());
  EXPECT_EQ(buf.size(), 64u);
}

TEST_F(HRDWRingBufferDeviceScopeTest, MoveConstruction) {
  DeviceBuffer buf(32);
  ASSERT_TRUE(buf.valid());

  DeviceBuffer moved(std::move(buf));
  EXPECT_TRUE(moved.valid());
  EXPECT_EQ(moved.size(), 32u);
  EXPECT_FALSE(buf.valid()); // NOLINT(bugprone-use-after-move)
}

TEST_F(HRDWRingBufferDeviceScopeTest, MoveAssignment) {
  DeviceBuffer buf1(32);
  DeviceBuffer buf2(64);
  ASSERT_TRUE(buf1.valid());
  ASSERT_TRUE(buf2.valid());

  buf1 = std::move(buf2);
  EXPECT_TRUE(buf1.valid());
  EXPECT_EQ(buf1.size(), 64u);
  EXPECT_FALSE(buf2.valid()); // NOLINT(bugprone-use-after-move)
}

TEST_F(HRDWRingBufferDeviceScopeTest, SingleWriteAndDrain) {
  DeviceBuffer buf(16);
  ASSERT_TRUE(buf.valid());
  DeviceReader reader(buf);

  ASSERT_EQ(buf.write(stream_, DeviceTestEvent{42}), cudaSuccess);

  std::vector<uint32_t> seen;
  auto result = reader.poll(stream_, [&](const auto& entry, uint64_t) {
    seen.push_back(entry.data.tag);
  });
  ASSERT_EQ(result.error, cudaSuccess);

  EXPECT_EQ(result.entriesRead, 1u);
  EXPECT_EQ(result.entriesLost, 0u);

  const std::vector<uint32_t> expected{42};
  EXPECT_EQ(seen, expected);
}

TEST_F(HRDWRingBufferDeviceScopeTest, MultipleWritesAndDrain) {
  DeviceBuffer buf(64);
  ASSERT_TRUE(buf.valid());
  DeviceReader reader(buf);

  for (uint32_t i = 0; i < 10; ++i) {
    ASSERT_EQ(buf.write(stream_, DeviceTestEvent{i}), cudaSuccess);
  }

  std::vector<uint32_t> seen;
  auto result = reader.poll(stream_, [&](const auto& entry, uint64_t) {
    seen.push_back(entry.data.tag);
  });
  ASSERT_EQ(result.error, cudaSuccess);

  EXPECT_EQ(result.entriesRead, 10u);
  EXPECT_EQ(result.entriesLost, 0u);

  const std::vector<uint32_t> expected{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_EQ(seen, expected);
}

TEST_F(HRDWRingBufferDeviceScopeTest, InlineDeviceWriteAndDrain) {
  constexpr int kCount = 128;
  DeviceBuffer buf(256);
  ASSERT_TRUE(buf.valid());
  DeviceReader reader(buf);

  auto handle = buf.deviceHandle();

  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  deviceScopeWriteKernel<<<1, kCount, 0, stream_>>>(handle, kCount);

  std::vector<uint32_t> tags;
  auto result = reader.poll(stream_, [&](const auto& entry, uint64_t) {
    tags.push_back(entry.data.tag);
  });
  ASSERT_EQ(result.error, cudaSuccess);

  EXPECT_EQ(result.entriesRead, kCount);
  EXPECT_EQ(result.entriesLost, 0u);

  std::sort(tags.begin(), tags.end());
  std::vector<uint32_t> expected(kCount);
  std::iota(expected.begin(), expected.end(), 0u);
  EXPECT_EQ(tags, expected);
}

TEST_F(HRDWRingBufferDeviceScopeTest, MultiBlockConcurrentWriteAndDrain) {
  constexpr int kThreadsPerBlock = 64;
  constexpr int kNumBlocks = 8;
  constexpr int kTotal = kThreadsPerBlock * kNumBlocks;

  DeviceBuffer buf(1024);
  ASSERT_TRUE(buf.valid());
  DeviceReader reader(buf);

  auto handle = buf.deviceHandle();

  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  deviceScopeWriteKernel<<<kNumBlocks, kThreadsPerBlock, 0, stream_>>>(
      handle, kTotal);

  std::vector<uint32_t> tags;
  auto result = reader.poll(stream_, [&](const auto& entry, uint64_t) {
    tags.push_back(entry.data.tag);
  });

  EXPECT_EQ(result.entriesRead, kTotal);
  EXPECT_EQ(result.entriesLost, 0u);

  std::sort(tags.begin(), tags.end());
  std::vector<uint32_t> expected(kTotal);
  std::iota(expected.begin(), expected.end(), 0u);
  EXPECT_EQ(tags, expected);
}

TEST_F(HRDWRingBufferDeviceScopeTest, DrainWithLapping) {
  constexpr uint32_t kRingSize = 16;
  DeviceBuffer buf(kRingSize);
  ASSERT_TRUE(buf.valid());
  DeviceReader reader(buf);

  // Write more than the ring can hold.
  for (uint32_t i = 0; i < kRingSize + 5; ++i) {
    ASSERT_EQ(buf.write(stream_, DeviceTestEvent{i}), cudaSuccess);
  }

  std::vector<uint32_t> seen;
  auto result = reader.poll(stream_, [&](const auto& entry, uint64_t) {
    seen.push_back(entry.data.tag);
  });
  ASSERT_EQ(result.error, cudaSuccess);

  EXPECT_EQ(result.entriesRead, kRingSize);
  EXPECT_EQ(result.entriesLost, 5u);
  EXPECT_EQ(result.entriesRead + result.entriesLost, kRingSize + 5u);
}

TEST_F(HRDWRingBufferDeviceScopeTest, PollAutoAdvancesReadCursor) {
  DeviceBuffer buf(16);
  ASSERT_TRUE(buf.valid());
  DeviceReader reader(buf);

  ASSERT_EQ(buf.write(stream_, DeviceTestEvent{1}), cudaSuccess);
  ASSERT_EQ(buf.write(stream_, DeviceTestEvent{2}), cudaSuccess);

  // First poll reads both initial entries.
  uint64_t firstCount = 0;
  reader.poll(stream_, [&](const auto&, uint64_t) { ++firstCount; });
  EXPECT_EQ(firstCount, 2u);

  // Write more entries — poll() must auto-advance its read cursor so the
  // second poll only sees the newly-written entries, not the original two.
  ASSERT_EQ(buf.write(stream_, DeviceTestEvent{10}), cudaSuccess);
  ASSERT_EQ(buf.write(stream_, DeviceTestEvent{20}), cudaSuccess);
  ASSERT_EQ(buf.write(stream_, DeviceTestEvent{30}), cudaSuccess);

  std::vector<uint32_t> seen;
  auto result = reader.poll(stream_, [&](const auto& entry, uint64_t) {
    seen.push_back(entry.data.tag);
  });

  EXPECT_EQ(result.entriesRead, 3u);
  EXPECT_EQ(result.entriesLost, 0u);

  const std::vector<uint32_t> expected{10, 20, 30};
  EXPECT_EQ(seen, expected);
}

TEST_F(HRDWRingBufferDeviceScopeTest, EmptyDrainReturnsNothing) {
  DeviceBuffer buf(16);
  ASSERT_TRUE(buf.valid());
  DeviceReader reader(buf);

  uint64_t callbackCount = 0;
  auto result =
      reader.poll(stream_, [&](const auto&, uint64_t) { ++callbackCount; });
  ASSERT_EQ(result.error, cudaSuccess);

  EXPECT_EQ(result.entriesRead, 0u);
  EXPECT_EQ(result.entriesLost, 0u);
  EXPECT_EQ(callbackCount, 0u);
}

TEST_F(HRDWRingBufferDeviceScopeTest, MultipleDrainCycles) {
  DeviceBuffer buf(64);
  ASSERT_TRUE(buf.valid());
  DeviceReader reader(buf);

  uint64_t totalRead = 0;

  for (int cycle = 0; cycle < 5; ++cycle) {
    ASSERT_EQ(
        buf.write(stream_, DeviceTestEvent{static_cast<uint32_t>(cycle)}),
        cudaSuccess);
    auto result = reader.poll(stream_, [](const auto&, uint64_t) {});
    totalRead += result.entriesRead;
  }

  EXPECT_EQ(totalRead, 5u);
}

TEST_F(HRDWRingBufferDeviceScopeTest, TimestampsAreNonZero) {
  DeviceBuffer buf(16);
  ASSERT_TRUE(buf.valid());
  DeviceReader reader(buf);

  ASSERT_EQ(buf.write(stream_, DeviceTestEvent{1}), cudaSuccess);

  reader.poll(stream_, [](const auto& entry, uint64_t) {
    EXPECT_NE(entry.timestamp, 0u) << "GPU timestamp should be non-zero";
  });
}

TEST_F(HRDWRingBufferDeviceScopeTest, WrapAroundDrainReadsCorrectly) {
  constexpr uint32_t kRingSize = 8;
  DeviceBuffer buf(kRingSize);
  ASSERT_TRUE(buf.valid());
  DeviceReader reader(buf);

  // Fill ring exactly, drain, then write a few more that wrap.
  for (uint32_t i = 0; i < kRingSize; ++i) {
    ASSERT_EQ(buf.write(stream_, DeviceTestEvent{i}), cudaSuccess);
  }
  reader.poll(stream_, [](const auto&, uint64_t) {});

  ASSERT_EQ(buf.write(stream_, DeviceTestEvent{100}), cudaSuccess);
  ASSERT_EQ(buf.write(stream_, DeviceTestEvent{101}), cudaSuccess);

  std::vector<uint32_t> seen;
  auto result = reader.poll(stream_, [&](const auto& entry, uint64_t) {
    seen.push_back(entry.data.tag);
  });

  EXPECT_EQ(result.entriesRead, 2u);
  EXPECT_EQ(result.entriesLost, 0u);

  const std::vector<uint32_t> expected{100, 101};
  EXPECT_EQ(seen, expected);
}
