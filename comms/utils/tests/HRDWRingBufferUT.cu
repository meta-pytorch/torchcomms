// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "comms/utils/HRDWRingBuffer.h"
#include "comms/utils/HRDWRingBufferReader.h"
#include "comms/utils/tests/HRDWRingBufferTestTypes.h"

using meta::comms::colltrace::HRDWRingBuffer;
using meta::comms::colltrace::HRDWRingBufferReader;

using TestBuffer = HRDWRingBuffer<TestEvent>;
using TestReader = HRDWRingBufferReader<TestEvent>;
using TestEntry = TestBuffer::Entry;

// Test accessor — friend of HRDWRingBuffer, provides access to internals
// for CPU-side simulation of GPU writes.
namespace meta::comms::colltrace {
template <typename DataT>
class HRDWRingBufferTestAccessor {
 public:
  static HRDWEntry<DataT>* ring(const HRDWRingBuffer<DataT>& buf) {
    return buf.ring_;
  }
  static uint32_t mask(const HRDWRingBuffer<DataT>& buf) {
    return buf.mask_;
  }
  static uint32_t shift(const HRDWRingBuffer<DataT>& buf) {
    return buf.shift_;
  }
  static uint64_t* writeIndex(const HRDWRingBuffer<DataT>& buf) {
    return buf.writeIndex_;
  }
};
} // namespace meta::comms::colltrace
using TestAccess =
    meta::comms::colltrace::HRDWRingBufferTestAccessor<TestEvent>;

class HRDWRingBufferTest : public ::testing::Test {};

TEST_F(HRDWRingBufferTest, ConstructionAndAccessors) {
  TestBuffer buf(64);
  ASSERT_TRUE(buf.valid());
  EXPECT_EQ(buf.size(), 64u);
  EXPECT_NE(TestAccess::ring(buf), nullptr);
}

TEST_F(HRDWRingBufferTest, InitialEntriesMarkedSlotEmpty) {
  TestBuffer buf(16);
  ASSERT_TRUE(buf.valid());
  for (uint32_t i = 0; i < 16; ++i) {
    EXPECT_EQ(TestAccess::ring(buf)[i].epoch, HRDW_RINGBUFFER_SLOT_EMPTY);
  }
}

TEST_F(HRDWRingBufferTest, MoveConstruction) {
  TestBuffer buf(32);
  ASSERT_TRUE(buf.valid());

  TestBuffer moved(std::move(buf));
  EXPECT_TRUE(moved.valid());
  EXPECT_EQ(moved.size(), 32u);

  // Moved-from should be invalid.
  EXPECT_FALSE(buf.valid()); // NOLINT(bugprone-use-after-move)
}

TEST_F(HRDWRingBufferTest, MoveAssignment) {
  TestBuffer buf1(32);
  TestBuffer buf2(64);
  ASSERT_TRUE(buf1.valid());
  ASSERT_TRUE(buf2.valid());

  buf1 = std::move(buf2);
  EXPECT_TRUE(buf1.valid());
  EXPECT_EQ(buf1.size(), 64u);
  EXPECT_FALSE(buf2.valid()); // NOLINT(bugprone-use-after-move)
}

TEST_F(HRDWRingBufferTest, RoundsUpZeroSize) {
  TestBuffer buf(0);
  EXPECT_TRUE(buf.valid());
  EXPECT_EQ(buf.size(), 1u);
}

TEST_F(HRDWRingBufferTest, RoundsUpNonPowerOfTwo) {
  TestBuffer buf(10);
  EXPECT_TRUE(buf.valid());
  EXPECT_EQ(buf.size(), 16u);

  TestBuffer buf2(7);
  EXPECT_TRUE(buf2.valid());
  EXPECT_EQ(buf2.size(), 8u);

  // Already a power of 2 — no rounding.
  TestBuffer buf3(8);
  EXPECT_TRUE(buf3.valid());
  EXPECT_EQ(buf3.size(), 8u);
}

class HRDWRingBufferReaderTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kRingSize = 16;

  void SetUp() override {
    buf_.emplace(kRingSize);
    ASSERT_TRUE(buf_->valid());
    reader_.emplace(*buf_);
    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  }

  void TearDown() override {
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  void writeEntry(uint32_t tag) {
    buf_->write(stream_, TestEvent{tag});
    cudaStreamSynchronize(stream_);
  }

  // Advance writeIndex without publishing the slot bytes — simulates the
  // race where the host sees writeIndex bumped before the writer's
  // atom.exch.b128 lands. The slot's epoch stays at EMPTY (0), so the
  // reader classifies it as kNotReady and stops scanning.
  void writeUnstampedSlot() {
    (void)(*TestAccess::writeIndex(*buf_))++;
  }

  std::optional<TestBuffer> buf_;
  std::optional<TestReader> reader_;
  cudaStream_t stream_{nullptr};
};

TEST_F(HRDWRingBufferReaderTest, EmptyBufferReturnsNothing) {
  std::vector<uint32_t> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { seen.push_back(e.data.tag); });
  EXPECT_EQ(result.entriesRead, 0u);
  EXPECT_EQ(result.entriesLost, 0u);

  EXPECT_TRUE(seen.empty());
}

TEST_F(HRDWRingBufferReaderTest, ReadsSingleEntry) {
  writeEntry(42);

  std::vector<uint32_t> seen;
  auto result = reader_->poll([&](const TestEntry& e, uint64_t) {
    seen.push_back(e.data.tag);
    EXPECT_NE(e.timestamp, 0u) << "GPU timestamp should be non-zero";
  });

  EXPECT_EQ(result.entriesRead, 1u);
  EXPECT_EQ(result.entriesLost, 0u);

  const std::vector<uint32_t> expected{42};
  EXPECT_EQ(seen, expected);
}

TEST_F(HRDWRingBufferReaderTest, ReadsMultipleEntries) {
  writeEntry(1);
  writeEntry(2);
  writeEntry(3);

  std::vector<uint32_t> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { seen.push_back(e.data.tag); });

  EXPECT_EQ(result.entriesRead, 3u);
  const std::vector<uint32_t> expected{1, 2, 3};
  EXPECT_EQ(seen, expected);
}

TEST_F(HRDWRingBufferReaderTest, DoesNotReReadOldEntries) {
  writeEntry(1);
  writeEntry(2);

  // First poll reads entries 1 and 2.
  reader_->poll([](const TestEntry&, uint64_t) {});

  // Write a third entry.
  writeEntry(3);

  // Second poll should only see entry 3.
  std::vector<uint32_t> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { seen.push_back(e.data.tag); });

  EXPECT_EQ(result.entriesRead, 1u);
  const std::vector<uint32_t> expected{3};
  EXPECT_EQ(seen, expected);
}

// Models the race where writeIndex becomes host-visible before the
// writer's slot publication: the reader sees an EMPTY epoch at the
// half-published slot and stops scanning.
TEST_F(HRDWRingBufferReaderTest, UnstampedEntryStopsScanning) {
  writeEntry(1);
  writeUnstampedSlot(); // writeIndex advanced but slot not yet published
  writeEntry(3);

  std::vector<uint32_t> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { seen.push_back(e.data.tag); });

  // Reader stops at the unstamped slot (kNotReady). Only entry 1 delivered.
  EXPECT_EQ(result.entriesRead, 1u);
  EXPECT_EQ(result.entriesLost, 0u);
  const std::vector<uint32_t> expected{1};
  EXPECT_EQ(seen, expected);
}

TEST_F(HRDWRingBufferReaderTest, WrapAroundReadsCorrectly) {
  // Fill the entire ring.
  for (uint32_t i = 0; i < kRingSize; ++i) {
    writeEntry(i);
  }
  reader_->poll([](const TestEntry&, uint64_t) {});

  // Write a few more — these wrap around to the beginning of the ring.
  writeEntry(100);
  writeEntry(101);

  std::vector<uint32_t> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { seen.push_back(e.data.tag); });

  EXPECT_EQ(result.entriesRead, 2u);
  const std::vector<uint32_t> expected{100, 101};
  EXPECT_EQ(seen, expected);
}

TEST_F(HRDWRingBufferReaderTest, CallbackReceivesCorrectSlot) {
  writeEntry(10);
  writeEntry(20);

  std::vector<uint64_t> slots;
  reader_->poll(
      [&](const TestEntry&, uint64_t slot) { slots.push_back(slot); });

  const std::vector<uint64_t> expected{0, 1};
  EXPECT_EQ(slots, expected);
}

TEST_F(HRDWRingBufferReaderTest, MultiplePollCyclesAccumulate) {
  uint64_t totalRead = 0;

  for (int cycle = 0; cycle < 5; ++cycle) {
    writeEntry(cycle);
    writeEntry(cycle + 100);
    auto result = reader_->poll([](const TestEntry&, uint64_t) {});
    totalRead += result.entriesRead;
  }

  EXPECT_EQ(totalRead, 10u);
  EXPECT_EQ(reader_->lastReadIndex(), 10u);
}

// Verify that callbacks receive copies of entries, not references to shared
// memory. Mutate the ring entry after poll() captures it but before we
// inspect what the callback received — the callback should see the original.
TEST_F(HRDWRingBufferReaderTest, CallbackReceivesSnapshotNotReference) {
  writeEntry(42);

  uint32_t observedTimestamp = 0;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { observedTimestamp = e.timestamp; });

  EXPECT_EQ(result.entriesRead, 1u);
  EXPECT_NE(observedTimestamp, 0u);

  // Mutate the ring entry after poll completed. The callback already ran
  // with a snapshot, so this mutation should not affect the observed value.
  TestAccess::ring(*buf_)[0].timestamp = 9999;
  EXPECT_NE(observedTimestamp, 9999u);
}

// Verify that overwritten entries (epoch advanced past the reader's
// expected value) are counted as lost and never delivered.
TEST_F(HRDWRingBufferReaderTest, OverwrittenEntriesNeverDelivered) {
  // Write kRingSize entries to fill the buffer.
  for (uint32_t i = 0; i < kRingSize; ++i) {
    writeEntry(i);
  }
  // Read them all.
  reader_->poll([](const TestEntry&, uint64_t) {});

  // Write one more entry at slot kRingSize (ring index 0). Its epoch
  // advances past the reader's last seen value.
  writeEntry(100);

  // Manually bump epoch to simulate a further wrap-around overwrite.
  TestAccess::ring(*buf_)[0].epoch =
      static_cast<uint32_t>((2 * kRingSize) >> TestAccess::shift(*buf_)) + 1u;

  uint32_t callbackCount = 0;
  auto result =
      reader_->poll([&](const TestEntry&, uint64_t) { ++callbackCount; });

  // The entry should be skipped (epoch mismatch), counted as lost.
  EXPECT_EQ(callbackCount, 0u);
  EXPECT_EQ(result.entriesRead, 0u);
  EXPECT_EQ(result.entriesLost, 1u);
}

// Verify that data and timestamps are correctly preserved.
TEST_F(HRDWRingBufferReaderTest, DataAndTimestampsPreserved) {
  writeEntry(10);
  writeEntry(20);

  struct SeenEntry {
    uint32_t tag;
    uint32_t timestamp;
  };
  std::vector<SeenEntry> seen;
  auto result = reader_->poll([&](const TestEntry& e, uint64_t) {
    seen.push_back({e.data.tag, e.timestamp});
  });

  ASSERT_EQ(seen.size(), 2u);
  EXPECT_EQ(result.entriesRead, 2u);

  EXPECT_EQ(seen[0].tag, 10u);
  EXPECT_NE(seen[0].timestamp, 0u);

  EXPECT_EQ(seen[1].tag, 20u);
  EXPECT_NE(seen[1].timestamp, 0u);
  // Second write happened after first, so timestamp should be >= first.
  EXPECT_GE(seen[1].timestamp, seen[0].timestamp);
}

// Same race, then the writer's publication finally lands — the next poll
// should resume from where the previous one stopped.
TEST_F(HRDWRingBufferReaderTest, UnstampedEntryResumesAfterStamp) {
  writeEntry(1);
  writeUnstampedSlot();
  writeEntry(3);

  // First poll: stops at unstamped slot.
  std::vector<uint32_t> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { seen.push_back(e.data.tag); });

  const std::vector<uint32_t> expected{1};
  EXPECT_EQ(seen, expected);
  EXPECT_EQ(result.entriesRead, 1u);
  EXPECT_EQ(reader_->lastReadIndex(), 1u);

  // Manually publish slot 1 with the correct epoch.
  TestAccess::ring(*buf_)[1].timestamp = 200;
  TestAccess::ring(*buf_)[1].data = TestEvent{2};
  TestAccess::ring(*buf_)[1].epoch =
      static_cast<uint32_t>(1 >> TestAccess::shift(*buf_)) + 1u;

  // Second poll: picks up slots 1 and 2 (entries 2 and 3).
  std::vector<uint32_t> seen2;
  auto result2 = reader_->poll(
      [&](const TestEntry& e, uint64_t) { seen2.push_back(e.data.tag); });

  const std::vector<uint32_t> expected2{2, 3};
  EXPECT_EQ(seen2, expected2);
  EXPECT_EQ(result2.entriesRead, 2u);
  EXPECT_EQ(reader_->lastReadIndex(), 3u);
}

TEST_F(HRDWRingBufferReaderTest, TimeoutReturnsImmediatelyWhenNoEntries) {
  auto result = reader_->poll(
      [](const TestEntry&, uint64_t) {}, std::chrono::milliseconds{50});
  EXPECT_EQ(result.entriesRead, 0u);
  EXPECT_EQ(result.entriesLost, 0u);
  EXPECT_FALSE(result.timedOut);
}

TEST_F(HRDWRingBufferReaderTest, ZeroTimeoutReturnsImmediatelyWhenEmpty) {
  auto result = reader_->poll(
      [](const TestEntry&, uint64_t) {}, std::chrono::milliseconds{0});
  EXPECT_EQ(result.entriesRead, 0u);
  EXPECT_FALSE(result.timedOut);
}

TEST_F(HRDWRingBufferReaderTest, ZeroTimeoutStillReadsAvailableEntries) {
  writeEntry(1);
  writeEntry(2);
  writeEntry(3);

  std::vector<uint32_t> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { seen.push_back(e.data.tag); },
      std::chrono::milliseconds{0});

  // Zero timeout should still deliver all available entries.
  EXPECT_EQ(result.entriesRead, 3u);
  const std::vector<uint32_t> expected{1, 2, 3};
  EXPECT_EQ(seen, expected);
}

TEST_F(HRDWRingBufferReaderTest, TimeoutBoundsOverwrittenProcessing) {
  // Fill the ring completely, then overwrite all entries multiple times.
  // This simulates the reader being heavily lapped.
  for (uint32_t lap = 0; lap < 10; ++lap) {
    for (uint32_t i = 0; i < kRingSize; ++i) {
      writeEntry(i + lap * kRingSize);
    }
  }

  std::vector<uint32_t> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { seen.push_back(e.data.tag); },
      std::chrono::milliseconds{50});

  // 10 laps * 16 entries = 160 total. Reader jumps to tail (160 - 16 = 144).
  // 144 entries lost, 16 entries read.
  constexpr uint64_t kTotalEntries = 10 * kRingSize;
  EXPECT_EQ(result.entriesLost, kTotalEntries - kRingSize);
  EXPECT_EQ(result.entriesRead, kRingSize);
  EXPECT_EQ(result.entriesRead + result.entriesLost, kTotalEntries);
}

TEST_F(HRDWRingBufferReaderTest, OnePastLapJumpsToTailLosesOne) {
  // Write ringSize + 1 entries. Reader should jump to tail, losing 1.
  for (uint32_t i = 0; i < kRingSize + 1; ++i) {
    writeEntry(i);
  }

  std::vector<uint32_t> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { seen.push_back(e.data.tag); });

  EXPECT_EQ(result.entriesLost, 1u);
  EXPECT_EQ(result.entriesRead, kRingSize);
  EXPECT_EQ(result.entriesRead + result.entriesLost, kRingSize + 1);
}

TEST_F(HRDWRingBufferReaderTest, JumpToTailAfterPartialRead) {
  // Write some entries, poll to read them, then write enough to lap.
  for (uint32_t i = 0; i < 4; ++i) {
    writeEntry(i);
  }
  reader_->poll([](const TestEntry&, uint64_t) {});
  EXPECT_EQ(reader_->lastReadIndex(), 4u);

  // Now write ringSize + 3 more entries — reader is lapped by 3.
  for (uint32_t i = 0; i < kRingSize + 3; ++i) {
    writeEntry(100 + i);
  }

  std::vector<uint32_t> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { seen.push_back(e.data.tag); });

  EXPECT_EQ(result.entriesLost, 3u);
  EXPECT_EQ(result.entriesRead, kRingSize);
  EXPECT_EQ(
      result.entriesRead + result.entriesLost,
      static_cast<uint64_t>(kRingSize + 3));
}

// ---------------------------------------------------------------------------
// Destruction: verify ~HRDWRingBuffer destructs live entries for non-trivially-
// destructible DataT. Uses a mapped counter accessible from both GPU (kernel
// writes) and CPU (ring buffer teardown destructor).
// ---------------------------------------------------------------------------

// 8-byte event that tracks live instance count via a mapped counter.
struct CountedEvent {
  int* counter{nullptr};

  __host__ __device__ CountedEvent() = default;

  __host__ __device__ explicit CountedEvent(int* counter_) : counter(counter_) {
    if (counter) {
#ifdef __CUDA_ARCH__
      atomicAdd(counter, 1);
#else
      ++(*counter);
#endif
    }
  }

  __host__ __device__ ~CountedEvent() {
    if (counter) {
#ifdef __CUDA_ARCH__
      atomicAdd(counter, -1);
#else
      --(*counter);
#endif
    }
  }

  __host__ __device__ CountedEvent(const CountedEvent& o) : counter(o.counter) {
    if (counter) {
#ifdef __CUDA_ARCH__
      atomicAdd(counter, 1);
#else
      ++(*counter);
#endif
    }
  }

  __host__ __device__ CountedEvent(CountedEvent&& o) noexcept
      : counter(o.counter) {
    // Transfer ownership — no increment. Moved-from object won't decrement.
    o.counter = nullptr;
  }

  __host__ __device__ CountedEvent& operator=(const CountedEvent& o) {
    if (this != &o) {
      if (counter) {
#ifdef __CUDA_ARCH__
        atomicAdd(counter, -1);
#else
        --(*counter);
#endif
      }
      counter = o.counter;
      if (counter) {
#ifdef __CUDA_ARCH__
        atomicAdd(counter, 1);
#else
        ++(*counter);
#endif
      }
    }
    return *this;
  }

  __host__ __device__ CountedEvent& operator=(CountedEvent&& o) noexcept {
    if (this != &o) {
      if (counter) {
#ifdef __CUDA_ARCH__
        atomicAdd(counter, -1);
#else
        --(*counter);
#endif
      }
      counter = o.counter;
      o.counter = nullptr;
    }
    return *this;
  }
};

TEST(HRDWRingBufferDestructionTest, TeardownDestructsAllEntries) {
  int* counter = nullptr;
  ASSERT_EQ(
      cudaHostAlloc(&counter, sizeof(int), cudaHostAllocMapped), cudaSuccess);
  *counter = 0;

  int beforeTeardown;
  {
    HRDWRingBuffer<CountedEvent> buf(4);
    ASSERT_TRUE(buf.valid());

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    for (int i = 0; i < 4; ++i) {
      buf.write(stream, CountedEvent(counter));
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaStreamDestroy(stream);

    beforeTeardown = *counter;
  }
  // Teardown should have decremented by exactly 4 (one per ring entry).
  EXPECT_EQ(beforeTeardown - *counter, 4)
      << "Teardown should destruct exactly 4 entries";
  cudaFreeHost(counter);
}

TEST(HRDWRingBufferDestructionTest, PartialFillOnlyDestructsWritten) {
  int* counter = nullptr;
  ASSERT_EQ(
      cudaHostAlloc(&counter, sizeof(int), cudaHostAllocMapped), cudaSuccess);
  *counter = 0;

  int beforeTeardown;
  {
    HRDWRingBuffer<CountedEvent> buf(8);
    ASSERT_TRUE(buf.valid());

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    for (int i = 0; i < 3; ++i) {
      buf.write(stream, CountedEvent(counter));
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaStreamDestroy(stream);

    beforeTeardown = *counter;
  }
  // Only 3 written entries should be destructed, not all 8 slots.
  EXPECT_EQ(beforeTeardown - *counter, 3)
      << "Teardown should destruct only the 3 written entries";
  cudaFreeHost(counter);
}

// ---------------------------------------------------------------------------
// Explicit instantiation for CountedEvent so buf.write() works.
// ---------------------------------------------------------------------------

namespace meta::comms::colltrace {

namespace {
template <typename DataT>
__global__ void countedEventRingBufferWriteKernel(
    HRDWEntry<DataT>* ring,
    uint64_t* writeIdx,
    uint32_t mask,
    uint32_t shift,
    DataT data) {
  hrdwRingBufferWrite(ring, writeIdx, mask, shift, data);
}
} // namespace

template <>
cudaError_t launchRingBufferWrite<CountedEvent>(
    cudaStream_t stream,
    HRDWEntry<CountedEvent>* ring,
    uint64_t* writeIdx,
    uint32_t mask,
    uint32_t shift,
    CountedEvent data) {
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  countedEventRingBufferWriteKernel<<<1, 1, 0, stream>>>(
      ring, writeIdx, mask, shift, data);
  return cudaGetLastError();
}

} // namespace meta::comms::colltrace
