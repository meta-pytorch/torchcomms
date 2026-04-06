// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "comms/utils/HRDWRingBuffer.h"
#include "comms/utils/HRDWRingBufferReader.h"

using meta::comms::colltrace::HRDWRingBuffer;
using meta::comms::colltrace::HRDWRingBufferReader;

using TestBuffer = HRDWRingBuffer<void*>;
using TestReader = HRDWRingBufferReader<void*>;
using TestEntry = TestBuffer::Entry;

// Test accessor — friend of HRDWRingBuffer, provides access to internals
// for CPU-side simulation of GPU writes.
namespace meta::comms::colltrace {
class HRDWRingBufferTestAccessor {
 public:
  static HRDWEntry* ring(const TestBuffer& buf) {
    return buf.ring_;
  }
  static uint32_t mask(const TestBuffer& buf) {
    return buf.mask_;
  }
  static uint64_t* writeIndex(const TestBuffer& buf) {
    return buf.writeIndex_;
  }
};
} // namespace meta::comms::colltrace
using TestAccess = meta::comms::colltrace::HRDWRingBufferTestAccessor;

// Helper to store/retrieve a uint32_t tag in the void* data field.
static void* tagToData(uint32_t tag) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<void*>(static_cast<uintptr_t>(tag));
}

static uint32_t dataToTag(void* data) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(data));
}

TEST(HRDWRingBuffer, ConstructionAndAccessors) {
  TestBuffer buf(64);
  ASSERT_TRUE(buf.valid());
  EXPECT_EQ(buf.size(), 64u);
  EXPECT_NE(TestAccess::ring(buf), nullptr);
}

TEST(HRDWRingBuffer, InitialEntriesMarkedSlotEmpty) {
  TestBuffer buf(16);
  ASSERT_TRUE(buf.valid());
  for (uint32_t i = 0; i < 16; ++i) {
    EXPECT_EQ(TestAccess::ring(buf)[i].sequence, HRDW_RINGBUFFER_SLOT_EMPTY);
  }
}

TEST(HRDWRingBuffer, MoveConstruction) {
  TestBuffer buf(32);
  ASSERT_TRUE(buf.valid());

  TestBuffer moved(std::move(buf));
  EXPECT_TRUE(moved.valid());
  EXPECT_EQ(moved.size(), 32u);

  // Moved-from should be invalid.
  EXPECT_FALSE(buf.valid()); // NOLINT(bugprone-use-after-move)
}

TEST(HRDWRingBuffer, MoveAssignment) {
  TestBuffer buf1(32);
  TestBuffer buf2(64);
  ASSERT_TRUE(buf1.valid());
  ASSERT_TRUE(buf2.valid());

  buf1 = std::move(buf2);
  EXPECT_TRUE(buf1.valid());
  EXPECT_EQ(buf1.size(), 64u);
  EXPECT_FALSE(buf2.valid()); // NOLINT(bugprone-use-after-move)
}

TEST(HRDWRingBuffer, RoundsUpZeroSize) {
  TestBuffer buf(0);
  EXPECT_TRUE(buf.valid());
  EXPECT_EQ(buf.size(), 1u);
}

TEST(HRDWRingBuffer, RoundsUpNonPowerOfTwo) {
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
  }

  // Simulate a GPU writing a complete entry at the next slot.
  void writeEntry(uint32_t tag, uint64_t timestampNs = 100) {
    uint64_t slot = (*TestAccess::writeIndex(*buf_))++;
    uint64_t idx = slot & TestAccess::mask(*buf_);
    auto& entry = TestAccess::ring(*buf_)[idx];
    entry.timestamp_ns = timestampNs;
    entry.data = tagToData(tag);
    entry.sequence = slot; // Mark complete
  }

  // Advance writeIndex but leave the entry unstamped (sequence mismatch).
  // Simulates a GPU write that hasn't completed yet.
  void writeUnstampedSlot() {
    uint64_t slot = (*TestAccess::writeIndex(*buf_))++;
    uint64_t idx = slot & TestAccess::mask(*buf_);
    auto& entry = TestAccess::ring(*buf_)[idx];
    entry.timestamp_ns = 0;
    entry.data = nullptr;
    // Don't stamp sequence — it still has SLOT_EMPTY or a stale value.
    // The reader will see preSeq != slot and count it as lost.
    (void)slot;
    (void)entry;
  }

  std::optional<TestBuffer> buf_;
  std::optional<TestReader> reader_;
};

TEST_F(HRDWRingBufferReaderTest, EmptyBufferReturnsNothing) {
  std::vector<uint32_t> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { seen.push_back(dataToTag(e.data)); });
  EXPECT_EQ(result.entriesRead, 0u);
  EXPECT_EQ(result.entriesLost, 0u);

  EXPECT_TRUE(seen.empty());
}

TEST_F(HRDWRingBufferReaderTest, ReadsSingleEntry) {
  writeEntry(42, 1000);

  std::vector<uint32_t> seen;
  auto result = reader_->poll([&](const TestEntry& e, uint64_t) {
    seen.push_back(dataToTag(e.data));
    EXPECT_EQ(e.timestamp_ns, 1000u);
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
      [&](const TestEntry& e, uint64_t) { seen.push_back(dataToTag(e.data)); });

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
      [&](const TestEntry& e, uint64_t) { seen.push_back(dataToTag(e.data)); });

  EXPECT_EQ(result.entriesRead, 1u);
  const std::vector<uint32_t> expected{3};
  EXPECT_EQ(seen, expected);
}

TEST_F(HRDWRingBufferReaderTest, UnstampedEntryStopsScanning) {
  writeEntry(1);
  writeUnstampedSlot(); // writeIndex advanced but sequence not stamped
  writeEntry(3);

  std::vector<uint32_t> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { seen.push_back(dataToTag(e.data)); });

  // Reader stops at unstamped entry (kNotReady). Only entry 1 delivered.
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
      [&](const TestEntry& e, uint64_t) { seen.push_back(dataToTag(e.data)); });

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
  writeEntry(42, 1000);

  uint64_t observedTimestamp = 0;
  auto result = reader_->poll([&](const TestEntry& e, uint64_t) {
    observedTimestamp = e.timestamp_ns;
  });

  EXPECT_EQ(result.entriesRead, 1u);
  EXPECT_EQ(observedTimestamp, 1000u);

  // Mutate the ring entry after poll completed. The callback already ran
  // with a snapshot, so this mutation should not affect the observed value.
  TestAccess::ring(*buf_)[0].timestamp_ns = 9999;
  EXPECT_EQ(observedTimestamp, 1000u);
}

// Verify that overwritten entries (where sequence is a different valid
// slot from a later wrap-around) are counted as lost and never delivered.
TEST_F(HRDWRingBufferReaderTest, OverwrittenEntriesNeverDelivered) {
  // Write kRingSize entries to fill the buffer.
  for (uint32_t i = 0; i < kRingSize; ++i) {
    writeEntry(i);
  }
  // Read them all.
  reader_->poll([](const TestEntry&, uint64_t) {});

  // Write one more entry at slot kRingSize (ring index 0). Its sequence
  // is kRingSize, which won't match the reader's expected slot.
  writeEntry(100);

  // Manually set sequence to a different valid slot to simulate an
  // overwrite from a later wrap-around.
  TestAccess::ring(*buf_)[0].sequence = kRingSize + kRingSize; // wrong epoch

  uint32_t callbackCount = 0;
  auto result =
      reader_->poll([&](const TestEntry&, uint64_t) { ++callbackCount; });

  // The entry should be skipped (sequence mismatch), counted as lost.
  EXPECT_EQ(callbackCount, 0u);
  EXPECT_EQ(result.entriesRead, 0u);
  EXPECT_EQ(result.entriesLost, 1u);
}

// Verify that data pointers and timestamps are correctly preserved.
TEST_F(HRDWRingBufferReaderTest, DataAndTimestampsPreserved) {
  writeEntry(10, 1000);
  writeEntry(10, 2000);

  struct SeenEntry {
    uint32_t tag;
    uint64_t timestamp_ns;
  };
  std::vector<SeenEntry> seen;
  auto result = reader_->poll([&](const TestEntry& e, uint64_t) {
    seen.push_back({dataToTag(e.data), e.timestamp_ns});
  });

  ASSERT_EQ(seen.size(), 2u);
  EXPECT_EQ(result.entriesRead, 2u);

  EXPECT_EQ(seen[0].tag, 10u);
  EXPECT_EQ(seen[0].timestamp_ns, 1000u);

  EXPECT_EQ(seen[1].tag, 10u);
  EXPECT_EQ(seen[1].timestamp_ns, 2000u);
}

// Verify that unstamped entries stop scanning, and subsequent polls
// pick up the entry once it's stamped.
TEST_F(HRDWRingBufferReaderTest, UnstampedEntryResumesAfterStamp) {
  writeEntry(/*tag=*/1, /*timestampNs=*/100);
  writeUnstampedSlot();
  writeEntry(/*tag=*/3, /*timestampNs=*/300);

  // First poll: stops at unstamped entry.
  std::vector<uint32_t> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { seen.push_back(dataToTag(e.data)); });

  const std::vector<uint32_t> expected{1};
  EXPECT_EQ(seen, expected);
  EXPECT_EQ(result.entriesRead, 1u);
  EXPECT_EQ(reader_->lastReadIndex(), 1u);

  // Stamp the unstamped entry.
  TestAccess::ring(*buf_)[1].timestamp_ns = 200;
  TestAccess::ring(*buf_)[1].data = tagToData(2);
  TestAccess::ring(*buf_)[1].sequence = 1;

  // Second poll: picks up entries 2 and 3.
  std::vector<uint32_t> seen2;
  auto result2 = reader_->poll([&](const TestEntry& e, uint64_t) {
    seen2.push_back(dataToTag(e.data));
  });

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
      [&](const TestEntry& e, uint64_t) { seen.push_back(dataToTag(e.data)); },
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
      [&](const TestEntry& e, uint64_t) { seen.push_back(dataToTag(e.data)); },
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
      [&](const TestEntry& e, uint64_t) { seen.push_back(dataToTag(e.data)); });

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
      [&](const TestEntry& e, uint64_t) { seen.push_back(dataToTag(e.data)); });

  EXPECT_EQ(result.entriesLost, 3u);
  EXPECT_EQ(result.entriesRead, kRingSize);
  EXPECT_EQ(
      result.entriesRead + result.entriesLost,
      static_cast<uint64_t>(kRingSize + 3));
}
