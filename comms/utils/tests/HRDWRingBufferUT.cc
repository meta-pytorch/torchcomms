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
  EXPECT_EQ(buf.mask(), 63u);
  EXPECT_NE(buf.ring(), nullptr);
  EXPECT_NE(buf.writeIndex(), nullptr);
  EXPECT_EQ(*buf.writeIndex(), 0u);
}

TEST(HRDWRingBuffer, InitialEntriesMarkedWritePending) {
  TestBuffer buf(16);
  ASSERT_TRUE(buf.valid());
  for (uint32_t i = 0; i < 16; ++i) {
    EXPECT_EQ(buf.ring()[i].sequence, HRDW_RINGBUFFER_WRITE_PENDING);
  }
}

TEST(HRDWRingBuffer, MoveConstruction) {
  TestBuffer buf(32);
  ASSERT_TRUE(buf.valid());
  auto* ring = buf.ring();
  auto* writeIdx = buf.writeIndex();

  TestBuffer moved(std::move(buf));
  EXPECT_TRUE(moved.valid());
  EXPECT_EQ(moved.ring(), ring);
  EXPECT_EQ(moved.writeIndex(), writeIdx);
  EXPECT_EQ(moved.size(), 32u);

  // Moved-from should be invalid.
  EXPECT_FALSE(buf.valid()); // NOLINT(bugprone-use-after-move)
}

TEST(HRDWRingBuffer, MoveAssignment) {
  TestBuffer buf1(32);
  TestBuffer buf2(64);
  ASSERT_TRUE(buf1.valid());
  ASSERT_TRUE(buf2.valid());
  auto* ring2 = buf2.ring();

  buf1 = std::move(buf2);
  EXPECT_TRUE(buf1.valid());
  EXPECT_EQ(buf1.ring(), ring2);
  EXPECT_EQ(buf1.size(), 64u);
  EXPECT_FALSE(buf2.valid()); // NOLINT(bugprone-use-after-move)
}

TEST(HRDWRingBuffer, RoundsUpZeroSize) {
  TestBuffer buf(0);
  EXPECT_TRUE(buf.valid());
  EXPECT_EQ(buf.size(), 1u);
  EXPECT_EQ(buf.mask(), 0u);
}

TEST(HRDWRingBuffer, RoundsUpNonPowerOfTwo) {
  TestBuffer buf(10);
  EXPECT_TRUE(buf.valid());
  EXPECT_EQ(buf.size(), 16u);
  EXPECT_EQ(buf.mask(), 15u);

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
  void writeEntry(uint32_t tag, uint64_t startNs = 100, uint64_t endNs = 200) {
    uint64_t slot = (*buf_->writeIndex())++;
    uint64_t idx = slot & buf_->mask();
    auto& entry = buf_->ring()[idx];
    entry.start_ns = startNs;
    entry.end_ns = endNs;
    entry.data = tagToData(tag);
    entry.sequence = slot; // Mark complete
  }

  // Simulate a GPU writing a write-pending entry (start kernel ran, end
  // kernel hasn't stamped yet).
  void writeIncompleteEntry(uint32_t tag, uint64_t startNs = 100) {
    uint64_t slot = (*buf_->writeIndex())++;
    uint64_t idx = slot & buf_->mask();
    auto& entry = buf_->ring()[idx];
    entry.start_ns = startNs;
    entry.end_ns = 0;
    entry.data = tagToData(tag);
    entry.sequence = HRDW_RINGBUFFER_WRITE_PENDING;
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
  EXPECT_FALSE(result.tornReadDetected);
  EXPECT_TRUE(seen.empty());
}

TEST_F(HRDWRingBufferReaderTest, ReadsSingleEntry) {
  writeEntry(42, 1000, 2000);

  std::vector<uint32_t> seen;
  auto result = reader_->poll([&](const TestEntry& e, uint64_t) {
    seen.push_back(dataToTag(e.data));
    EXPECT_EQ(e.start_ns, 1000u);
    EXPECT_EQ(e.end_ns, 2000u);
  });

  EXPECT_EQ(result.entriesRead, 1u);
  EXPECT_EQ(result.entriesLost, 0u);
  EXPECT_FALSE(result.tornReadDetected);
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

TEST_F(HRDWRingBufferReaderTest, StopsAtWritePendingEntry) {
  writeEntry(1);
  writeIncompleteEntry(
      2); // This entry has sequence = HRDW_RINGBUFFER_WRITE_PENDING
  writeEntry(3);

  std::vector<uint32_t> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { seen.push_back(dataToTag(e.data)); });

  // Reader stops at entry 2 (in-flight). Only entry 1 is delivered.
  // Entries 2 and 3 are deferred (entriesInFlight=2).
  EXPECT_EQ(result.entriesRead, 1u);
  EXPECT_EQ(result.entriesLost, 0u);
  EXPECT_EQ(result.entriesInFlight, 2u);
  const std::vector<uint32_t> expected{1};
  EXPECT_EQ(seen, expected);
  EXPECT_EQ(reader_->lastReadIndex(), 1u);

  // Complete entry 2 — next poll picks up entries 2 and 3.
  buf_->ring()[1].sequence = 1;
  std::vector<uint32_t> seen2;
  auto result2 = reader_->poll([&](const TestEntry& e, uint64_t) {
    seen2.push_back(dataToTag(e.data));
  });

  EXPECT_EQ(result2.entriesRead, 2u);
  EXPECT_EQ(result2.entriesLost, 0u);
  EXPECT_EQ(result2.entriesInFlight, 0u);
  const std::vector<uint32_t> expected2{2, 3};
  EXPECT_EQ(seen2, expected2);
}

TEST_F(HRDWRingBufferReaderTest, DetectsLostEntriesWhenFarBehind) {
  // Write more entries than the ring can hold, so the reader falls behind.
  for (uint32_t i = 0; i < kRingSize + 5; ++i) {
    writeEntry(i);
  }

  std::vector<uint32_t> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { seen.push_back(dataToTag(e.data)); });

  // The first 5 entries were overwritten before we could read them.
  EXPECT_EQ(result.entriesLost, 5u);
  // We should read the remaining kRingSize entries.
  EXPECT_EQ(result.entriesRead, kRingSize);
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
  writeEntry(42, 1000, 2000);

  uint64_t observedStartNs = 0;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { observedStartNs = e.start_ns; });

  EXPECT_EQ(result.entriesRead, 1u);
  EXPECT_EQ(observedStartNs, 1000u);

  // Mutate the ring entry after poll completed. The callback already ran
  // with a snapshot, so this mutation should not affect the observed value.
  buf_->ring()[0].start_ns = 9999;
  EXPECT_EQ(observedStartNs, 1000u);
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
  buf_->ring()[0].sequence = kRingSize + kRingSize; // wrong epoch

  uint32_t callbackCount = 0;
  auto result =
      reader_->poll([&](const TestEntry&, uint64_t) { ++callbackCount; });

  // The entry should be skipped (sequence mismatch), counted as lost.
  EXPECT_EQ(callbackCount, 0u);
  EXPECT_EQ(result.entriesRead, 0u);
  EXPECT_EQ(result.entriesLost, 1u);
}

// Verify that write-pending entries are delivered through the callback
// with sequence == HRDW_RINGBUFFER_WRITE_PENDING when scanIncomplete=true,
// allowing consumers to identify exactly which collectives are in-flight.
TEST_F(HRDWRingBufferReaderTest, WritePendingEntriesDeliveredViaCallback) {
  // Write one complete entry, then two write-pending entries with distinct
  // collId tags and start timestamps.
  writeEntry(/*tag=*/10, /*startNs=*/1000, /*endNs=*/2000);
  writeIncompleteEntry(/*tag=*/20, /*startNs=*/3000);
  writeIncompleteEntry(/*tag=*/30, /*startNs=*/4000);

  struct SeenEntry {
    uint32_t tag;
    uint64_t start_ns;
    uint64_t slot;
    bool isPending;
  };
  std::vector<SeenEntry> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t slot) {
        seen.push_back({
            dataToTag(e.data),
            e.start_ns,
            slot,
            e.sequence == HRDW_RINGBUFFER_WRITE_PENDING,
        });
      },
      /*scanIncomplete=*/true);

  // 3 entries delivered: 1 complete + 2 write-pending.
  ASSERT_EQ(seen.size(), 3u);
  EXPECT_EQ(result.entriesRead, 3u);
  EXPECT_EQ(result.entriesInFlight, 2u);
  EXPECT_EQ(reader_->lastReadIndex(), 1u);

  // First entry: complete.
  EXPECT_EQ(seen[0].tag, 10u);
  EXPECT_FALSE(seen[0].isPending);

  // Second and third: write-pending with correct start_ns and slot.
  EXPECT_EQ(seen[1].tag, 20u);
  EXPECT_TRUE(seen[1].isPending);
  EXPECT_EQ(seen[1].start_ns, 3000u);
  EXPECT_EQ(seen[1].slot, 1u);

  EXPECT_EQ(seen[2].tag, 30u);
  EXPECT_TRUE(seen[2].isPending);
  EXPECT_EQ(seen[2].start_ns, 4000u);
  EXPECT_EQ(seen[2].slot, 2u);

  // Complete entry at slot 1 — next poll delivers slot 1 as complete,
  // slot 2 still as write-pending.
  buf_->ring()[1].end_ns = 3500;
  buf_->ring()[1].sequence = 1;

  std::vector<SeenEntry> seen2;
  auto result2 = reader_->poll(
      [&](const TestEntry& e, uint64_t slot) {
        seen2.push_back({
            dataToTag(e.data),
            e.start_ns,
            slot,
            e.sequence == HRDW_RINGBUFFER_WRITE_PENDING,
        });
      },
      /*scanIncomplete=*/true);

  ASSERT_EQ(seen2.size(), 2u);
  EXPECT_EQ(seen2[0].tag, 20u);
  EXPECT_FALSE(seen2[0].isPending);
  EXPECT_EQ(seen2[1].tag, 30u);
  EXPECT_TRUE(seen2[1].isPending);
  EXPECT_EQ(result2.entriesInFlight, 1u);

  // Complete the last entry — everything clears.
  buf_->ring()[2].end_ns = 5000;
  buf_->ring()[2].sequence = 2;

  std::vector<SeenEntry> seen3;
  auto result3 = reader_->poll(
      [&](const TestEntry& e, uint64_t slot) {
        seen3.push_back({
            dataToTag(e.data),
            e.start_ns,
            slot,
            e.sequence == HRDW_RINGBUFFER_WRITE_PENDING,
        });
      },
      /*scanIncomplete=*/true);
  ASSERT_EQ(seen3.size(), 1u);
  EXPECT_EQ(seen3[0].tag, 30u);
  EXPECT_FALSE(seen3[0].isPending);
  EXPECT_EQ(result3.entriesInFlight, 0u);
  EXPECT_EQ(reader_->lastReadIndex(), 3u);
}

// Verify that when all entries are write-pending, they're all delivered
// through the callback with the pending sentinel.
TEST_F(HRDWRingBufferReaderTest, AllEntriesPendingDeliveredViaCallback) {
  writeIncompleteEntry(/*tag=*/5, /*startNs=*/500);
  writeIncompleteEntry(/*tag=*/6, /*startNs=*/600);
  writeIncompleteEntry(/*tag=*/7, /*startNs=*/700);

  struct SeenEntry {
    uint32_t tag;
    uint64_t start_ns;
    bool isPending;
  };
  std::vector<SeenEntry> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) {
        seen.push_back({
            dataToTag(e.data),
            e.start_ns,
            e.sequence == HRDW_RINGBUFFER_WRITE_PENDING,
        });
      },
      /*scanIncomplete=*/true);

  // All three delivered as write-pending.
  ASSERT_EQ(seen.size(), 3u);
  EXPECT_EQ(result.entriesRead, 3u);
  EXPECT_EQ(result.entriesInFlight, 3u);
  EXPECT_EQ(reader_->lastReadIndex(), 0u);

  EXPECT_EQ(seen[0].tag, 5u);
  EXPECT_EQ(seen[0].start_ns, 500u);
  EXPECT_TRUE(seen[0].isPending);
  EXPECT_EQ(seen[1].tag, 6u);
  EXPECT_EQ(seen[1].start_ns, 600u);
  EXPECT_TRUE(seen[1].isPending);
  EXPECT_EQ(seen[2].tag, 7u);
  EXPECT_EQ(seen[2].start_ns, 700u);
  EXPECT_TRUE(seen[2].isPending);
}

// Verify that the default scanIncomplete=false still breaks at the first
// WRITE_PENDING and does NOT deliver pending entries via the callback.
TEST_F(HRDWRingBufferReaderTest, DefaultPollDoesNotScanIncomplete) {
  writeEntry(/*tag=*/1, /*startNs=*/100, /*endNs=*/200);
  writeIncompleteEntry(/*tag=*/2, /*startNs=*/300);
  writeIncompleteEntry(/*tag=*/3, /*startNs=*/400);

  std::vector<uint32_t> seen;
  auto result = reader_->poll(
      [&](const TestEntry& e, uint64_t) { seen.push_back(dataToTag(e.data)); });

  // Completed entry delivered, reader stopped at first WRITE_PENDING.
  const std::vector<uint32_t> expectedSeen{1};
  EXPECT_EQ(seen, expectedSeen);
  EXPECT_EQ(result.entriesRead, 1u);
  EXPECT_EQ(result.entriesInFlight, 2u);
}

// ---------------------------------------------------------------------------
// Integration tests: simulate the pollGraphEvents() consumer pattern.
//
// These tests exercise the full reader → collId dispatch pipeline that
// pollGraphEvents() uses: the callback checks entry.sequence to distinguish
// complete vs. write-pending entries and dispatches per-collective actions.
// ---------------------------------------------------------------------------

namespace {

// Minimal stand-in for GraphCollectiveEntry used in pollGraphEvents().
struct FakeCollEntry {
  uint32_t collId{};
  bool progressingFired{false};
  uint64_t detectedStartNs{0};
};

} // namespace

// Simulate pollGraphEvents() with 4 collectives: 2 complete, 2 in-flight.
// Verify that only the in-flight collectives get progressing actions, and
// that the completed collectives are delivered normally.
TEST_F(HRDWRingBufferReaderTest, PerCollectiveDispatchMixedState) {
  writeEntry(/*tag=*/10, /*startNs=*/1000, /*endNs=*/1500);
  writeEntry(/*tag=*/20, /*startNs=*/2000, /*endNs=*/2500);
  writeIncompleteEntry(/*tag=*/30, /*startNs=*/3000);
  writeIncompleteEntry(/*tag=*/40, /*startNs=*/4000);

  FakeCollEntry coll10{10}, coll20{20}, coll30{30}, coll40{40};
  std::unordered_map<uint32_t, FakeCollEntry*> collIdMap{
      {10, &coll10}, {20, &coll20}, {30, &coll30}, {40, &coll40}};

  std::vector<uint32_t> completedCollIds;
  std::vector<uint32_t> progressingCollIds;
  reader_->poll(
      [&](const TestEntry& e, uint64_t) {
        auto collId = dataToTag(e.data);
        auto it = collIdMap.find(collId);
        if (it == collIdMap.end()) {
          return;
        }
        if (e.sequence == HRDW_RINGBUFFER_WRITE_PENDING) {
          it->second->progressingFired = true;
          it->second->detectedStartNs = e.start_ns;
          progressingCollIds.push_back(collId);
        } else {
          completedCollIds.push_back(collId);
        }
      },
      /*scanIncomplete=*/true);

  const std::vector<uint32_t> expectedCompleted{10, 20};
  EXPECT_EQ(completedCollIds, expectedCompleted);

  const std::vector<uint32_t> expectedProgressing{30, 40};
  EXPECT_EQ(progressingCollIds, expectedProgressing);

  EXPECT_FALSE(coll10.progressingFired);
  EXPECT_FALSE(coll20.progressingFired);
  EXPECT_TRUE(coll30.progressingFired);
  EXPECT_EQ(coll30.detectedStartNs, 3000u);
  EXPECT_TRUE(coll40.progressingFired);
  EXPECT_EQ(coll40.detectedStartNs, 4000u);
}

// Verify that a write-pending entry that completes between polls is
// delivered as a normal completed entry on the next poll.
TEST_F(HRDWRingBufferReaderTest, PendingEntryCompletedBetweenPolls) {
  writeIncompleteEntry(/*tag=*/50, /*startNs=*/9000);

  // Poll 1 — entry delivered as write-pending.
  bool sawPending = false;
  reader_->poll(
      [&](const TestEntry& e, uint64_t) {
        if (e.sequence == HRDW_RINGBUFFER_WRITE_PENDING) {
          EXPECT_EQ(dataToTag(e.data), 50u);
          sawPending = true;
        }
      },
      /*scanIncomplete=*/true);
  EXPECT_TRUE(sawPending);

  // Complete the entry.
  buf_->ring()[0].end_ns = 9500;
  buf_->ring()[0].sequence = 0;

  // Poll 2 — delivered as completed.
  std::vector<uint32_t> completed;
  auto result2 = reader_->poll(
      [&](const TestEntry& e, uint64_t) {
        EXPECT_NE(e.sequence, HRDW_RINGBUFFER_WRITE_PENDING);
        completed.push_back(dataToTag(e.data));
      },
      /*scanIncomplete=*/true);

  const std::vector<uint32_t> expectedCompleted{50};
  EXPECT_EQ(completed, expectedCompleted);
  EXPECT_EQ(result2.entriesInFlight, 0u);
}
