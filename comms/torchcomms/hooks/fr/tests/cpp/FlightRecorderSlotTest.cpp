// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comms/torchcomms/hooks/fr/FlightRecorder.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace torch::comms::fr {
namespace {

// Op ids are allocated globally before the pre-hook records them, so the
// recorder can see them out of order (two comms on two threads) or with gaps
// (a comm without the FR hook). Placement and lookup must not depend on them.
class FlightRecorderSlotTest : public ::testing::Test {
 protected:
  static void record(FlightRecorder& fr, size_t op_id) {
    fr.record(
        /*pg_id=*/0,
        {"pg", "desc"},
        op_id,
        "all_reduce",
        /*inputs=*/{},
        /*outputs=*/{},
        /*start=*/nullptr,
        /*end=*/nullptr,
        std::chrono::milliseconds(1000),
        std::make_shared<ProcessGroupStatus>());
  }

  static void retire(FlightRecorder& fr, size_t op_id) {
    fr.retire_id(op_id, at::Device(at::kCPU), /*compute_duration=*/false);
  }

  static std::vector<size_t> opIds(FlightRecorder& fr) {
    std::vector<size_t> ids;
    for (const auto& e : fr.dump_entries()) {
      ids.push_back(e.op_id_);
    }
    return ids;
  }

  static bool isRetired(FlightRecorder& fr, size_t op_id) {
    for (const auto& e : fr.dump_entries()) {
      if (e.op_id_ == op_id) {
        return e.retired_;
      }
    }
    return false;
  }
};

TEST_F(FlightRecorderSlotTest, OutOfOrderOpIdsAreRecordedInArrivalOrder) {
  FlightRecorder fr(/*max_entries=*/8, /*capture_cpp_stack=*/false);
  record(fr, 1); // e.g. a Gloo op that allocated its id later but records first
  record(fr, 0);
  EXPECT_EQ(opIds(fr), (std::vector<size_t>{1, 0}));
}

TEST_F(FlightRecorderSlotTest, GapsInOpIdsAreTolerated) {
  FlightRecorder fr(8, false);
  for (size_t op_id : {0, 2, 3, 7}) {
    record(fr, op_id);
  }
  EXPECT_EQ(opIds(fr), (std::vector<size_t>{0, 2, 3, 7}));
}

TEST_F(FlightRecorderSlotTest, RetireFindsEntriesRecordedOutOfOrder) {
  FlightRecorder fr(8, false);
  record(fr, 1);
  record(fr, 0);
  retire(fr, 0);
  EXPECT_TRUE(isRetired(fr, 0));
  EXPECT_FALSE(isRetired(fr, 1));
  retire(fr, 1);
  EXPECT_TRUE(isRetired(fr, 1));
}

TEST_F(FlightRecorderSlotTest, WrapAroundKeepsLatestEntriesInArrivalOrder) {
  FlightRecorder fr(4, false);
  for (size_t op_id : {5, 3, 4, 0, 9, 1}) {
    record(fr, op_id);
  }
  EXPECT_EQ(opIds(fr), (std::vector<size_t>{4, 0, 9, 1}));
  retire(fr, 9);
  EXPECT_TRUE(isRetired(fr, 9));
  retire(fr, 5); // overwritten by the wrap: must not mark another entry
  for (size_t op_id : {4, 0, 1}) {
    EXPECT_FALSE(isRetired(fr, op_id));
  }
}

TEST_F(FlightRecorderSlotTest, ResetMidFillStartsNewEpochAtNextSlot) {
  FlightRecorder fr(4, false);
  record(fr, 1);
  record(fr, 0);
  fr.reset_all();
  record(fr, 6);
  record(fr, 4);
  EXPECT_EQ(opIds(fr), (std::vector<size_t>{6, 4}));
  retire(fr, 4);
  EXPECT_TRUE(isRetired(fr, 4));
  record(fr, 5); // buffer full: overwrites the old epoch's slots
  record(fr, 7);
  EXPECT_EQ(opIds(fr), (std::vector<size_t>{6, 4, 5, 7}));
}

TEST_F(FlightRecorderSlotTest, ConcurrentRecordWhereLaterOpIdWinsTheRace) {
  // Two threads: op_id 0 is allocated first but its pre-hook is delayed until
  // op_id 1 has been recorded from the other thread.
  FlightRecorder fr(16, false);
  std::mutex mutex;
  std::condition_variable cv;
  bool higherRecorded = false;

  std::thread lower([&] {
    std::unique_lock lock(mutex);
    cv.wait(lock, [&] { return higherRecorded; });
    lock.unlock();
    EXPECT_NO_THROW(record(fr, 0));
  });
  std::thread higher([&] {
    EXPECT_NO_THROW(record(fr, 1));
    {
      std::lock_guard lock(mutex);
      higherRecorded = true;
    }
    cv.notify_all();
  });
  higher.join();
  lower.join();

  EXPECT_EQ(opIds(fr), (std::vector<size_t>{1, 0}));
  retire(fr, 0);
  retire(fr, 1);
  EXPECT_TRUE(isRetired(fr, 0));
  EXPECT_TRUE(isRetired(fr, 1));
}

} // namespace
} // namespace torch::comms::fr
