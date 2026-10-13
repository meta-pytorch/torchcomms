// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "comms/mccl/tests/MockMcclComm.h"
#include "comms/torchcomms/mccl/McclGraphEventTracker.hpp"
#include "comms/torchcomms/mccl/TorchCommMCCL.hpp"
#include "comms/torchcomms/mccl/tests/unit/cpp/MockCudaApi.h"

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace torch::comms::test {

namespace {
constexpr unsigned long long kGraphId = 42;
const auto kGraph = reinterpret_cast<cudaGraph_t>(0xB000);
const auto kStream = reinterpret_cast<cudaStream_t>(0x5001);
const auto kStartEvent = reinterpret_cast<cudaEvent_t>(0xE001);
const auto kEndEvent = reinterpret_cast<cudaEvent_t>(0xE002);

struct FiredEvent {
  uint64_t replay_id;
  size_t collective_index;
  std::string event;
  bool operator==(const FiredEvent& other) const {
    return replay_id == other.replay_id &&
        collective_index == other.collective_index && event == other.event;
  }
};
} // namespace

// Unit-tests McclGraphEventTracker's per-replay clog event firing on CPU.
//
// The tracker is constructed standalone over a TorchCommMCCL built with the
// (non-init) constructor, so graph_monitor_side_stream_ stays null and
// forkGraphMonitorSideStream() runs the recording callback directly on the
// (mocked) stream — no real CUDA / GPU needed. The replay counter's GPU kernel
// is replaced by a CPU atomic-add (AtomicAddMock.cpp), and its backing memory
// is this fixture's `counter_` so the test can drive the replay number.
class McclGraphEventTrackerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    resetMcclGraphTimeoutMonitoringCacheForTest();
    cuda_api_ = std::make_shared<NiceMock<MockCudaApi>>();
    mock_comm_ = std::make_unique<::mccl::testing::MockMcclComm>();
    mock_comm_ptr_ = mock_comm_.get();

    // DeviceCounter::create() allocates its counter via hostAlloc(); hand it
    // this fixture's `counter_` so the test controls the replay number.
    ON_CALL(*cuda_api_, hostAlloc(_, _, _))
        .WillByDefault(DoAll(
            SetArgPointee<0>(static_cast<void*>(&counter_)),
            Return(cudaSuccess)));
    ON_CALL(*cuda_api_, hostFree(_)).WillByDefault(Return(cudaSuccess));

    // `stream` is actively capturing graph kGraphId.
    ON_CALL(*cuda_api_, streamGetCaptureInfo_v2(_, _, _, _, _, _))
        .WillByDefault(DoAll(
            SetArgPointee<1>(cudaStreamCaptureStatusActive),
            SetArgPointee<2>(kGraphId),
            SetArgPointee<3>(kGraph),
            Return(cudaSuccess)));
    ON_CALL(*cuda_api_, userObjectCreate(_, _, _, _, _))
        .WillByDefault(DoAll(
            SetArgPointee<0>(reinterpret_cast<cudaUserObject_t>(0x3000)),
            Return(cudaSuccess)));
    ON_CALL(*cuda_api_, graphRetainUserObject(_, _, _, _))
        .WillByDefault(Return(cudaSuccess));
    ON_CALL(*cuda_api_, eventRecordWithFlags(_, _, _))
        .WillByDefault(Return(cudaSuccess));
    ON_CALL(*cuda_api_, eventDestroy(_)).WillByDefault(Return(cudaSuccess));

    // Event completion is driven by start_status_ / end_status_.
    ON_CALL(*cuda_api_, eventQuery(kStartEvent))
        .WillByDefault([this](cudaEvent_t) { return start_status_; });
    ON_CALL(*cuda_api_, eventQuery(kEndEvent))
        .WillByDefault([this](cudaEvent_t) { return end_status_; });
  }

  // Builds the comm (consuming mock_comm_) and registers a hook that records
  // every fired (replay_id, collective_index, event) into fired_.
  std::shared_ptr<TorchCommMCCL> makeComm() {
    auto comm = std::make_shared<TorchCommMCCL>(
        std::move(mock_comm_), /*rank=*/0, /*size=*/2, cuda_api_);
    comm->registerGraphReplayHook(
        /*hookId=*/0,
        [this](
            uint64_t graph_id,
            uint64_t replay_id,
            void* /*stream*/,
            size_t collective_index,
            std::string_view event) {
          EXPECT_EQ(graph_id, kGraphId);
          fired_.push_back({replay_id, collective_index, std::string(event)});
        });
    return comm;
  }

  std::shared_ptr<NiceMock<MockCudaApi>> cuda_api_;
  std::unique_ptr<::mccl::testing::MockMcclComm> mock_comm_;
  ::mccl::testing::MockMcclComm* mock_comm_ptr_{};
  uint64_t counter_{0};
  cudaError_t start_status_{cudaErrorNotReady};
  cudaError_t end_status_{cudaErrorNotReady};
  std::vector<FiredEvent> fired_;
};

TEST_F(McclGraphEventTrackerTest, CheckAllWithNoGraphsReturnsOk) {
  auto comm = makeComm();
  McclGraphEventTracker tracker(comm.get());
  EXPECT_EQ(tracker.checkAll(), McclGraphEventTracker::CheckResult::OK);
}

// One captured collective replayed 3 times must produce exactly one S then one
// E per replay, numbered R1..R3.
TEST_F(McclGraphEventTrackerTest, FiresStartThenEndForEachReplay) {
  auto comm = makeComm();
  McclGraphEventTracker tracker(comm.get());
  ASSERT_TRUE(tracker.initOnGraphStart(kStream));
  tracker.addEntry(
      kStream, kStartEvent, kEndEvent, std::chrono::milliseconds(-1));

  for (uint64_t replay = 1; replay <= 3; ++replay) {
    counter_ = replay;
    // Collective in flight: start done, end pending -> S.
    start_status_ = cudaSuccess;
    end_status_ = cudaErrorNotReady;
    EXPECT_EQ(tracker.checkAll(), McclGraphEventTracker::CheckResult::OK);
    // Collective finished: end done -> E.
    end_status_ = cudaSuccess;
    EXPECT_EQ(tracker.checkAll(), McclGraphEventTracker::CheckResult::OK);
  }

  const std::vector<FiredEvent> expected = {
      {1, 0, "S"},
      {1, 0, "E"},
      {2, 0, "S"},
      {2, 0, "E"},
      {3, 0, "S"},
      {3, 0, "E"}};
  EXPECT_EQ(fired_, expected);
}

// Replays the watchdog "missed" (counter jumps past the last-notified replay)
// are reconstructed: a single checkAll() at replay 3 catches up R1 and R2.
TEST_F(McclGraphEventTrackerTest, CatchesUpMissedReplays) {
  auto comm = makeComm();
  McclGraphEventTracker tracker(comm.get());
  ASSERT_TRUE(tracker.initOnGraphStart(kStream));
  tracker.addEntry(
      kStream, kStartEvent, kEndEvent, std::chrono::milliseconds(-1));

  // The watchdog only observes the graph at replay 3, with the collective
  // already complete. It must still emit S/E for replays 1 and 2.
  counter_ = 3;
  start_status_ = cudaSuccess;
  end_status_ = cudaSuccess;
  EXPECT_EQ(tracker.checkAll(), McclGraphEventTracker::CheckResult::OK);

  const std::vector<FiredEvent> expected = {
      {1, 0, "S"},
      {1, 0, "E"},
      {2, 0, "S"},
      {2, 0, "E"},
      {3, 0, "S"},
      {3, 0, "E"}};
  EXPECT_EQ(fired_, expected);
}

// When the comm's abort flag is set, fired events use the aborted variant
// (S!/E!) so the clog shows the collective ran under an aborted comm.
TEST_F(McclGraphEventTrackerTest, AbortedCommMarksReplayEvents) {
  ON_CALL(*mock_comm_ptr_, isAborted()).WillByDefault(Return(true));
  auto comm = makeComm();
  McclGraphEventTracker tracker(comm.get());
  ASSERT_TRUE(tracker.initOnGraphStart(kStream));
  tracker.addEntry(
      kStream, kStartEvent, kEndEvent, std::chrono::milliseconds(-1));

  counter_ = 1;
  start_status_ = cudaSuccess;
  end_status_ = cudaSuccess;
  EXPECT_EQ(tracker.checkAll(), McclGraphEventTracker::CheckResult::OK);

  const std::vector<FiredEvent> expected = {{1, 0, "S!"}, {1, 0, "E!"}};
  EXPECT_EQ(fired_, expected);
}

// A CUDA error from an event query surfaces as CheckResult::ERROR.
TEST_F(McclGraphEventTrackerTest, CheckAllReturnsErrorOnCudaFailure) {
  auto comm = makeComm();
  McclGraphEventTracker tracker(comm.get());
  ASSERT_TRUE(tracker.initOnGraphStart(kStream));
  tracker.addEntry(
      kStream, kStartEvent, kEndEvent, std::chrono::milliseconds(-1));

  counter_ = 1;
  start_status_ = cudaErrorIllegalAddress;
  end_status_ = cudaErrorIllegalAddress;
  EXPECT_EQ(tracker.checkAll(), McclGraphEventTracker::CheckResult::ERROR);
}

} // namespace torch::comms::test
