// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/tests/unit/cpp/TorchCommNCCLXTestBase.hpp"

#include <algorithm>
#include <vector>

namespace torch::comms::test {

class GraphEventTrackerTest : public TorchCommNCCLXTest {
 protected:
  struct GraphEvents {
    cudaEvent_t start = reinterpret_cast<cudaEvent_t>(0xA001);
    cudaEvent_t end = reinterpret_cast<cudaEvent_t>(0xA002);
    cudaEvent_t sync = reinterpret_cast<cudaEvent_t>(0xA003);
  };

  CommOptions createAbortModeOptions(
      std::chrono::milliseconds timeout = std::chrono::milliseconds(2000)) {
    CommOptions options;
    options.abort_process_on_timeout_or_error = true;
    options.timeout = timeout;
    options.store = store_;
    return options;
  }

  void setupGraphCaptureMocks(
      unsigned long long graph_id = 42,
      cudaGraph_t graph = reinterpret_cast<cudaGraph_t>(0xB000)) {
    ON_CALL(*cuda_mock_, streamIsCapturing(_, _))
        .WillByDefault(DoAll(
            SetArgPointee<1>(cudaStreamCaptureStatusActive),
            Return(cudaSuccess)));
    ON_CALL(*cuda_mock_, streamGetCaptureInfo_v2(_, _, _, _, _, _))
        .WillByDefault(DoAll(
            SetArgPointee<1>(cudaStreamCaptureStatusActive),
            SetArgPointee<2>(graph_id),
            SetArgPointee<3>(graph),
            Return(cudaSuccess)));
    ON_CALL(*cuda_mock_, userObjectCreate(_, _, _, _, _))
        .WillByDefault(DoAll(
            SetArgPointee<0>(reinterpret_cast<cudaUserObject_t>(0x3000)),
            Return(cudaSuccess)));
    ON_CALL(*cuda_mock_, graphRetainUserObject(_, _, _, _))
        .WillByDefault(Return(cudaSuccess));
  }

  GraphEvents setupGraphCaptureEvents() {
    GraphEvents events;
    EXPECT_CALL(*cuda_mock_, eventCreateWithFlags(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(events.start), Return(cudaSuccess)))
        .WillOnce(DoAll(SetArgPointee<0>(events.end), Return(cudaSuccess)))
        .WillOnce(DoAll(SetArgPointee<0>(events.sync), Return(cudaSuccess)))
        .WillRepeatedly(DoAll(
            SetArgPointee<0>(reinterpret_cast<cudaEvent_t>(0xA100)),
            Return(cudaSuccess)));
    return events;
  }

  void setupEventRecordMocks() {
    ON_CALL(*cuda_mock_, eventRecord(_, _)).WillByDefault(Return(cudaSuccess));
    ON_CALL(*cuda_mock_, eventRecordWithFlags(_, _, _))
        .WillByDefault(Return(cudaSuccess));
  }

  void switchToReplayMode() {
    ON_CALL(*cuda_mock_, streamIsCapturing(_, _))
        .WillByDefault(DoAll(
            SetArgPointee<1>(cudaStreamCaptureStatusNone),
            Return(cudaSuccess)));
  }

  void setupFinalizeExpectations(TestTorchCommNCCLX& comm) {
    EXPECT_CALL(*cuda_mock_, eventDestroy(_))
        .WillRepeatedly(Return(cudaSuccess));
    EXPECT_CALL(*cuda_mock_, free(_)).WillRepeatedly(Return(cudaSuccess));
    EXPECT_CALL(*cuda_mock_, streamDestroy(_)).WillOnce(Return(cudaSuccess));
    EXPECT_CALL(*nccl_mock_, commDestroy(_)).WillOnce(Return(ncclSuccess));
    comm.finalize();
  }
};

TEST_F(GraphEventTrackerTest, GraphTimeoutCausesProcessDeath) {
  EXPECT_DEATH(
      {
        cuda_mock_->setupDefaultBehaviors();
        nccl_mock_->setupDefaultBehaviors();

        auto options = createAbortModeOptions(std::chrono::milliseconds(100));
        auto comm = createMockedTorchComm();
        comm->init(*device_, "test_graph_timeout", options);

        setupGraphCaptureMocks();
        auto events = setupGraphCaptureEvents();
        setupEventRecordMocks();

        auto tensor = createTestTensor({10, 10});
        auto work = comm->send(tensor, 1, true);

        switchToReplayMode();

        ON_CALL(*cuda_mock_, eventQuery(events.start))
            .WillByDefault(Return(cudaSuccess));
        ON_CALL(*cuda_mock_, eventQuery(events.end))
            .WillByDefault(Return(cudaErrorNotReady));

        std::this_thread::sleep_for(std::chrono::seconds(5));
      },
      "Graph monitor: collective TIMED OUT for graph");
}

TEST_F(
    GraphEventTrackerTest,
    GraphTimeoutAfterSuccessfulReplayCausesProcessDeath) {
  // After a first replay completes, a second that hangs is still detected
  // as a timeout.  The sequence:
  //   1. end_event NOT REACHED  (first poll — replay in progress)
  //   2. end_event COMPLETED    (first replay finishes — timer resets)
  //   3. end_event NOT REACHED  (second replay hangs)
  //   ... timer counts from step 3 and eventually fires.
  // start_event is always COMPLETED because the hang is *after* the
  // collective starts.
  EXPECT_DEATH(
      {
        cuda_mock_->setupDefaultBehaviors();
        nccl_mock_->setupDefaultBehaviors();

        auto options = createAbortModeOptions(std::chrono::milliseconds(100));
        auto comm = createMockedTorchComm();
        comm->init(*device_, "test_graph_timeout_after_success", options);

        setupGraphCaptureMocks();
        auto events = setupGraphCaptureEvents();
        setupEventRecordMocks();

        auto tensor = createTestTensor({10, 10});
        auto work = comm->send(tensor, 1, true);

        switchToReplayMode();

        EXPECT_CALL(*cuda_mock_, eventQuery(events.end))
            .WillOnce(Return(cudaErrorNotReady))
            .WillOnce(Return(cudaSuccess))
            .WillRepeatedly(Return(cudaErrorNotReady));

        EXPECT_CALL(*cuda_mock_, eventQuery(events.start))
            .WillRepeatedly(Return(cudaSuccess));

        std::this_thread::sleep_for(std::chrono::seconds(5));
      },
      "Graph monitor: collective TIMED OUT for graph");
}

TEST_F(GraphEventTrackerTest, GraphCaptureWorkObjectsDestroyedAfterCapture) {
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions();
  comm->init(*device_, "test_graph_work_destroyed", options);

  setupGraphCaptureMocks();
  auto events = setupGraphCaptureEvents();

  auto tensor = createTestTensor({10, 10});

  EXPECT_CALL(*cuda_mock_, eventDestroy(events.sync))
      .WillOnce(Return(cudaSuccess));

  {
    auto work = comm->send(tensor, 1, true);
  }

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  switchToReplayMode();
  setupFinalizeExpectations(*comm);
}

TEST_F(GraphEventTrackerTest, GraphDestroyCleanupDestroysMonitorEvents) {
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions();
  comm->init(*device_, "test_graph_destroy_cleanup", options);

  setupGraphCaptureMocks();
  setupGraphCaptureEvents();

  void* captured_cleanup_data = nullptr;
  cudaHostFn_t captured_cleanup_fn = nullptr;
  // Override userObjectCreate to capture the cleanup callback
  EXPECT_CALL(*cuda_mock_, userObjectCreate(_, _, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaUserObject_t>(0x3000)),
          SaveArg<1>(&captured_cleanup_data),
          SaveArg<2>(&captured_cleanup_fn),
          Return(cudaSuccess)));

  auto tensor = createTestTensor({10, 10});

  {
    auto work = comm->send(tensor, 1, true);
  }

  ASSERT_NE(captured_cleanup_fn, nullptr);
  ASSERT_NE(captured_cleanup_data, nullptr);

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  // Invoke cleanup callback — should only set released flag, NO eventDestroy
  EXPECT_CALL(*cuda_mock_, eventDestroy(_)).Times(0);
  captured_cleanup_fn(captured_cleanup_data);
  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  // Events destroyed during finalize (which calls destroyAll)
  switchToReplayMode();
  setupFinalizeExpectations(*comm);
}

TEST_F(GraphEventTrackerTest, CheckAllReturnsOKWhenNoEntries) {
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions();
  comm->init(*device_, "test_empty_graph_check", options);

  auto tensor = createTestTensor({10, 10});

  setupEventsForWork(*comm, 1);
  auto work = comm->send(tensor, 1, true);

  // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  WorkEvent& we = work_events_[0];
  ON_CALL(*cuda_mock_, eventQuery(we.start_event))
      .WillByDefault(Return(cudaSuccess));
  ON_CALL(*cuda_mock_, eventQuery(we.end_event))
      .WillByDefault(Return(cudaSuccess));

  setupFinalizeExpectations(*comm);
}

TEST_F(GraphEventTrackerTest, CheckAllReturnsErrorOnCudaFailure) {
  EXPECT_DEATH(
      {
        cuda_mock_->setupDefaultBehaviors();
        nccl_mock_->setupDefaultBehaviors();

        auto options = createAbortModeOptions();
        auto comm = createMockedTorchComm();
        comm->init(*device_, "test_graph_cuda_error", options);

        setupGraphCaptureMocks();
        auto events = setupGraphCaptureEvents();
        setupEventRecordMocks();

        auto tensor = createTestTensor({10, 10});
        auto work = comm->send(tensor, 1, true);

        switchToReplayMode();

        ON_CALL(*cuda_mock_, eventQuery(events.start))
            .WillByDefault(Return(cudaErrorInvalidValue));

        std::this_thread::sleep_for(std::chrono::seconds(5));
      },
      ".*");
}

TEST_F(GraphEventTrackerTest, ReplayCounterResetsTimer) {
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions(std::chrono::milliseconds(200));
  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_replay_counter_reset", options);

  setupGraphCaptureMocks();
  auto events = setupGraphCaptureEvents();
  setupEventRecordMocks();

  // Capture the replay counter pointer from hostAlloc.
  // DeviceCounter::create calls api->hostAlloc(sizeof(uint64_t)).
  uint64_t* captured_counter = nullptr;
  ON_CALL(*cuda_mock_, hostAlloc(_, _, _))
      .WillByDefault(
          [&captured_counter](void** ptr, size_t size, unsigned int) {
            *ptr = std::calloc(1, size);
            if (size == sizeof(uint64_t)) {
              captured_counter = static_cast<uint64_t*>(*ptr);
            }
            return cudaSuccess;
          });

  auto tensor = createTestTensor({10, 10});
  auto work = comm->send(tensor, 1, true);

  switchToReplayMode();

  ON_CALL(*cuda_mock_, eventQuery(events.start))
      .WillByDefault(Return(cudaSuccess));
  ON_CALL(*cuda_mock_, eventQuery(events.end))
      .WillByDefault(Return(cudaErrorNotReady));

  ASSERT_NE(captured_counter, nullptr);

  std::atomic<bool> stop_replays{false};
  std::thread replay_thread([&]() {
    while (!stop_replays.load()) {
      // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (!stop_replays.load()) {
        // Simulate GPU kernel incrementing the counter on replay.
        // In the mock, mapped memory is plain host memory, so a direct
        // increment is equivalent to what launchAtomicAdd does.
        ++(*captured_counter);
      }
    }
  });

  // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
  std::this_thread::sleep_for(std::chrono::milliseconds(800));

  stop_replays.store(true);
  replay_thread.join();

  ON_CALL(*cuda_mock_, eventQuery(events.end))
      .WillByDefault(Return(cudaSuccess));

  // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());
  setupFinalizeExpectations(*comm);
}

// destroyAll should stash the entry-owned events into the tracker event pool;
// the actual cudaEventDestroy calls fire at ~GraphEventTracker (comm
// destruction). We track destroys to verify the events are eventually freed.
TEST_F(GraphEventTrackerTest, DestroyAllCleansUpGraphEntryEvents) {
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions();
  comm->init(*device_, "test_destroy_all_cleanup", options);

  setupGraphCaptureMocks();
  auto events = setupGraphCaptureEvents();

  auto tensor = createTestTensor({10, 10});

  EXPECT_CALL(*cuda_mock_, eventDestroy(events.sync))
      .WillOnce(Return(cudaSuccess));

  {
    auto work = comm->send(tensor, 1, true);
  }

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  switchToReplayMode();

  std::vector<cudaEvent_t> destroyed_events;
  EXPECT_CALL(*cuda_mock_, eventDestroy(_))
      .WillRepeatedly(DoAll(
          [&destroyed_events](cudaEvent_t event) {
            destroyed_events.push_back(event);
          },
          Return(cudaSuccess)));
  EXPECT_CALL(*cuda_mock_, free(_)).WillRepeatedly(Return(cudaSuccess));
  EXPECT_CALL(*cuda_mock_, streamDestroy(_)).WillOnce(Return(cudaSuccess));
  EXPECT_CALL(*nccl_mock_, commDestroy(_)).WillOnce(Return(ncclSuccess));

  comm->finalize();

  // After finalize, events are stashed in the tracker pool, not destroyed.
  EXPECT_EQ(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.start),
      destroyed_events.end())
      << "start event must defer destruction past finalize";
  EXPECT_EQ(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.end),
      destroyed_events.end())
      << "end event must defer destruction past finalize";

  // ~GraphEventTracker drains the pool at comm destruction.
  comm.reset();
  EXPECT_NE(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.start),
      destroyed_events.end())
      << "start event was not destroyed by ~GraphEventTracker";
  EXPECT_NE(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.end),
      destroyed_events.end())
      << "end event was not destroyed by ~GraphEventTracker";
}

// Verify that the cleanup callback only sets the released flag when a graph
// contains multiple captured collectives, without destroying events directly.
TEST_F(GraphEventTrackerTest, MultipleCollectivesInSameGraphCleanedUp) {
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions();
  comm->init(*device_, "test_multi_collective_cleanup", options);

  setupGraphCaptureMocks();

  // Two collectives: each creates 3 events (start, end, sync)
  cudaEvent_t start1 = reinterpret_cast<cudaEvent_t>(0xA001);
  cudaEvent_t end1 = reinterpret_cast<cudaEvent_t>(0xA002);
  cudaEvent_t sync1 = reinterpret_cast<cudaEvent_t>(0xA003);
  cudaEvent_t start2 = reinterpret_cast<cudaEvent_t>(0xA004);
  cudaEvent_t end2 = reinterpret_cast<cudaEvent_t>(0xA005);
  cudaEvent_t sync2 = reinterpret_cast<cudaEvent_t>(0xA006);

  EXPECT_CALL(*cuda_mock_, eventCreateWithFlags(_, _))
      .WillOnce(DoAll(SetArgPointee<0>(start1), Return(cudaSuccess)))
      .WillOnce(DoAll(SetArgPointee<0>(end1), Return(cudaSuccess)))
      .WillOnce(DoAll(SetArgPointee<0>(sync1), Return(cudaSuccess)))
      .WillOnce(DoAll(SetArgPointee<0>(start2), Return(cudaSuccess)))
      .WillOnce(DoAll(SetArgPointee<0>(end2), Return(cudaSuccess)))
      .WillOnce(DoAll(SetArgPointee<0>(sync2), Return(cudaSuccess)))
      .WillRepeatedly(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaEvent_t>(0xA100)),
          Return(cudaSuccess)));

  void* captured_cleanup_data = nullptr;
  cudaHostFn_t captured_cleanup_fn = nullptr;
  // Capture the cleanup callback
  EXPECT_CALL(*cuda_mock_, userObjectCreate(_, _, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaUserObject_t>(0x3000)),
          SaveArg<1>(&captured_cleanup_data),
          SaveArg<2>(&captured_cleanup_fn),
          Return(cudaSuccess)));

  auto tensor = createTestTensor({10, 10});

  {
    auto work1 = comm->send(tensor, 1, true);
    auto work2 = comm->send(tensor, 1, true);
  }

  ASSERT_NE(captured_cleanup_fn, nullptr);
  ASSERT_NE(captured_cleanup_data, nullptr);

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  // Invoke cleanup callback — should only set released flag, NO eventDestroy
  EXPECT_CALL(*cuda_mock_, eventDestroy(_)).Times(0);
  captured_cleanup_fn(captured_cleanup_data);
  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  // Events destroyed during finalize
  switchToReplayMode();
  setupFinalizeExpectations(*comm);
}

// Verify that when userObjectCreate fails during graph capture init,
// the pool entry is harmless (no leak) and the error propagates.
TEST_F(GraphEventTrackerTest, UserObjectCreateFailureCleanup) {
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions();
  comm->init(*device_, "test_user_object_create_failure", options);

  setupGraphCaptureMocks();

  // Override userObjectCreate to fail
  ON_CALL(*cuda_mock_, userObjectCreate(_, _, _, _, _))
      .WillByDefault(Return(cudaErrorMemoryAllocation));

  auto tensor = createTestTensor({10, 10});

  // send() should throw because userObjectCreate fails during graph init
  EXPECT_THROW(comm->send(tensor, 1, true), std::runtime_error);

  switchToReplayMode();

  // The work was never fully created, so cleanup needs to handle the
  // partially-initialized state gracefully.
  EXPECT_CALL(*cuda_mock_, eventDestroy(_)).WillRepeatedly(Return(cudaSuccess));
  EXPECT_CALL(*cuda_mock_, free(_)).WillRepeatedly(Return(cudaSuccess));
  EXPECT_CALL(*cuda_mock_, streamDestroy(_)).WillOnce(Return(cudaSuccess));
  EXPECT_CALL(*nccl_mock_, commDestroy(_)).WillOnce(Return(ncclSuccess));

  comm->finalize();
}

// Verify that when graphRetainUserObject fails, the RAII guard releases the
// user_object (via userObjectRelease) and the error propagates.
TEST_F(GraphEventTrackerTest, GraphRetainUserObjectFailureCleanup) {
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions();
  comm->init(*device_, "test_graph_retain_failure", options);

  setupGraphCaptureMocks();

  // Override graphRetainUserObject to fail
  ON_CALL(*cuda_mock_, graphRetainUserObject(_, _, _, _))
      .WillByDefault(Return(cudaErrorInvalidValue));

  // userObjectRelease should be called by the RAII guard
  EXPECT_CALL(*cuda_mock_, userObjectRelease(_, _))
      .WillOnce(Return(cudaSuccess));

  auto tensor = createTestTensor({10, 10});

  EXPECT_THROW(comm->send(tensor, 1, true), std::runtime_error);

  switchToReplayMode();

  EXPECT_CALL(*cuda_mock_, eventDestroy(_)).WillRepeatedly(Return(cudaSuccess));
  EXPECT_CALL(*cuda_mock_, free(_)).WillRepeatedly(Return(cudaSuccess));
  EXPECT_CALL(*cuda_mock_, streamDestroy(_)).WillOnce(Return(cudaSuccess));
  EXPECT_CALL(*nccl_mock_, commDestroy(_)).WillOnce(Return(ncclSuccess));

  comm->finalize();
}

TEST_F(GraphEventTrackerTest, CheckAllCleansUpReleasedGraphs) {
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions();
  comm->init(*device_, "test_checkall_cleanup", options);

  setupGraphCaptureMocks();
  auto events = setupGraphCaptureEvents();

  void* captured_cleanup_data = nullptr;
  cudaHostFn_t captured_cleanup_fn = nullptr;
  EXPECT_CALL(*cuda_mock_, userObjectCreate(_, _, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaUserObject_t>(0x3000)),
          SaveArg<1>(&captured_cleanup_data),
          SaveArg<2>(&captured_cleanup_fn),
          Return(cudaSuccess)));

  auto tensor = createTestTensor({10, 10});

  {
    auto work = comm->send(tensor, 1, true);
  }

  ASSERT_NE(captured_cleanup_fn, nullptr);
  ASSERT_NE(captured_cleanup_data, nullptr);

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  switchToReplayMode();

  // Invoke cleanup callback — sets released flag
  captured_cleanup_fn(captured_cleanup_data);

  // checkGraphEvents calls checkAll which calls cleanupReleasedGraphs.
  // Events are now stashed in the tracker event pool rather than destroyed
  // inline, so checkAll must NOT call eventDestroy on the captured events.
  std::vector<cudaEvent_t> destroyed_events;
  EXPECT_CALL(*cuda_mock_, eventDestroy(_))
      .WillRepeatedly(DoAll(
          [&destroyed_events](cudaEvent_t event) {
            destroyed_events.push_back(event);
          },
          Return(cudaSuccess)));

  comm->checkGraphEvents();

  EXPECT_EQ(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.start),
      destroyed_events.end())
      << "start_event must be stashed by cleanupReleasedGraphs, not destroyed";
  EXPECT_EQ(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.end),
      destroyed_events.end())
      << "end_event must be stashed by cleanupReleasedGraphs, not destroyed";

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  // Re-install the destroy tracker for the finalize/destroy sequence.
  EXPECT_CALL(*cuda_mock_, eventDestroy(_))
      .WillRepeatedly(DoAll(
          [&destroyed_events](cudaEvent_t event) {
            destroyed_events.push_back(event);
          },
          Return(cudaSuccess)));
  EXPECT_CALL(*cuda_mock_, free(_)).WillRepeatedly(Return(cudaSuccess));
  EXPECT_CALL(*cuda_mock_, streamDestroy(_)).WillOnce(Return(cudaSuccess));
  EXPECT_CALL(*nccl_mock_, commDestroy(_)).WillOnce(Return(ncclSuccess));
  comm->finalize();

  // Drain happens at comm destruction.
  comm.reset();
  EXPECT_NE(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.start),
      destroyed_events.end())
      << "start_event not destroyed at comm destruction";
  EXPECT_NE(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.end),
      destroyed_events.end())
      << "end_event not destroyed at comm destruction";
}

TEST_F(GraphEventTrackerTest, MultipleGraphsOnlyReleasedOneCleanedUp) {
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions();
  comm->init(*device_, "test_multi_graph_partial_cleanup", options);

  cudaEvent_t start1 = reinterpret_cast<cudaEvent_t>(0xC001);
  cudaEvent_t end1 = reinterpret_cast<cudaEvent_t>(0xC002);
  cudaEvent_t sync1 = reinterpret_cast<cudaEvent_t>(0xC003);
  cudaEvent_t start2 = reinterpret_cast<cudaEvent_t>(0xC004);
  cudaEvent_t end2 = reinterpret_cast<cudaEvent_t>(0xC005);
  cudaEvent_t sync2 = reinterpret_cast<cudaEvent_t>(0xC006);

  void* cleanup_data_1 = nullptr;
  cudaHostFn_t cleanup_fn_1 = nullptr;

  setupGraphCaptureMocks(/*graph_id=*/100);

  EXPECT_CALL(*cuda_mock_, eventCreateWithFlags(_, _))
      .WillOnce(DoAll(SetArgPointee<0>(start1), Return(cudaSuccess)))
      .WillOnce(DoAll(SetArgPointee<0>(end1), Return(cudaSuccess)))
      .WillOnce(DoAll(SetArgPointee<0>(sync1), Return(cudaSuccess)))
      .WillRepeatedly(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaEvent_t>(0xA100)),
          Return(cudaSuccess)));

  EXPECT_CALL(*cuda_mock_, userObjectCreate(_, _, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaUserObject_t>(0x3000)),
          SaveArg<1>(&cleanup_data_1),
          SaveArg<2>(&cleanup_fn_1),
          Return(cudaSuccess)));

  auto tensor = createTestTensor({10, 10});

  {
    auto work1 = comm->send(tensor, 1, true);
  }

  ASSERT_NE(cleanup_fn_1, nullptr);
  ASSERT_NE(cleanup_data_1, nullptr);

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());
  cuda_mock_->setupDefaultBehaviors();

  void* cleanup_data_2 = nullptr;
  cudaHostFn_t cleanup_fn_2 = nullptr;

  setupGraphCaptureMocks(
      /*graph_id=*/200, reinterpret_cast<cudaGraph_t>(0xB001));

  EXPECT_CALL(*cuda_mock_, eventCreateWithFlags(_, _))
      .WillOnce(DoAll(SetArgPointee<0>(start2), Return(cudaSuccess)))
      .WillOnce(DoAll(SetArgPointee<0>(end2), Return(cudaSuccess)))
      .WillOnce(DoAll(SetArgPointee<0>(sync2), Return(cudaSuccess)))
      .WillRepeatedly(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaEvent_t>(0xA100)),
          Return(cudaSuccess)));

  EXPECT_CALL(*cuda_mock_, userObjectCreate(_, _, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaUserObject_t>(0x3001)),
          SaveArg<1>(&cleanup_data_2),
          SaveArg<2>(&cleanup_fn_2),
          Return(cudaSuccess)));

  {
    auto work2 = comm->send(tensor, 1, true);
  }

  ASSERT_NE(cleanup_fn_2, nullptr);
  ASSERT_NE(cleanup_data_2, nullptr);

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  switchToReplayMode();

  // Release only graph 1
  cleanup_fn_1(cleanup_data_1);

  // checkAll stashes graph 1's events in the tracker pool; graph 2 stays in
  // graphs_ and its events are untouched. No eventDestroy fires on either.
  std::vector<cudaEvent_t> destroyed_events;
  EXPECT_CALL(*cuda_mock_, eventDestroy(_))
      .WillRepeatedly(DoAll(
          [&destroyed_events](cudaEvent_t event) {
            destroyed_events.push_back(event);
          },
          Return(cudaSuccess)));

  ON_CALL(*cuda_mock_, eventQuery(start2))
      .WillByDefault(Return(cudaErrorNotReady));
  ON_CALL(*cuda_mock_, eventQuery(end2))
      .WillByDefault(Return(cudaErrorNotReady));

  comm->checkGraphEvents();

  for (auto e : {start1, end1, start2, end2}) {
    EXPECT_EQ(
        std::find(destroyed_events.begin(), destroyed_events.end(), e),
        destroyed_events.end())
        << "event must be stashed by cleanupReleasedGraphs, not destroyed";
  }

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  // Re-install destroy tracker, then run finalize and comm destruction.
  // graph 2's events get stashed by destroyAll, then both graphs' events
  // drain at ~GraphEventTracker.
  EXPECT_CALL(*cuda_mock_, eventDestroy(_))
      .WillRepeatedly(DoAll(
          [&destroyed_events](cudaEvent_t event) {
            destroyed_events.push_back(event);
          },
          Return(cudaSuccess)));
  EXPECT_CALL(*cuda_mock_, free(_)).WillRepeatedly(Return(cudaSuccess));
  EXPECT_CALL(*cuda_mock_, streamDestroy(_)).WillOnce(Return(cudaSuccess));
  EXPECT_CALL(*nccl_mock_, commDestroy(_)).WillOnce(Return(ncclSuccess));
  comm->finalize();
  comm.reset();

  for (auto e : {start1, end1, start2, end2}) {
    EXPECT_NE(
        std::find(destroyed_events.begin(), destroyed_events.end(), e),
        destroyed_events.end())
        << "event was not destroyed at comm destruction";
  }
}

// Verify the pinned-memory counter region is reused across recaptures rather
// than freed on the watchdog thread. cudaFreeHost is synchronizing — running
// it inline inside cleanupReleasedGraphs can block the watchdog (which also
// services NCCL work-progress) when the device is busy with the next eager
// warmup forward, causing a peer-visible deadlock. See GraphEventTracker.cpp
// cleanupReleasedGraphs() for the full rationale.
TEST_F(GraphEventTrackerTest, CounterPoolReusesAcrossRecaptures) {
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions();
  comm->init(*device_, "test_counter_pool_reuse", options);

  // Track every uint64_t pinned allocation handed out. Pool reuse means
  // the second recapture should NOT add a new entry here.
  std::vector<uint64_t*> allocated_counters;
  ON_CALL(*cuda_mock_, hostAlloc(_, _, _))
      .WillByDefault(
          [&allocated_counters](void** ptr, size_t size, unsigned int) {
            *ptr = std::calloc(1, size);
            if (size == sizeof(uint64_t)) {
              allocated_counters.push_back(static_cast<uint64_t*>(*ptr));
            }
            return cudaSuccess;
          });

  // Track hostFree calls — must be empty during steady-state recapture.
  std::vector<void*> freed_pointers;
  ON_CALL(*cuda_mock_, hostFree(_)).WillByDefault([&freed_pointers](void* ptr) {
    freed_pointers.push_back(ptr);
    std::free(ptr);
    return cudaSuccess;
  });

  // -------- Capture graph 1 --------
  void* cleanup_data_1 = nullptr;
  cudaHostFn_t cleanup_fn_1 = nullptr;
  setupGraphCaptureMocks(/*graph_id=*/100);
  EXPECT_CALL(*cuda_mock_, eventCreateWithFlags(_, _))
      .WillRepeatedly(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaEvent_t>(0xA100)),
          Return(cudaSuccess)));
  EXPECT_CALL(*cuda_mock_, userObjectCreate(_, _, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaUserObject_t>(0x3000)),
          SaveArg<1>(&cleanup_data_1),
          SaveArg<2>(&cleanup_fn_1),
          Return(cudaSuccess)));

  auto tensor = createTestTensor({10, 10});
  {
    auto work = comm->send(tensor, 1, true);
  }

  ASSERT_EQ(allocated_counters.size(), 1u)
      << "First graph should allocate exactly one counter region";
  uint64_t* first_counter = allocated_counters[0];
  ASSERT_NE(first_counter, nullptr);

  // Pretend the GPU kernel had incremented the counter — pool acquire must
  // reset it back to zero before handing it to the next graph.
  *first_counter = 7;

  // -------- Release graph 1 --------
  cleanup_fn_1(cleanup_data_1);

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());
  cuda_mock_->setupDefaultBehaviors();

  switchToReplayMode();
  comm->checkGraphEvents();
  EXPECT_TRUE(freed_pointers.empty())
      << "hostFree must not run on the watchdog cleanup path; got "
      << freed_pointers.size() << " calls";

  // -------- Capture graph 2 (should reuse pooled counter) --------
  setupGraphCaptureMocks(
      /*graph_id=*/200, reinterpret_cast<cudaGraph_t>(0xB001));
  EXPECT_CALL(*cuda_mock_, eventCreateWithFlags(_, _))
      .WillRepeatedly(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaEvent_t>(0xA200)),
          Return(cudaSuccess)));
  EXPECT_CALL(*cuda_mock_, userObjectCreate(_, _, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaUserObject_t>(0x3001)),
          Return(cudaSuccess)));

  {
    auto work = comm->send(tensor, 1, true);
  }

  EXPECT_EQ(allocated_counters.size(), 1u)
      << "Second capture should reuse the pooled counter, not allocate";
  // maybeInitGraphState calls acquireCounter() (which resets to 0), then
  // immediately calls replay_counter->increment(stream) which adds 1.
  // If reset() didn't run, this would be 7 + 1 = 8 (the sentinel above).
  EXPECT_EQ(*first_counter, 1u)
      << "Pooled counter must be reset to 0 before the per-capture increment "
         "(would be 8 if the sentinel survived re-acquire)";

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());
  switchToReplayMode();
  setupFinalizeExpectations(*comm);
}

// Verify that pinned-memory frees are deferred until comm shutdown rather
// than running during steady-state recaptures. This is the core invariant
// the pool fix establishes: zero hostFree on the watchdog thread.
TEST_F(GraphEventTrackerTest, CounterPoolDrainsAtFinalize) {
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions();
  comm->init(*device_, "test_counter_pool_drains_at_finalize", options);

  std::vector<uint64_t*> allocated_counters;
  ON_CALL(*cuda_mock_, hostAlloc(_, _, _))
      .WillByDefault(
          [&allocated_counters](void** ptr, size_t size, unsigned int) {
            *ptr = std::calloc(1, size);
            if (size == sizeof(uint64_t)) {
              allocated_counters.push_back(static_cast<uint64_t*>(*ptr));
            }
            return cudaSuccess;
          });

  std::vector<void*> freed_pointers;
  // setupDefaultBehaviors() re-installs the default hostFree ON_CALL each
  // time it's called, so wrap our tracking install in a helper and call it
  // again after every setupDefaultBehaviors().
  auto installHostFreeTracker = [this, &freed_pointers]() {
    ON_CALL(*cuda_mock_, hostFree(_))
        .WillByDefault([&freed_pointers](void* ptr) {
          freed_pointers.push_back(ptr);
          std::free(ptr);
          return cudaSuccess;
        });
  };
  installHostFreeTracker();

  ON_CALL(*cuda_mock_, eventCreateWithFlags(_, _))
      .WillByDefault(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaEvent_t>(0xA100)),
          Return(cudaSuccess)));

  auto tensor = createTestTensor({10, 10});

  // Run two capture/release cycles. Pool reuses the same counter, so only
  // one hostAlloc total.
  for (int cycle = 0; cycle < 2; ++cycle) {
    void* cleanup_data = nullptr;
    cudaHostFn_t cleanup_fn = nullptr;
    setupGraphCaptureMocks(
        /*graph_id=*/static_cast<unsigned long long>(100 + cycle),
        reinterpret_cast<cudaGraph_t>(0xB000 + cycle));
    EXPECT_CALL(*cuda_mock_, userObjectCreate(_, _, _, _, _))
        .WillOnce(DoAll(
            SetArgPointee<0>(
                reinterpret_cast<cudaUserObject_t>(0x3000 + cycle)),
            SaveArg<1>(&cleanup_data),
            SaveArg<2>(&cleanup_fn),
            Return(cudaSuccess)));

    {
      auto work = comm->send(tensor, 1, true);
    }

    ASSERT_NE(cleanup_fn, nullptr);
    cleanup_fn(cleanup_data);
    switchToReplayMode();
    comm->checkGraphEvents();

    EXPECT_TRUE(freed_pointers.empty())
        << "hostFree must not run during recapture cycle " << cycle;

    ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());
    cuda_mock_->setupDefaultBehaviors();
    installHostFreeTracker();
  }

  EXPECT_EQ(allocated_counters.size(), 1u)
      << "Pool should serve all recaptures from one allocation";

  // finalize() runs destroyAll() but does not destroy the comm — the pool
  // drain happens in ~GraphEventTracker, which fires when the comm itself
  // is destroyed. So no hostFree yet.
  setupFinalizeExpectations(*comm);
  installHostFreeTracker(); // setupFinalizeExpectations doesn't touch host*
  EXPECT_TRUE(freed_pointers.empty())
      << "hostFree must not run at finalize() — only at comm destruction";

  // Destroy the comm; the pool drains here, releasing pinned counters.
  comm.reset();
  EXPECT_GE(freed_pointers.size(), 1u)
      << "Pool should drain at comm destruction, freeing the pinned "
         "counter region";
}

// Verify that graph events captured by GraphEventTracker are destroyed at
// comm destruction (~GraphEventTracker drains event_pool_), not earlier.
// finalize() runs destroyAll() which only stashes; the actual cudaEventDestroy
// calls happen when the comm itself is destroyed and the device is quiescent.
TEST_F(GraphEventTrackerTest, EventPoolDrainsAtCommDestruction) {
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions();
  comm->init(*device_, "test_event_pool_drains_at_destruction", options);

  // Track every eventDestroy call across the full lifecycle.
  std::vector<cudaEvent_t> destroyed_events;
  auto trackEventDestroy = [&destroyed_events](cudaEvent_t e) {
    destroyed_events.push_back(e);
    return cudaSuccess;
  };
  ON_CALL(*cuda_mock_, eventDestroy(_)).WillByDefault(trackEventDestroy);

  // -------- Capture / release a graph --------
  void* cleanup_data = nullptr;
  cudaHostFn_t cleanup_fn = nullptr;
  setupGraphCaptureMocks(/*graph_id=*/100);
  auto events = setupGraphCaptureEvents();
  EXPECT_CALL(*cuda_mock_, userObjectCreate(_, _, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaUserObject_t>(0x3000)),
          SaveArg<1>(&cleanup_data),
          SaveArg<2>(&cleanup_fn),
          Return(cudaSuccess)));

  {
    auto tensor = createTestTensor({10, 10});
    auto work = comm->send(tensor, 1, true);
  }

  ASSERT_NE(cleanup_fn, nullptr);
  cleanup_fn(cleanup_data);
  switchToReplayMode();
  comm->checkGraphEvents();

  // Watchdog cleanup must not destroy start/end events.
  EXPECT_EQ(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.start),
      destroyed_events.end())
      << "start_event destroyed during watchdog cleanup";
  EXPECT_EQ(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.end),
      destroyed_events.end())
      << "end_event destroyed during watchdog cleanup";

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());
  cuda_mock_->setupDefaultBehaviors();

  // Manually set up finalize expectations so our tracker keeps capturing
  // eventDestroy across both finalize() and the subsequent comm destruction.
  // (The default setupFinalizeExpectations installs an EXPECT_CALL on
  // eventDestroy that would shadow our ON_CALL tracker.)
  EXPECT_CALL(*cuda_mock_, eventDestroy(_)).WillRepeatedly(trackEventDestroy);
  EXPECT_CALL(*cuda_mock_, free(_)).WillRepeatedly(Return(cudaSuccess));
  EXPECT_CALL(*cuda_mock_, streamDestroy(_)).WillOnce(Return(cudaSuccess));
  EXPECT_CALL(*nccl_mock_, commDestroy(_)).WillOnce(Return(ncclSuccess));
  comm->finalize();

  // finalize stashes events into event_pool_ but doesn't destroy them.
  EXPECT_EQ(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.start),
      destroyed_events.end())
      << "start_event destroyed during finalize (must defer to ~GraphEventTracker)";
  EXPECT_EQ(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.end),
      destroyed_events.end())
      << "end_event destroyed during finalize (must defer to ~GraphEventTracker)";

  // Destroy the comm; ~GraphEventTracker drains event_pool_ here.
  comm.reset();
  EXPECT_NE(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.start),
      destroyed_events.end())
      << "start_event was not destroyed when comm was destroyed";
  EXPECT_NE(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.end),
      destroyed_events.end())
      << "end_event was not destroyed when comm was destroyed";
}

TEST_F(GraphEventTrackerTest, DestroyAllIgnoresReleasedFlag) {
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions();
  comm->init(*device_, "test_destroy_all_ignores_flag", options);

  setupGraphCaptureMocks();
  auto events = setupGraphCaptureEvents();

  void* captured_cleanup_data = nullptr;
  cudaHostFn_t captured_cleanup_fn = nullptr;
  EXPECT_CALL(*cuda_mock_, userObjectCreate(_, _, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaUserObject_t>(0x3000)),
          SaveArg<1>(&captured_cleanup_data),
          SaveArg<2>(&captured_cleanup_fn),
          Return(cudaSuccess)));

  auto tensor = createTestTensor({10, 10});

  {
    auto work = comm->send(tensor, 1, true);
  }

  ASSERT_NE(captured_cleanup_fn, nullptr);
  ASSERT_NE(captured_cleanup_data, nullptr);

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  // Set released flag — but destroyAll should still clean up
  captured_cleanup_fn(captured_cleanup_data);

  switchToReplayMode();

  // destroyAll (via finalize) stashes events into the tracker pool regardless
  // of the released flag; the actual destruction fires at ~GraphEventTracker.
  std::vector<cudaEvent_t> destroyed_events;
  EXPECT_CALL(*cuda_mock_, eventDestroy(_))
      .WillRepeatedly(DoAll(
          [&destroyed_events](cudaEvent_t event) {
            destroyed_events.push_back(event);
          },
          Return(cudaSuccess)));
  EXPECT_CALL(*cuda_mock_, free(_)).WillRepeatedly(Return(cudaSuccess));
  EXPECT_CALL(*cuda_mock_, streamDestroy(_)).WillOnce(Return(cudaSuccess));
  EXPECT_CALL(*nccl_mock_, commDestroy(_)).WillOnce(Return(ncclSuccess));

  comm->finalize();
  comm.reset();

  EXPECT_NE(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.start),
      destroyed_events.end())
      << "start event was not destroyed at comm destruction";
  EXPECT_NE(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.end),
      destroyed_events.end())
      << "end event was not destroyed at comm destruction";
}

TEST_F(GraphEventTrackerTest, EventResetByReplayDefeatsTimeout) {
  EXPECT_DEATH(
      {
        cuda_mock_->setupDefaultBehaviors();
        nccl_mock_->setupDefaultBehaviors();

        auto options = createAbortModeOptions(std::chrono::milliseconds(100));
        auto comm = createMockedTorchComm();
        comm->init(*device_, "test_event_reset_defeats_timeout", options);

        setupGraphCaptureMocks();
        auto events = setupGraphCaptureEvents();
        setupEventRecordMocks();

        auto tensor = createTestTensor({10, 10});
        auto work = comm->send(tensor, 1, true);

        switchToReplayMode();

        std::atomic<bool> events_reset{false};

        ON_CALL(*cuda_mock_, eventQuery(events.start))
            .WillByDefault(Invoke([&events_reset](cudaEvent_t) -> cudaError_t {
              return events_reset.load(std::memory_order_relaxed)
                  ? cudaErrorNotReady
                  : cudaSuccess;
            }));
        ON_CALL(*cuda_mock_, eventQuery(events.end))
            .WillByDefault(Return(cudaErrorNotReady));

        // wait for watchdog poll to observe the collective
        // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));

        // simulate new replay submission
        events_reset.store(true, std::memory_order_relaxed);

        // wait for another watchdog poll -- should timeout
        // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
        std::this_thread::sleep_for(std::chrono::seconds(5));
      },
      "Graph monitor: collective TIMED OUT for graph");
}

TEST_F(GraphEventTrackerTest, TimeoutMonitoringDisabled_NoStartEndEvents) {
  resetGraphTimeoutMonitoringCacheForTest();

  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions();
  ::setenv("TORCHCOMM_NCCLX_GRAPH_TIMEOUT_MONITORING", "0", 1);
  comm->init(*device_, "test_no_timeout_events", options);

  setupGraphCaptureMocks();

  // Only 1 eventCreateWithFlags (sync_event_), not 3
  cudaEvent_t sync = reinterpret_cast<cudaEvent_t>(0xA003);
  EXPECT_CALL(*cuda_mock_, eventCreateWithFlags(_, _))
      .WillOnce(DoAll(SetArgPointee<0>(sync), Return(cudaSuccess)));

  // No launchHostFunc for replay counter
  EXPECT_CALL(*cuda_mock_, launchHostFunc(_, _, _)).Times(0);

  // Cleanup callback still installed
  EXPECT_CALL(*cuda_mock_, userObjectCreate(_, _, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaUserObject_t>(0x3000)),
          Return(cudaSuccess)));
  EXPECT_CALL(*cuda_mock_, graphRetainUserObject(_, _, _, _))
      .WillOnce(Return(cudaSuccess));

  setupEventRecordMocks();

  auto tensor = createTestTensor({10, 10});

  {
    auto work = comm->send(tensor, 1, true);
  }

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  switchToReplayMode();
  ::unsetenv("TORCHCOMM_NCCLX_GRAPH_TIMEOUT_MONITORING");
  resetGraphTimeoutMonitoringCacheForTest();
  setupFinalizeExpectations(*comm);
}

TEST_F(
    GraphEventTrackerTest,
    TimeoutMonitoringDisabled_CheckGraphEventsNoEventQueries) {
  resetGraphTimeoutMonitoringCacheForTest();

  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions();
  ::setenv("TORCHCOMM_NCCLX_GRAPH_TIMEOUT_MONITORING", "0", 1);
  comm->init(*device_, "test_no_event_queries", options);

  setupGraphCaptureMocks();

  cudaEvent_t sync = reinterpret_cast<cudaEvent_t>(0xA003);
  EXPECT_CALL(*cuda_mock_, eventCreateWithFlags(_, _))
      .WillOnce(DoAll(SetArgPointee<0>(sync), Return(cudaSuccess)))
      .WillRepeatedly(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaEvent_t>(0xA100)),
          Return(cudaSuccess)));

  setupEventRecordMocks();

  auto tensor = createTestTensor({10, 10});

  {
    auto work = comm->send(tensor, 1, true);
  }

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  switchToReplayMode();

  // No eventQuery calls should be made (no GraphWork entries)
  EXPECT_CALL(*cuda_mock_, eventQuery(_)).Times(0);

  comm->checkGraphEvents();

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  ::unsetenv("TORCHCOMM_NCCLX_GRAPH_TIMEOUT_MONITORING");
  resetGraphTimeoutMonitoringCacheForTest();
  setupFinalizeExpectations(*comm);
}
// --- Colltrace graph tracing disables GraphEventTracker monitoring ---

TEST_F(
    GraphEventTrackerTest,
    ColltraceGraphTracing_DisablesGraphTimeoutMonitoring) {
  resetGraphTimeoutMonitoringCacheForTest();

  ::setenv("NCCL_COLLTRACE", "trace", 1);
  ::setenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH", "1", 1);
  EXPECT_FALSE(isGraphTimeoutMonitoringEnabled());

  ::unsetenv("NCCL_COLLTRACE");
  ::unsetenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH");
  resetGraphTimeoutMonitoringCacheForTest();
}

TEST_F(
    GraphEventTrackerTest,
    ColltraceGraphTracing_MonitoringEnabledWhenNotSet) {
  resetGraphTimeoutMonitoringCacheForTest();

  ::unsetenv("NCCL_COLLTRACE");
  ::unsetenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH");
  EXPECT_TRUE(isGraphTimeoutMonitoringEnabled());

  resetGraphTimeoutMonitoringCacheForTest();
}

// NCCL_COLLTRACE_TRACE_CUDA_GRAPH only disables monitoring when colltrace is
// actually active (NCCL_COLLTRACE=trace|verbose). With NCCL_COLLTRACE unset the
// colltrace watchdog plugin is not running, so monitoring must stay enabled.
TEST_F(
    GraphEventTrackerTest,
    ColltraceGraphTracing_MonitoringEnabledWhenColltraceUnset) {
  resetGraphTimeoutMonitoringCacheForTest();

  ::unsetenv("NCCL_COLLTRACE");
  ::setenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH", "1", 1);
  EXPECT_TRUE(isGraphTimeoutMonitoringEnabled());

  ::unsetenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH");
  resetGraphTimeoutMonitoringCacheForTest();
}

// NCCL_COLLTRACE set to a value other than trace/verbose does not activate the
// colltrace watchdog plugin, so monitoring stays enabled.
TEST_F(
    GraphEventTrackerTest,
    ColltraceGraphTracing_MonitoringEnabledWhenColltraceNotTraceOrVerbose) {
  resetGraphTimeoutMonitoringCacheForTest();

  ::setenv("NCCL_COLLTRACE", "1", 1);
  ::setenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH", "1", 1);
  EXPECT_TRUE(isGraphTimeoutMonitoringEnabled());

  ::unsetenv("NCCL_COLLTRACE");
  ::unsetenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH");
  resetGraphTimeoutMonitoringCacheForTest();
}

// NCCL_COLLTRACE=verbose also activates the colltrace watchdog plugin, so
// monitoring is disabled just like the trace case.
TEST_F(
    GraphEventTrackerTest,
    ColltraceGraphTracing_VerboseDisablesGraphTimeoutMonitoring) {
  resetGraphTimeoutMonitoringCacheForTest();

  ::setenv("NCCL_COLLTRACE", "verbose", 1);
  ::setenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH", "1", 1);
  EXPECT_FALSE(isGraphTimeoutMonitoringEnabled());

  ::unsetenv("NCCL_COLLTRACE");
  ::unsetenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH");
  resetGraphTimeoutMonitoringCacheForTest();
}

TEST_F(
    GraphEventTrackerTest,
    ColltraceGraphTracing_ExplicitDisableOverridesColltrace) {
  resetGraphTimeoutMonitoringCacheForTest();

  ::setenv("TORCHCOMM_NCCLX_GRAPH_TIMEOUT_MONITORING", "0", 1);
  ::setenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH", "1", 1);
  EXPECT_FALSE(isGraphTimeoutMonitoringEnabled());

  ::unsetenv("TORCHCOMM_NCCLX_GRAPH_TIMEOUT_MONITORING");
  ::unsetenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH");
  resetGraphTimeoutMonitoringCacheForTest();
}

TEST_F(
    GraphEventTrackerTest,
    ColltraceGraphTracing_NoStartEndEventsOrReplayCounter) {
  resetGraphTimeoutMonitoringCacheForTest();

  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions();
  ::setenv("NCCL_COLLTRACE", "trace", 1);
  ::setenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH", "1", 1);
  comm->init(*device_, "test_colltrace_graph_no_events", options);

  setupGraphCaptureMocks();

  // Only sync_event_ created — no start_event_ or end_event_
  cudaEvent_t sync = reinterpret_cast<cudaEvent_t>(0xA003);
  EXPECT_CALL(*cuda_mock_, eventCreateWithFlags(_, _))
      .WillOnce(DoAll(SetArgPointee<0>(sync), Return(cudaSuccess)));

  // No replay counter kernel
  EXPECT_CALL(*cuda_mock_, launchHostFunc(_, _, _)).Times(0);

  // Cleanup callback still installed
  EXPECT_CALL(*cuda_mock_, userObjectCreate(_, _, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaUserObject_t>(0x3000)),
          Return(cudaSuccess)));
  EXPECT_CALL(*cuda_mock_, graphRetainUserObject(_, _, _, _))
      .WillOnce(Return(cudaSuccess));

  setupEventRecordMocks();

  auto tensor = createTestTensor({10, 10});

  {
    auto work = comm->send(tensor, 1, true);
  }

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  switchToReplayMode();
  ::unsetenv("NCCL_COLLTRACE");
  ::unsetenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH");
  resetGraphTimeoutMonitoringCacheForTest();
  setupFinalizeExpectations(*comm);
}

TEST_F(GraphEventTrackerTest, ColltraceGraphTracing_CheckGraphEventsReturnsOK) {
  resetGraphTimeoutMonitoringCacheForTest();

  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createAbortModeOptions();
  ::setenv("NCCL_COLLTRACE", "trace", 1);
  ::setenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH", "1", 1);
  comm->init(*device_, "test_colltrace_graph_check_ok", options);

  setupGraphCaptureMocks();

  cudaEvent_t sync = reinterpret_cast<cudaEvent_t>(0xA003);
  EXPECT_CALL(*cuda_mock_, eventCreateWithFlags(_, _))
      .WillOnce(DoAll(SetArgPointee<0>(sync), Return(cudaSuccess)))
      .WillRepeatedly(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaEvent_t>(0xA100)),
          Return(cudaSuccess)));

  setupEventRecordMocks();

  auto tensor = createTestTensor({10, 10});

  {
    auto work = comm->send(tensor, 1, true);
  }

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  switchToReplayMode();

  // No event queries — checkAll() returns OK immediately
  EXPECT_CALL(*cuda_mock_, eventQuery(_)).Times(0);

  comm->checkGraphEvents();

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  ::unsetenv("NCCL_COLLTRACE");
  ::unsetenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH");
  resetGraphTimeoutMonitoringCacheForTest();
  setupFinalizeExpectations(*comm);
}

// --- tryEnableColltraceTimeoutWatchdog env var gating ---

TEST_F(
    GraphEventTrackerTest,
    TryEnableColltraceWatchdog_ReturnsFalseWhenGraphTracingDisabled) {
  ::unsetenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH");
  EXPECT_FALSE(
      tryEnableColltraceTimeoutWatchdog(std::chrono::milliseconds{5000}));
}

// Regression: with NCCL_COLLTRACE unset the colltrace watchdog is never created
// (this path short-circuits), so it must not be relied on to replace the
// GraphEventTracker timeout monitoring.
TEST_F(
    GraphEventTrackerTest,
    TryEnableColltraceWatchdog_ReturnsFalseWhenColltraceUnset) {
  ::unsetenv("NCCL_COLLTRACE");
  ::setenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH", "1", 1);
  EXPECT_FALSE(
      tryEnableColltraceTimeoutWatchdog(std::chrono::milliseconds{5000}));

  ::unsetenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH");
}

TEST_F(
    GraphEventTrackerTest,
    TryEnableColltraceWatchdog_ReturnsFalseWhenMonitoringExplicitlyDisabled) {
  ::setenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH", "1", 1);
  ::setenv("TORCHCOMM_NCCLX_GRAPH_TIMEOUT_MONITORING", "0", 1);
  EXPECT_FALSE(
      tryEnableColltraceTimeoutWatchdog(std::chrono::milliseconds{5000}));

  ::unsetenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH");
  ::unsetenv("TORCHCOMM_NCCLX_GRAPH_TIMEOUT_MONITORING");
}

// --- NCCL_COLLTRACE stringlist parsing in isGraphTimeoutMonitoringEnabled ---
// NCCL_COLLTRACE is a comma-separated stringlist. Graph-timeout monitoring is
// handed to colltrace (GraphEventTracker disabled) iff a "trace"/"verbose"
// token is present AND NCCL_COLLTRACE_TRACE_CUDA_GRAPH is set — consistent with
// CollTraceWrapper::newCollTraceInit. Otherwise GraphEventTracker must remain
// active so a watchdog always covers graph-captured collectives.

TEST_F(
    GraphEventTrackerTest,
    GraphTimeoutMonitoring_DisabledForMultiTokenTrace) {
  // "trace,algostat" starts the colltrace worker + watchdog, so colltrace owns
  // graph-timeout detection and GraphEventTracker monitoring is disabled.
  ::setenv("NCCL_COLLTRACE", "trace,algostat", 1);
  ::setenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH", "1", 1);
  resetGraphTimeoutMonitoringCacheForTest();

  EXPECT_FALSE(isGraphTimeoutMonitoringEnabled());

  ::unsetenv("NCCL_COLLTRACE");
  ::unsetenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH");
  resetGraphTimeoutMonitoringCacheForTest();
}

TEST_F(GraphEventTrackerTest, GraphTimeoutMonitoring_KeptForAlgostatOnly) {
  // "algostat" alone does NOT start the colltrace worker/watchdog, so
  // GraphEventTracker must stay active even though cudagraph tracing is asked.
  ::setenv("NCCL_COLLTRACE", "algostat", 1);
  ::setenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH", "1", 1);
  resetGraphTimeoutMonitoringCacheForTest();

  EXPECT_TRUE(isGraphTimeoutMonitoringEnabled());

  ::unsetenv("NCCL_COLLTRACE");
  ::unsetenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH");
  resetGraphTimeoutMonitoringCacheForTest();
}

TEST_F(GraphEventTrackerTest, GraphTimeoutMonitoring_KeptForEmptyColltrace) {
  // Regression for the observed MAST hang: NCCL_COLLTRACE blanked while
  // cudagraph tracing is requested. No colltrace watchdog is installed, so
  // GraphEventTracker must remain the graph-collective watchdog.
  ::setenv("NCCL_COLLTRACE", "", 1);
  ::setenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH", "1", 1);
  resetGraphTimeoutMonitoringCacheForTest();

  EXPECT_TRUE(isGraphTimeoutMonitoringEnabled());

  ::unsetenv("NCCL_COLLTRACE");
  ::unsetenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH");
  resetGraphTimeoutMonitoringCacheForTest();
}

TEST_F(
    GraphEventTrackerTest,
    GraphTimeoutMonitoring_KeptForWrongCaseColltrace) {
  // Token matching is case-sensitive (mirrors CollTraceWrapper); "Trace" does
  // not enable colltrace, so GraphEventTracker must stay active.
  ::setenv("NCCL_COLLTRACE", "Trace", 1);
  ::setenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH", "1", 1);
  resetGraphTimeoutMonitoringCacheForTest();

  EXPECT_TRUE(isGraphTimeoutMonitoringEnabled());

  ::unsetenv("NCCL_COLLTRACE");
  ::unsetenv("NCCL_COLLTRACE_TRACE_CUDA_GRAPH");
  resetGraphTimeoutMonitoringCacheForTest();
}

} // namespace torch::comms::test
