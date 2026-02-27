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
    ON_CALL(*cuda_mock_, launchHostFunc(_, _, _))
        .WillByDefault(Return(cudaSuccess));
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
    EXPECT_CALL(*cuda_mock_, free(_)).WillOnce(Return(cudaSuccess));
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

  cudaHostFn_t captured_replay_fn = nullptr;
  void* captured_replay_data = nullptr;
  // Override launchHostFunc to capture the replay callback
  ON_CALL(*cuda_mock_, launchHostFunc(_, _, _))
      .WillByDefault(DoAll(
          SaveArg<1>(&captured_replay_fn),
          SaveArg<2>(&captured_replay_data),
          Return(cudaSuccess)));

  auto tensor = createTestTensor({10, 10});
  auto work = comm->send(tensor, 1, true);

  switchToReplayMode();

  ON_CALL(*cuda_mock_, eventQuery(events.start))
      .WillByDefault(Return(cudaSuccess));
  ON_CALL(*cuda_mock_, eventQuery(events.end))
      .WillByDefault(Return(cudaErrorNotReady));

  ASSERT_NE(captured_replay_fn, nullptr);
  ASSERT_NE(captured_replay_data, nullptr);

  std::atomic<bool> stop_replays{false};
  std::thread replay_thread([&]() {
    while (!stop_replays.load()) {
      // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (!stop_replays.load()) {
        captured_replay_fn(captured_replay_data);
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

// destroyAll should destroy exactly the entry-owned events. We track which
// events are destroyed to verify precise cleanup.
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
  EXPECT_CALL(*cuda_mock_, free(_)).WillOnce(Return(cudaSuccess));
  EXPECT_CALL(*cuda_mock_, streamDestroy(_)).WillOnce(Return(cudaSuccess));
  EXPECT_CALL(*nccl_mock_, commDestroy(_)).WillOnce(Return(ncclSuccess));

  comm->finalize();

  EXPECT_TRUE(
      std::find(
          destroyed_events.begin(), destroyed_events.end(), events.start) !=
      destroyed_events.end())
      << "start event was not destroyed by destroyAll";
  EXPECT_TRUE(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.end) !=
      destroyed_events.end())
      << "end event was not destroyed by destroyAll";
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
  EXPECT_CALL(*cuda_mock_, free(_)).WillOnce(Return(cudaSuccess));
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
  EXPECT_CALL(*cuda_mock_, free(_)).WillOnce(Return(cudaSuccess));
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

  // checkGraphEvents calls checkAll which calls cleanupReleasedGraphs
  EXPECT_CALL(*cuda_mock_, eventDestroy(events.start))
      .WillOnce(Return(cudaSuccess));
  EXPECT_CALL(*cuda_mock_, eventDestroy(events.end))
      .WillOnce(Return(cudaSuccess));

  comm->checkGraphEvents();

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  // finalize should have nothing to clean up (already cleaned by checkAll)
  setupFinalizeExpectations(*comm);
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

  // checkAll should destroy graph 1's events but not graph 2's
  EXPECT_CALL(*cuda_mock_, eventDestroy(start1)).WillOnce(Return(cudaSuccess));
  EXPECT_CALL(*cuda_mock_, eventDestroy(end1)).WillOnce(Return(cudaSuccess));
  EXPECT_CALL(*cuda_mock_, eventDestroy(start2)).Times(0);
  EXPECT_CALL(*cuda_mock_, eventDestroy(end2)).Times(0);

  ON_CALL(*cuda_mock_, eventQuery(start2))
      .WillByDefault(Return(cudaErrorNotReady));
  ON_CALL(*cuda_mock_, eventQuery(end2))
      .WillByDefault(Return(cudaErrorNotReady));

  comm->checkGraphEvents();

  ::testing::Mock::VerifyAndClearExpectations(cuda_mock_.get());

  // finalize should destroy graph 2's events
  setupFinalizeExpectations(*comm);
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

  // destroyAll (via finalize) should destroy events regardless of released flag
  std::vector<cudaEvent_t> destroyed_events;
  EXPECT_CALL(*cuda_mock_, eventDestroy(_))
      .WillRepeatedly(DoAll(
          [&destroyed_events](cudaEvent_t event) {
            destroyed_events.push_back(event);
          },
          Return(cudaSuccess)));
  EXPECT_CALL(*cuda_mock_, free(_)).WillOnce(Return(cudaSuccess));
  EXPECT_CALL(*cuda_mock_, streamDestroy(_)).WillOnce(Return(cudaSuccess));
  EXPECT_CALL(*nccl_mock_, commDestroy(_)).WillOnce(Return(ncclSuccess));

  comm->finalize();

  EXPECT_TRUE(
      std::find(
          destroyed_events.begin(), destroyed_events.end(), events.start) !=
      destroyed_events.end())
      << "start event was not destroyed by destroyAll";
  EXPECT_TRUE(
      std::find(destroyed_events.begin(), destroyed_events.end(), events.end) !=
      destroyed_events.end())
      << "end event was not destroyed by destroyAll";
}

} // namespace torch::comms::test
