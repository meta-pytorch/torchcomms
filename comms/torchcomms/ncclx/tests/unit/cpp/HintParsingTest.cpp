// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/tests/unit/cpp/TorchCommNCCLXTestBase.hpp"

namespace torch::comms::test {

class HintParsingTest : public TorchCommNCCLXTest {
 protected:
  CommOptions createOptions() {
    CommOptions options;
    options.timeout = std::chrono::milliseconds(2000);
    options.abort_process_on_timeout_or_error = false;
    options.store = store_;
    return options;
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

TEST_F(HintParsingTest, DefaultConfigValues) {
  setupCCAExpectations(1, 2, 1);
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  const auto options = createOptions();
  comm->init(*device_, "test_defaults", options);

  EXPECT_FALSE(comm->testGetHighPriorityStream());
  EXPECT_EQ(comm->testGetMaxEventPoolSize(), 1000);
  EXPECT_EQ(comm->testGetGarbageCollectIntervalMs(), 100);
  EXPECT_TRUE(comm->testGetEnableCudaGraphSupport());
  EXPECT_EQ(comm->testGetGraphTimeoutCheckIntervalMs(), 1000);

  setupFinalizeExpectations(*comm);
}

TEST_F(HintParsingTest, MaxEventPoolSizeHint) {
  setupCCAExpectations(1, 2, 1);
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createOptions();
  options.hints["max_event_pool_size"] = "500";
  comm->init(*device_, "test_max_event_pool", options);

  EXPECT_EQ(comm->testGetMaxEventPoolSize(), 500);

  setupFinalizeExpectations(*comm);
}

TEST_F(HintParsingTest, GarbageCollectIntervalMsHint) {
  setupCCAExpectations(1, 2, 1);
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createOptions();
  options.hints["garbage_collect_interval_ms"] = "200";
  comm->init(*device_, "test_gc_interval", options);

  EXPECT_EQ(comm->testGetGarbageCollectIntervalMs(), 200);

  setupFinalizeExpectations(*comm);
}

TEST_F(HintParsingTest, EnableCudaGraphSupportHint) {
  setupCCAExpectations(1, 2, 1);
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createOptions();
  options.hints["enable_cuda_graph_support"] = "false";
  comm->init(*device_, "test_cuda_graph_off", options);

  EXPECT_FALSE(comm->testGetEnableCudaGraphSupport());

  setupFinalizeExpectations(*comm);
}

TEST_F(HintParsingTest, GraphTimeoutCheckIntervalMsHint) {
  setupCCAExpectations(1, 2, 1);
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createOptions();
  options.hints["graph_timeout_check_interval_ms"] = "2000";
  comm->init(*device_, "test_graph_timeout_interval", options);

  EXPECT_EQ(comm->testGetGraphTimeoutCheckIntervalMs(), 2000);

  setupFinalizeExpectations(*comm);
}

TEST_F(HintParsingTest, AllHintsCombined) {
  setupCCAExpectations(1, 2, 1);
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  // high_priority_stream hint triggers getStreamPriorityRange call
  ON_CALL(*cuda_mock_, getStreamPriorityRange(_, _))
      .WillByDefault(DoAll(
          SetArgPointee<0>(0), SetArgPointee<1>(-10), Return(cudaSuccess)));

  auto options = createOptions();
  options.hints["max_event_pool_size"] = "2000";
  options.hints["garbage_collect_interval_ms"] = "50";
  options.hints["enable_cuda_graph_support"] = "false";
  options.hints["high_priority_stream"] = "true";
  options.hints["graph_timeout_check_interval_ms"] = "3000";
  comm->init(*device_, "test_all_hints", options);

  EXPECT_TRUE(comm->testGetHighPriorityStream());
  EXPECT_EQ(comm->testGetMaxEventPoolSize(), 2000);
  EXPECT_EQ(comm->testGetGarbageCollectIntervalMs(), 50);
  EXPECT_FALSE(comm->testGetEnableCudaGraphSupport());
  EXPECT_EQ(comm->testGetGraphTimeoutCheckIntervalMs(), 3000);

  setupFinalizeExpectations(*comm);
}

TEST_F(HintParsingTest, UnknownHintsIgnored) {
  setupCCAExpectations(1, 2, 1);
  auto comm = createMockedTorchComm();
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  auto options = createOptions();
  options.hints["some_other_backend::key"] = "value";
  options.hints["unrelated_key"] = "42";
  comm->init(*device_, "test_non_prefixed", options);

  // Defaults unchanged
  EXPECT_EQ(comm->testGetMaxEventPoolSize(), 1000);
  EXPECT_EQ(comm->testGetGarbageCollectIntervalMs(), 100);
  EXPECT_TRUE(comm->testGetEnableCudaGraphSupport());

  setupFinalizeExpectations(*comm);
}

} // namespace torch::comms::test
