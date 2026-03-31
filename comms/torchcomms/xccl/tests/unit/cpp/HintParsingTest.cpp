#include "comms/torchcomms/xccl/tests/unit/cpp/TorchCommXCCLTestBase.hpp"

namespace torch::comms::test {

class HintParsingTest : public TorchCommXCCLTest {
 protected:
  CommOptions createOptions() {
    CommOptions options;
    options.timeout = std::chrono::milliseconds(2000);
    options.abort_process_on_timeout_or_error = false;
    options.store = store_;
    return options;
  }
};

TEST_F(HintParsingTest, DefaultConfigValues) {
  setupRankAndSize(0, 2);
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();

  auto comm = createMockedTorchComm();
  const auto options = createOptions();
  comm->init(*device_, "test_defaults", options);

  EXPECT_FALSE(comm->testGetHighPriorityStream());
  EXPECT_EQ(comm->testGetMaxEventPoolSize(), 1000);

  comm->finalize();
}

TEST_F(HintParsingTest, HighPriorityStreamHint) {
  setupRankAndSize(0, 2);
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();

  auto comm = createMockedTorchComm();
  auto options = createOptions();
  options.hints["high_priority_stream"] = "true";
  comm->init(*device_, "test_high_priority", options);

  EXPECT_TRUE(comm->testGetHighPriorityStream());

  comm->finalize();
}

TEST_F(HintParsingTest, MaxEventPoolSizeHint) {
  setupRankAndSize(0, 2);
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();

  auto comm = createMockedTorchComm();
  auto options = createOptions();
  options.hints["max_event_pool_size"] = "500";
  comm->init(*device_, "test_max_event_pool", options);

  EXPECT_EQ(comm->testGetMaxEventPoolSize(), 500);

  comm->finalize();
}

TEST_F(HintParsingTest, AllHintsCombined) {
  setupRankAndSize(0, 2);
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();

  auto comm = createMockedTorchComm();
  auto options = createOptions();
  options.hints["high_priority_stream"] = "true";
  options.hints["max_event_pool_size"] = "2000";
  comm->init(*device_, "test_all_hints", options);

  EXPECT_TRUE(comm->testGetHighPriorityStream());
  EXPECT_EQ(comm->testGetMaxEventPoolSize(), 2000);

  comm->finalize();
}

TEST_F(HintParsingTest, UnknownHintsIgnored) {
  setupRankAndSize(0, 2);
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();

  auto comm = createMockedTorchComm();
  auto options = createOptions();
  options.hints["some_other_backend::key"] = "value";
  options.hints["unrelated_key"] = "42";
  comm->init(*device_, "test_unknown_hints", options);

  // Defaults unchanged
  EXPECT_FALSE(comm->testGetHighPriorityStream());
  EXPECT_EQ(comm->testGetMaxEventPoolSize(), 1000);

  comm->finalize();
}

} // namespace torch::comms::test
