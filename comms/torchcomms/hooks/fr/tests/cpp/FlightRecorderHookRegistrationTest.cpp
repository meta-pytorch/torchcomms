// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comms/torchcomms/TorchComm.hpp>
#include <comms/torchcomms/TorchCommOptions.hpp>
#include <comms/torchcomms/fake/TorchCommFake.hpp>
#include <comms/torchcomms/hooks/fr/FlightRecorder.hpp>

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace torch::comms::fr {

namespace {
constexpr const char* kBackendName = "fake_test";
constexpr const char* kBackendEnvKey = "TORCHCOMMS_BACKEND_LIB_PATH_FAKE_TEST";
constexpr const char* kCommName = "dynamic_flight_recorder";
constexpr const char* kHealthCheckWaitEnvKey = "TORCHCOMM_HEALTH_CHECK_WAIT_MS";
} // namespace

class FlightRecorderHookRegistrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (const char* backendLibPath = std::getenv(kBackendEnvKey)) {
      previousBackendLibPath_ = backendLibPath;
    }
    if (const char* waitMs = std::getenv(kHealthCheckWaitEnvKey)) {
      previousHealthCheckWaitMs_ = waitMs;
    }

    const char* libPath = std::getenv("FAKE_TEST_BACKEND_LIB_PATH");
    ASSERT_NE(libPath, nullptr) << "FAKE_TEST_BACKEND_LIB_PATH not set";
    setenv(kBackendEnvKey, libPath, 1);
    setenv(kHealthCheckWaitEnvKey, "0", 1);
  }

  void TearDown() override {
    if (previousBackendLibPath_) {
      setenv(kBackendEnvKey, previousBackendLibPath_->c_str(), 1);
    } else {
      unsetenv(kBackendEnvKey);
    }
    if (previousHealthCheckWaitMs_) {
      setenv(kHealthCheckWaitEnvKey, previousHealthCheckWaitMs_->c_str(), 1);
    } else {
      unsetenv(kHealthCheckWaitEnvKey);
    }
  }

 private:
  std::optional<std::string> previousBackendLibPath_;
  std::optional<std::string> previousHealthCheckWaitMs_;
};

TEST_F(
    FlightRecorderHookRegistrationTest,
    DefersMembershipUntilFirstCollectiveForUninitializedComm) {
  CommOptions options;
  options.enable_reconfigure = true;
  auto comm = new_comm(kBackendName, at::Device(at::kCPU), kCommName, options);
  ASSERT_NE(comm, nullptr);

  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm->getBackendImpl());
  ASSERT_NE(backend, nullptr);
  backend->finalize();
  ASSERT_FALSE(backend->isInitialized());

  auto recorder = std::make_shared<FlightRecorderHook>(100, true);
  EXPECT_NO_THROW(recorder->registerWithComm(comm));
  EXPECT_TRUE(recorder->isEnabled());
  EXPECT_EQ(recorder->size(), 0);
  EXPECT_EQ(recorder->dump_json().find(kCommName), std::string::npos);

  ReconfigureOptions reconfigureOptions;
  reconfigureOptions.handles = std::vector<InitHandle>{"fake:0"};
  ASSERT_TRUE(comm->reconfigure(reconfigureOptions)->isCompleted());

  auto tensor = at::ones({2, 2}, at::kFloat);
  EXPECT_NO_THROW(comm->all_reduce(tensor, ReduceOp::SUM, /*async_op=*/false));
  EXPECT_EQ(recorder->size(), 1);
  EXPECT_NE(recorder->dump_json().find(kCommName), std::string::npos);

  comm->finalize();
}

TEST_F(
    FlightRecorderHookRegistrationTest,
    UnavailableMembershipDoesNotBreakPreHook) {
  CommOptions options;
  options.enable_reconfigure = true;
  auto comm = new_comm(kBackendName, at::Device(at::kCPU), kCommName, options);
  ASSERT_NE(comm, nullptr);

  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm->getBackendImpl());
  ASSERT_NE(backend, nullptr);
  backend->finalize();

  auto recorder = std::make_shared<FlightRecorderHook>(100, true);
  recorder->registerWithComm(comm);
  auto tensor = at::ones({2, 2}, at::kFloat);
  EXPECT_NO_THROW(comm->all_reduce(tensor, ReduceOp::SUM, /*async_op=*/false));
  EXPECT_EQ(recorder->getRecorder()->getRank(), -1);
}

TEST_F(
    FlightRecorderHookRegistrationTest,
    RecordsMembershipOnAbortBeforeFirstCollective) {
  CommOptions options;
  options.enable_reconfigure = true;
  auto comm = new_comm(kBackendName, at::Device(at::kCPU), kCommName, options);
  ASSERT_NE(comm, nullptr);

  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm->getBackendImpl());
  ASSERT_NE(backend, nullptr);
  backend->finalize();

  auto recorder = std::make_shared<FlightRecorderHook>(100, true);
  recorder->registerWithComm(comm);
  EXPECT_EQ(recorder->getRecorder()->getRank(), -1);

  ReconfigureOptions reconfigureOptions;
  reconfigureOptions.handles = std::vector<InitHandle>{"fake:0"};
  ASSERT_TRUE(comm->reconfigure(reconfigureOptions)->isCompleted());

  backend->triggerAbort();
  EXPECT_EQ(recorder->getRecorder()->getRank(), 0);

  comm->finalize();
}

} // namespace torch::comms::fr
