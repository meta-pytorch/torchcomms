// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comms/torchcomms/TorchComm.hpp>
#include <comms/torchcomms/TorchCommFactory.hpp>
#include <comms/torchcomms/fake/TorchCommFake.hpp>
#include <gtest/gtest.h>
#include <cstdlib>

namespace torch::comms {

namespace {
constexpr const char* kBackendName = "fake_test";
constexpr const char* kBackendEnvKey = "TORCHCOMMS_BACKEND_LIB_PATH_FAKE_TEST";
} // namespace

class TorchCommAbortTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* lib_path = std::getenv("FAKE_TEST_BACKEND_LIB_PATH");
    ASSERT_NE(lib_path, nullptr) << "FAKE_TEST_BACKEND_LIB_PATH not set";
    setenv(kBackendEnvKey, lib_path, 1);

    comm_ = new_comm(kBackendName, at::Device(at::kCPU), "abort_test");
    ASSERT_NE(comm_, nullptr);
  }

  void TearDown() override {
    comm_.reset();
    unsetenv(kBackendEnvKey);
  }

  std::shared_ptr<TorchComm> comm_;
};

TEST_F(TorchCommAbortTest, IsAbortSupportedDefaultsFalse) {
  EXPECT_FALSE(comm_->isAbortSupported());
}

TEST_F(TorchCommAbortTest, IsAbortedDefaultsFalse) {
  EXPECT_FALSE(comm_->isAborted());
}

TEST_F(TorchCommAbortTest, AbortIsNoOpByDefault) {
  EXPECT_NO_THROW(comm_->abort());
  EXPECT_FALSE(comm_->isAborted());
}

TEST_F(TorchCommAbortTest, IsAbortSupportedReturnsTrue) {
  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm_->getBackendImpl());
  ASSERT_NE(backend, nullptr);

  backend->enableAbort();
  EXPECT_TRUE(comm_->isAbortSupported());
}

TEST_F(TorchCommAbortTest, AbortSetsAbortedStateWhenEnabled) {
  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm_->getBackendImpl());
  ASSERT_NE(backend, nullptr);

  backend->enableAbort();
  EXPECT_FALSE(comm_->isAborted());

  comm_->abort();
  EXPECT_TRUE(comm_->isAborted());
  EXPECT_EQ(comm_->getAbortInfo(), AbortInfo{});
}

TEST_F(TorchCommAbortTest, ContextualAbortFallsBackForLegacyBackend) {
  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm_->getBackendImpl());
  ASSERT_NE(backend, nullptr);
  backend->enableAbort();

  comm_->abort(
      AbortInfo{
          .reason = AbortReason::NETWORK_ERROR,
          .context = "ignored by legacy backend",
      });

  EXPECT_TRUE(comm_->isAborted());
  EXPECT_EQ(comm_->getAbortInfo(), (AbortInfo{.reason = AbortReason::ABORTED}));
}

TEST_F(TorchCommAbortTest, ConcreteBackendExposesContextualAbortFallback) {
  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm_->getBackendImpl());
  ASSERT_NE(backend, nullptr);
  backend->enableAbort();

  backend->abort(
      AbortInfo{
          .reason = AbortReason::NETWORK_ERROR,
          .context = "ignored by legacy backend",
      });

  EXPECT_TRUE(backend->isAborted());
  EXPECT_EQ(
      backend->getAbortInfo(), (AbortInfo{.reason = AbortReason::ABORTED}));
}

TEST_F(TorchCommAbortTest, ConcreteBackendRejectsNonTerminalReason) {
  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm_->getBackendImpl());
  ASSERT_NE(backend, nullptr);
  backend->enableAbort();

  EXPECT_THROW(
      backend->abort(AbortInfo{.reason = AbortReason::NONE}),
      std::invalid_argument);
  EXPECT_FALSE(backend->isAborted());
}

TEST_F(TorchCommAbortTest, ContextualAbortRejectsNonTerminalReason) {
  EXPECT_THROW(
      comm_->abort(AbortInfo{.reason = AbortReason::NONE}),
      std::invalid_argument);
  EXPECT_THROW(
      comm_->abort(AbortInfo{.reason = static_cast<AbortReason>(99)}),
      std::invalid_argument);
  EXPECT_FALSE(comm_->isAborted());
}

TEST_F(TorchCommAbortTest, SetTimeoutDelegatesToBackend) {
  constexpr std::chrono::milliseconds kTimeout{1234};
  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm_->getBackendImpl());
  ASSERT_NE(backend, nullptr);

  comm_->setTimeout(kTimeout);

  EXPECT_EQ(backend->getTimeoutForTest(), kTimeout);
}

TEST_F(TorchCommAbortTest, SetHintsDelegatesToBackend) {
  const std::unordered_map<std::string, std::string> kHints{
      {"step", "42"}, {"attempt", "1"}};
  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm_->getBackendImpl());
  ASSERT_NE(backend, nullptr);

  comm_->setHints(kHints);

  EXPECT_EQ(backend->getHintsForTest(), kHints);
}

} // namespace torch::comms
