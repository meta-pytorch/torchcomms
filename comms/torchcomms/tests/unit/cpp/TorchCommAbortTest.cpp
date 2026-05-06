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

TEST_F(TorchCommAbortTest, AbortEnabledDefaultsFalse) {
  EXPECT_FALSE(comm_->abortEnabled());
}

TEST_F(TorchCommAbortTest, IsAbortedDefaultsFalse) {
  EXPECT_FALSE(comm_->isAborted());
}

TEST_F(TorchCommAbortTest, AbortIsNoOpByDefault) {
  EXPECT_NO_THROW(comm_->abort());
  EXPECT_FALSE(comm_->isAborted());
}

TEST_F(TorchCommAbortTest, AbortEnabledReturnsTrue) {
  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm_->getBackendImpl());
  ASSERT_NE(backend, nullptr);

  backend->enableAbort();
  EXPECT_TRUE(comm_->abortEnabled());
}

TEST_F(TorchCommAbortTest, AbortSetsAbortedStateWhenEnabled) {
  auto backend =
      std::dynamic_pointer_cast<TorchCommFake>(comm_->getBackendImpl());
  ASSERT_NE(backend, nullptr);

  backend->enableAbort();
  EXPECT_FALSE(comm_->isAborted());

  comm_->abort();
  EXPECT_TRUE(comm_->isAborted());
}

} // namespace torch::comms
