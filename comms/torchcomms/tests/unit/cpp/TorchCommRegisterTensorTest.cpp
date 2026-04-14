// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <comms/torchcomms/TorchComm.hpp>
#include <comms/torchcomms/TorchCommFactory.hpp>
#include <comms/torchcomms/dummy/TorchCommDummy.hpp>
#include <cstdlib>

namespace torch::comms {

namespace {
constexpr const char* kBackendName = "dummy_test";
constexpr const char* kBackendEnvKey = "TORCHCOMMS_BACKEND_LIB_PATH_DUMMY_TEST";
} // namespace

class TorchCommRegisterTensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* lib_path = std::getenv("DUMMY_TEST_BACKEND_LIB_PATH");
    ASSERT_NE(lib_path, nullptr) << "DUMMY_TEST_BACKEND_LIB_PATH not set";
    setenv(kBackendEnvKey, lib_path, 1);

    comm_ = new_comm(kBackendName, at::Device(at::kCPU), "register_test");
    ASSERT_NE(comm_, nullptr);
  }

  void TearDown() override {
    comm_.reset();
    unsetenv(kBackendEnvKey);
  }

  TorchCommDummy* getDummyBackend() {
    return dynamic_cast<TorchCommDummy*>(comm_->getBackendImpl().get());
  }

  std::shared_ptr<TorchComm> comm_;
};

TEST_F(TorchCommRegisterTensorTest, RegisterDelegatesToBackend) {
  auto tensor = at::ones({1024}, at::kFloat);
  auto* backend = getDummyBackend();
  ASSERT_NE(backend, nullptr);

  EXPECT_FALSE(backend->is_tensor_registered(tensor));
  comm_->tensor_register(tensor);
  EXPECT_TRUE(backend->is_tensor_registered(tensor));
}

TEST_F(TorchCommRegisterTensorTest, DeregisterRemovesRegistration) {
  auto tensor = at::ones({1024}, at::kFloat);
  auto* backend = getDummyBackend();
  ASSERT_NE(backend, nullptr);

  comm_->tensor_register(tensor);
  EXPECT_TRUE(backend->is_tensor_registered(tensor));

  comm_->tensor_deregister(tensor);
  EXPECT_FALSE(backend->is_tensor_registered(tensor));
}

TEST_F(TorchCommRegisterTensorTest, RegisterMultipleTensors) {
  auto t1 = at::ones({1024}, at::kFloat);
  auto t2 = at::zeros({2048}, at::kByte);
  auto* backend = getDummyBackend();
  ASSERT_NE(backend, nullptr);

  comm_->tensor_register(t1);
  comm_->tensor_register(t2);
  EXPECT_TRUE(backend->is_tensor_registered(t1));
  EXPECT_TRUE(backend->is_tensor_registered(t2));

  comm_->tensor_deregister(t1);
  EXPECT_FALSE(backend->is_tensor_registered(t1));
  EXPECT_TRUE(backend->is_tensor_registered(t2));
}

} // namespace torch::comms
