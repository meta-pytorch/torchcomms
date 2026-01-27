// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comms/torchcomms/TorchComm.hpp>
#include <comms/torchcomms/TorchCommFactory.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>

namespace torch::comms {

namespace {
// Backend name must match the exported symbol in the dummy backend library
constexpr const char* kBackendName = "dummy_test";
constexpr const char* kBackendEnvKey = "TORCHCOMMS_BACKEND_LIB_PATH_DUMMY_TEST";
} // namespace

class TorchCommBackendFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* lib_path = std::getenv("DUMMY_TEST_BACKEND_LIB_PATH");
    ASSERT_NE(lib_path, nullptr) << "DUMMY_TEST_BACKEND_LIB_PATH not set";
    setenv(kBackendEnvKey, lib_path, 1);
  }

  void TearDown() override {
    unsetenv(kBackendEnvKey);
  }

  // Helper to get a unique backend name for error tests (avoids cache)
  static std::string getUniqueBackendName() {
    const auto* test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    return std::string("test_") + test_info->name();
  }
};

TEST_F(TorchCommBackendFactoryTest, CreateGenericBackend) {
  at::Device device(at::kCPU);
  CommOptions options;

  // Test creating generic backend (which loads the dummy backend)
  auto backend = TorchCommFactory::get().create_backend(
      kBackendName, device, "my_comm", options);
  ASSERT_NE(backend, nullptr);

  // Test basic functionality
  EXPECT_EQ(backend->getRank(), 0);
  EXPECT_EQ(backend->getSize(), 1);
  EXPECT_EQ(backend->getDevice().type(), at::kCPU);
  EXPECT_EQ(backend->getBackendName(), "dummy");
}

TEST_F(TorchCommBackendFactoryTest, GenericBackendFunctionality) {
  at::Device device(at::kCPU);
  CommOptions options;

  auto backend = TorchCommFactory::get().create_backend(
      kBackendName, device, "my_comm", options);
  ASSERT_NE(backend, nullptr);

  // Test point-to-point operations
  auto tensor = at::ones({2, 2}, at::kFloat);
  auto send_work = backend->send(tensor, 1, true, SendOptions{});
  ASSERT_NE(send_work, nullptr);
  EXPECT_TRUE(send_work->isCompleted());

  auto recv_work = backend->recv(tensor, 0, true, RecvOptions{});
  ASSERT_NE(recv_work, nullptr);
  EXPECT_TRUE(recv_work->isCompleted());

  // Test collective operations
  auto bcast_work = backend->broadcast(tensor, 0, true, BroadcastOptions{});
  ASSERT_NE(bcast_work, nullptr);
  EXPECT_TRUE(bcast_work->isCompleted());

  auto allreduce_work =
      backend->all_reduce(tensor, ReduceOp::SUM, true, AllReduceOptions{});
  ASSERT_NE(allreduce_work, nullptr);
  EXPECT_TRUE(allreduce_work->isCompleted());

  // Test barrier
  auto barrier_work = backend->barrier(true, BarrierOptions{});
  ASSERT_NE(barrier_work, nullptr);
  EXPECT_TRUE(barrier_work->isCompleted());
}

TEST_F(TorchCommBackendFactoryTest, GenericBackendSplit) {
  at::Device device(at::kCPU);
  CommOptions options;

  auto backend = TorchCommFactory::get().create_backend(
      kBackendName, device, "my_comm", options);
  ASSERT_NE(backend, nullptr);

  ASSERT_EQ(backend->getBackendName(), "dummy");

  // Test split functionality
  std::vector<int> ranks = {0};
  auto split_backend = backend->split(ranks, "test_split_comm");
  ASSERT_NE(split_backend, nullptr);

  EXPECT_EQ(split_backend->getRank(), 0);
  EXPECT_EQ(split_backend->getSize(), 1);
  ASSERT_EQ(split_backend->getBackendName(), "dummy");
}

TEST_F(TorchCommBackendFactoryTest, UnsupportedBackend) {
  at::Device device(at::kCPU);
  CommOptions options;

  // Test unsupported backend
  EXPECT_THROW(
      TorchCommFactory::get().create_backend(
          "unsupported", device, "my_comm", options),
      std::runtime_error);
}

TEST_F(TorchCommBackendFactoryTest, MissingEnvironmentVariable) {
  // Use a unique backend name to avoid hitting the cache from other tests
  std::string unique_backend = getUniqueBackendName();

  at::Device device(at::kCPU);
  CommOptions options;

  // Test that missing environment variable throws error
  EXPECT_THROW(
      TorchCommFactory::get().create_backend(
          unique_backend, device, "my_comm", options),
      std::runtime_error);
}

TEST_F(TorchCommBackendFactoryTest, InvalidLibraryPath) {
  // Use a unique backend name to avoid hitting the cache from other tests
  std::string unique_backend = getUniqueBackendName();
  std::string env_key = "TORCHCOMMS_BACKEND_LIB_PATH_" + unique_backend;
  std::transform(
      env_key.begin(), env_key.end(), env_key.begin(), [](unsigned char c) {
        return std::toupper(c);
      });
  setenv(env_key.c_str(), "/invalid/path/libnonexistent.so", 1);

  at::Device device(at::kCPU);
  CommOptions options;

  // Test that invalid library path throws error
  EXPECT_THROW(
      TorchCommFactory::get().create_backend(
          unique_backend, device, "my_comm", options),
      std::runtime_error);

  unsetenv(env_key.c_str());
}

TEST_F(TorchCommBackendFactoryTest, NewCommIntegration) {
  // Test the init_comm function with the factory
  at::Device device(at::kCPU);
  CommOptions options;
  auto torchcomm = new_comm(kBackendName, device, "my_comm", options);
  ASSERT_NE(torchcomm, nullptr);

  // Test torchcomm functionality
  EXPECT_EQ(torchcomm->getRank(), 0);
  EXPECT_EQ(torchcomm->getSize(), 1);
  EXPECT_EQ(torchcomm->getBackend(), kBackendName);
  EXPECT_EQ(torchcomm->getDevice().type(), at::kCPU);

  // Test operations through torchcomm
  auto tensor = at::ones({2, 2}, at::kFloat);
  auto work =
      torchcomm->all_reduce(tensor, ReduceOp::SUM, true, AllReduceOptions{});
  ASSERT_NE(work, nullptr);
  EXPECT_TRUE(work->isCompleted());
}

} // namespace torch::comms
