// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comms/torchcomms/TorchComm.hpp>
#include <comms/torchcomms/TorchCommFactory.hpp>
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <memory>

namespace torch::comms {

namespace {
// Backend name must match the exported symbol in the fake backend library
constexpr const char* kBackendName = "fake_test";
constexpr const char* kBackendEnvKey = "TORCHCOMMS_BACKEND_LIB_PATH_FAKE_TEST";
constexpr const char* kLegacyBackendName = "legacy_direct_registration_test";
constexpr const char* kLegacyBackendPathEnvKey =
    "LEGACY_DIRECT_REGISTRATION_BACKEND_LIB_PATH";
constexpr const char* kMismatchedDynamicBackendName = "abi_mismatch_test";
constexpr const char* kMismatchedDynamicBackendEnvKey =
    "TORCHCOMMS_BACKEND_LIB_PATH_ABI_MISMATCH_TEST";
} // namespace

class TorchCommBackendFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* lib_path = std::getenv("FAKE_TEST_BACKEND_LIB_PATH");
    ASSERT_NE(lib_path, nullptr) << "FAKE_TEST_BACKEND_LIB_PATH not set";
    setenv(kBackendEnvKey, lib_path, 1);
  }

  void TearDown() override {
    unsetenv(kBackendEnvKey);
    unsetenv(kMismatchedDynamicBackendEnvKey);
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

  // Test creating generic backend (which loads the fake backend)
  auto backend = TorchCommFactory::get().create_backend(
      kBackendName, device, "my_comm", options);
  ASSERT_NE(backend, nullptr);

  // Test basic functionality
  EXPECT_EQ(backend->getRank(), 0);
  EXPECT_EQ(backend->getSize(), 1);
  EXPECT_EQ(backend->getDevice().type(), at::kCPU);
  EXPECT_EQ(backend->getBackendName(), "fake");
}

TEST_F(TorchCommBackendFactoryTest, CurrentExplicitAbiRegistrationSucceeds) {
  const auto backend_name = getUniqueBackendName();

  EXPECT_NO_THROW(
      TorchCommFactory::get().register_backend(
          backend_name,
          []() { return std::shared_ptr<TorchCommBackend>(); },
          TORCHCOMM_BACKEND_ABI_VERSION));
  EXPECT_TRUE(TorchCommFactory::get().is_backend_registered(backend_name));
}

TEST_F(
    TorchCommBackendFactoryTest,
    ExplicitAbiMismatchRejectedBeforeFactoryCall) {
  const auto backend_name = getUniqueBackendName();
  const auto factory_called = std::make_shared<bool>(false);
  TorchCommFactory::get().register_backend(
      backend_name,
      [factory_called]() {
        *factory_called = true;
        return std::shared_ptr<TorchCommBackend>();
      },
      "incompatible-test-version");

  try {
    TorchCommFactory::get().create_backend(
        backend_name, at::Device(at::kCPU), "my_comm", CommOptions{});
    FAIL() << "Expected an ABI mismatch";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find(backend_name), std::string::npos);
    EXPECT_NE(message.find("incompatible-test-version"), std::string::npos);
    EXPECT_NE(message.find(TORCHCOMM_BACKEND_ABI_VERSION), std::string::npos);
  }
  EXPECT_FALSE(*factory_called);
}

TEST_F(
    TorchCommBackendFactoryTest,
    LegacyDirectRegistrationRejectedBeforeFactoryCall) {
  const char* const lib_path = std::getenv(kLegacyBackendPathEnvKey);
  ASSERT_NE(lib_path, nullptr) << kLegacyBackendPathEnvKey << " not set";

  void* const handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(handle, nullptr) << dlerror();

  using FactoryCalledFn = bool (*)();
  auto* const factory_called = reinterpret_cast<FactoryCalledFn>(
      dlsym(handle, "legacy_direct_registration_factory_called"));
  ASSERT_NE(factory_called, nullptr) << dlerror();
  EXPECT_FALSE(factory_called());

  try {
    TorchCommFactory::get().create_backend(
        kLegacyBackendName, at::Device(at::kCPU), "my_comm", CommOptions{});
    FAIL() << "Expected a legacy ABI mismatch";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find(kLegacyBackendName), std::string::npos);
    EXPECT_NE(message.find("legacy-unversioned"), std::string::npos);
    EXPECT_NE(message.find(TORCHCOMM_BACKEND_ABI_VERSION), std::string::npos);
  }
  EXPECT_FALSE(factory_called());
}

TEST_F(TorchCommBackendFactoryTest, GenericLoaderRejectsMismatchedAbi) {
  const char* const lib_path = std::getenv(kLegacyBackendPathEnvKey);
  ASSERT_NE(lib_path, nullptr) << kLegacyBackendPathEnvKey << " not set";
  ASSERT_EQ(setenv(kMismatchedDynamicBackendEnvKey, lib_path, 1), 0);

  void* const handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(handle, nullptr) << dlerror();
  using NewCommCalledFn = bool (*)();
  auto* const new_comm_called = reinterpret_cast<NewCommCalledFn>(
      dlsym(handle, "mismatched_dynamic_new_comm_called"));
  ASSERT_NE(new_comm_called, nullptr) << dlerror();
  EXPECT_FALSE(new_comm_called());

  try {
    TorchCommFactory::get().create_backend(
        kMismatchedDynamicBackendName,
        at::Device(at::kCPU),
        "my_comm",
        CommOptions{});
    FAIL() << "Expected a dynamic-loader ABI mismatch";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("incompatible-test-version"), std::string::npos);
    EXPECT_NE(message.find(TORCHCOMM_BACKEND_ABI_VERSION), std::string::npos);
  }
  EXPECT_FALSE(new_comm_called());
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

  ASSERT_EQ(backend->getBackendName(), "fake");

  // Test split functionality
  std::vector<int> ranks = {0};
  auto split_backend = backend->split(ranks, "test_split_comm");
  ASSERT_NE(split_backend, nullptr);

  EXPECT_EQ(split_backend->getRank(), 0);
  EXPECT_EQ(split_backend->getSize(), 1);
  ASSERT_EQ(split_backend->getBackendName(), "fake");
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
