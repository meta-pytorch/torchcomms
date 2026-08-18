// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <cstdlib>

#include "comms/torchcomms/BackendWrapper.hpp"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/TorchCommOptions.hpp"
#include "comms/torchcomms/fake/TorchCommFake.hpp"
#include "comms/torchcomms/utils/Utils.hpp"

using ::testing::_;
using ::testing::DoAll;
using ::testing::InSequence;
using ::testing::Invoke;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SaveArg;
using ::testing::SetArgPointee;

namespace torch::comms::test {

TEST(TorchCommOptionsTest, EnvToValueImplBool) {
  const std::set<std::string> truthy_values = {"1", "true", "yes", "y"};
  const std::set<std::string> falsy_values = {"0", "false", "no", "n"};
  const char* env_key = "TEST_ENV_KEY";

  for (const auto& value : truthy_values) {
    setenv(env_key, value.c_str(), 1);
    bool result = torch::comms::env_to_value<bool>(env_key, false);
    ASSERT_TRUE(result);
    unsetenv(env_key);
  }
  for (const auto& value : falsy_values) {
    setenv(env_key, value.c_str(), 1);
    bool result = torch::comms::env_to_value<bool>(env_key, false);
    ASSERT_FALSE(result);
    unsetenv(env_key);
  }

  const char* env_value5 = "invalid";
  setenv(env_key, env_value5, 1);
  EXPECT_THROW(
      torch::comms::env_to_value<bool>(env_key, true), std::runtime_error);
  unsetenv(env_key);
}

TEST(TorchCommOptionsTest, StringToBool) {
  // Test true values with different case combinations
  EXPECT_TRUE(torch::comms::string_to_bool("1"));
  EXPECT_TRUE(torch::comms::string_to_bool("true"));
  EXPECT_TRUE(torch::comms::string_to_bool("TRUE"));
  EXPECT_TRUE(torch::comms::string_to_bool("True"));
  EXPECT_TRUE(torch::comms::string_to_bool("yes"));
  EXPECT_TRUE(torch::comms::string_to_bool("YES"));
  EXPECT_TRUE(torch::comms::string_to_bool("Yes"));
  EXPECT_TRUE(torch::comms::string_to_bool("y"));
  EXPECT_TRUE(torch::comms::string_to_bool("Y"));

  // Test false values with different case combinations
  EXPECT_FALSE(torch::comms::string_to_bool("0"));
  EXPECT_FALSE(torch::comms::string_to_bool("false"));
  EXPECT_FALSE(torch::comms::string_to_bool("FALSE"));
  EXPECT_FALSE(torch::comms::string_to_bool("False"));
  EXPECT_FALSE(torch::comms::string_to_bool("no"));
  EXPECT_FALSE(torch::comms::string_to_bool("NO"));
  EXPECT_FALSE(torch::comms::string_to_bool("No"));
  EXPECT_FALSE(torch::comms::string_to_bool("n"));
  EXPECT_FALSE(torch::comms::string_to_bool("N"));

  // Test invalid values that should throw exceptions
  EXPECT_THROW(torch::comms::string_to_bool(""), std::runtime_error);
  EXPECT_THROW(torch::comms::string_to_bool("invalid"), std::runtime_error);
  EXPECT_THROW(torch::comms::string_to_bool("2"), std::runtime_error);
  EXPECT_THROW(torch::comms::string_to_bool("truee"), std::runtime_error);
  EXPECT_THROW(torch::comms::string_to_bool("yess"), std::runtime_error);
  EXPECT_THROW(torch::comms::string_to_bool("nope"), std::runtime_error);
  EXPECT_THROW(torch::comms::string_to_bool("falsey"), std::runtime_error);
}

TEST(TorchCommOptionsTest, OptionsBaseDefaults) {
  AllGatherOptions gather;
  EXPECT_EQ(gather.timeout, kNoTimeout);
  EXPECT_TRUE(gather.hints.empty());

  // Options with extra fields keep the shared defaults from the base.
  SendOptions send;
  EXPECT_EQ(send.timeout, kNoTimeout);
  EXPECT_EQ(send.tag, 0);

  RecvOptions recv;
  EXPECT_EQ(recv.timeout, kNoTimeout);
  EXPECT_EQ(recv.tag, 0);

  // Window options share the same base defaults.
  PutOptions put;
  EXPECT_EQ(put.timeout, kNoTimeout);
  EXPECT_TRUE(put.hints.empty());
}

TEST(TorchCommOptionsTest, CommOptionsGetHint) {
  CommOptions options;
  options.hints["enabled"] = "yes";
  options.hints["count"] = "12";
  options.hints["name"] = "value";

  EXPECT_TRUE(options.getHint<bool>("enabled", false));
  EXPECT_EQ(options.getHint<size_t>("count", 0), 12U);
  EXPECT_EQ(options.getHint<std::string>("name", ""), "value");
  EXPECT_EQ(options.getHint<size_t>("missing", 3), 3U);
}

namespace {
constexpr const char* kBackendName = "fake_test";
constexpr const char* kBackendEnvKey = "TORCHCOMMS_BACKEND_LIB_PATH_FAKE_TEST";
} // namespace

// Verifies the behavior the tag field exists for: BackendWrapper::send/recv
// must thread the c10d `tag` into SendOptions/RecvOptions and pass it down to
// the backend. Uses the fake backend to capture the options actually received.
class BackendWrapperTagTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* lib_path = std::getenv("FAKE_TEST_BACKEND_LIB_PATH");
    ASSERT_NE(lib_path, nullptr) << "FAKE_TEST_BACKEND_LIB_PATH not set";
    setenv(kBackendEnvKey, lib_path, 1);

    comm_ = new_comm(kBackendName, at::Device(at::kCPU), "tag_test");
    ASSERT_NE(comm_, nullptr);
    // Widen the world so send/recv rank validation accepts non-zero peers.
    auto* fake = getFakeBackend();
    ASSERT_NE(fake, nullptr);
    fake->setSize(4);
    wrapper_ = c10::make_intrusive<BackendWrapper>(comm_);
  }

  void TearDown() override {
    wrapper_.reset();
    comm_.reset();
    unsetenv(kBackendEnvKey);
  }

  // Callers must ASSERT_NE the result before use: ASSERT_* cannot live in this
  // non-void helper, so a failed cast would otherwise fall through to a null
  // dereference instead of a clean gtest failure.
  TorchCommFake* getFakeBackend() {
    return dynamic_cast<TorchCommFake*>(comm_->getBackendImpl().get());
  }

  std::shared_ptr<TorchComm> comm_;
  c10::intrusive_ptr<BackendWrapper> wrapper_;
};

TEST_F(BackendWrapperTagTest, SendThreadsTagToBackend) {
  std::vector<at::Tensor> tensors = {at::ones({4}, at::kFloat)};
  wrapper_->send(tensors, /*dstRank=*/1, /*tag=*/123);

  auto* fake = getFakeBackend();
  ASSERT_NE(fake, nullptr);
  ASSERT_TRUE(fake->getLastSendOptionsForTest().has_value());
  EXPECT_EQ(fake->getLastSendOptionsForTest()->tag, 123);
  EXPECT_EQ(fake->getLastSendDstForTest(), 1);
}

TEST_F(BackendWrapperTagTest, RecvThreadsTagToBackend) {
  std::vector<at::Tensor> tensors = {at::empty({4}, at::kFloat)};
  wrapper_->recv(tensors, /*srcRank=*/2, /*tag=*/456);

  auto* fake = getFakeBackend();
  ASSERT_NE(fake, nullptr);
  ASSERT_TRUE(fake->getLastRecvOptionsForTest().has_value());
  EXPECT_EQ(fake->getLastRecvOptionsForTest()->tag, 456);
  EXPECT_EQ(fake->getLastRecvSrcForTest(), 2);
}

#ifdef C10D_BACKEND_HAS_WINDOW
TEST_F(BackendWrapperTagTest, WindowOperationsDelegateToTorchComm) {
  EXPECT_TRUE(wrapper_->supportsWindow());

  auto tensor = at::ones({4}, at::kFloat);
  auto window = wrapper_->new_window(tensor);
  ASSERT_NE(window, nullptr);

  auto putWork = window->put(
      tensor,
      /*dstRank=*/0,
      /*targetOffsetNelems=*/0,
      /*asyncOp=*/false);
  ASSERT_NE(putWork, nullptr);
  EXPECT_TRUE(putWork->wait());

  auto signalWork = window->signal(/*peerRank=*/0, /*asyncOp=*/false);
  ASSERT_NE(signalWork, nullptr);
  EXPECT_TRUE(signalWork->wait());

  auto waitWork = window->wait_signal(/*peerRank=*/0, /*asyncOp=*/false);
  ASSERT_NE(waitWork, nullptr);
  EXPECT_TRUE(waitWork->wait());

  EXPECT_FALSE(window->map_remote_tensor(/*rank=*/0).defined());
  EXPECT_EQ(
      window->get_attr(/*peerRank=*/0).access_type,
      c10d::WindowAccessType::UNIFIED);

  EXPECT_NO_THROW(window->tensor_deregister());
}
#endif

} // namespace torch::comms::test
