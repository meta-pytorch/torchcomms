// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comms/torchcomms/TorchComm.hpp>
#include <comms/torchcomms/TorchCommDummy.hpp>
#include <comms/torchcomms/TorchCommFactory.hpp>
#include <gtest/gtest.h>
#include <cstdlib>
#include <vector>

namespace torch::comms {

namespace {
constexpr const char* kBackendName = "dummy_test";
constexpr const char* kBackendEnvKey = "TORCHCOMMS_BACKEND_LIB_PATH_DUMMY_TEST";
} // namespace

class TorchCommHooksTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* lib_path = std::getenv("DUMMY_TEST_BACKEND_LIB_PATH");
    ASSERT_NE(lib_path, nullptr) << "DUMMY_TEST_BACKEND_LIB_PATH not set";
    setenv(kBackendEnvKey, lib_path, 1);
  }

  void TearDown() override {
    unsetenv(kBackendEnvKey);
  }
};

TEST_F(TorchCommHooksTest, PreAndPostHookCalledAfterRegistration) {
  at::Device device(at::kCPU);
  auto torchcomm = new_comm(kBackendName, device, "test_comm", {});
  ASSERT_NE(torchcomm, nullptr);

  std::vector<OpName> preHookCalls;
  std::vector<OpName> postHookCalls;

  auto preHandle =
      torchcomm->registerPreHook([&preHookCalls](TorchComm::PreHookArgs args) {
        preHookCalls.push_back(args.name);
      });

  auto postHandle = torchcomm->registerPostHook(
      [&postHookCalls](TorchComm::PostHookArgs args) {
        postHookCalls.push_back(args.name);
      });

  auto tensor = at::ones({2, 2}, at::kFloat);
  torchcomm->all_reduce(tensor, ReduceOp::SUM, true);

  ASSERT_EQ(preHookCalls.size(), 1);
  EXPECT_EQ(preHookCalls[0], OpName::all_reduce);

  ASSERT_EQ(postHookCalls.size(), 1);
  EXPECT_EQ(postHookCalls[0], OpName::all_reduce);
}

TEST_F(TorchCommHooksTest, PreAndPostHookOpIdIncreases) {
  at::Device device(at::kCPU);
  auto torchcomm = new_comm(kBackendName, device, "test_comm", {});
  ASSERT_NE(torchcomm, nullptr);

  std::vector<size_t> preOpIds;
  std::vector<size_t> postOpIds;

  auto preHandle =
      torchcomm->registerPreHook([&preOpIds](TorchComm::PreHookArgs args) {
        preOpIds.push_back(args.op_id);
      });

  auto postHandle =
      torchcomm->registerPostHook([&postOpIds](TorchComm::PostHookArgs args) {
        postOpIds.push_back(args.op_id);
      });

  auto tensor = at::ones({2, 2}, at::kFloat);

  torchcomm->all_reduce(tensor, ReduceOp::SUM, true);
  torchcomm->barrier(true);
  torchcomm->broadcast(tensor, 0, true);

  ASSERT_EQ(preOpIds.size(), 3);
  ASSERT_EQ(postOpIds.size(), 3);

  EXPECT_LT(preOpIds[0], preOpIds[1]);
  EXPECT_LT(preOpIds[1], preOpIds[2]);

  EXPECT_LT(postOpIds[0], postOpIds[1]);
  EXPECT_LT(postOpIds[1], postOpIds[2]);

  EXPECT_EQ(preOpIds[0], postOpIds[0]);
  EXPECT_EQ(preOpIds[1], postOpIds[1]);
  EXPECT_EQ(preOpIds[2], postOpIds[2]);
}

TEST_F(TorchCommHooksTest, PreAndPostHookNotCalledAfterRemoval) {
  at::Device device(at::kCPU);
  auto torchcomm = new_comm(kBackendName, device, "test_comm", {});
  ASSERT_NE(torchcomm, nullptr);

  int preHookCallCount = 0;
  int postHookCallCount = 0;

  auto preHandle = torchcomm->registerPreHook(
      [&preHookCallCount](TorchComm::PreHookArgs) { preHookCallCount++; });

  auto postHandle = torchcomm->registerPostHook(
      [&postHookCallCount](TorchComm::PostHookArgs) { postHookCallCount++; });

  auto tensor = at::ones({2, 2}, at::kFloat);
  auto work = torchcomm->all_reduce(tensor, ReduceOp::SUM, true);

  EXPECT_EQ(preHookCallCount, 1);
  EXPECT_EQ(postHookCallCount, 1);

  preHandle->remove();
  postHandle->remove();

  torchcomm->all_reduce(tensor, ReduceOp::SUM, true);

  EXPECT_EQ(preHookCallCount, 1);
  EXPECT_EQ(postHookCallCount, 1);
}

TEST_F(TorchCommHooksTest, MultiplePreAndPostHooksRegistered) {
  at::Device device(at::kCPU);
  auto torchcomm = new_comm(kBackendName, device, "test_comm", {});
  ASSERT_NE(torchcomm, nullptr);

  int preHook1CallCount = 0;
  int preHook2CallCount = 0;
  int postHook1CallCount = 0;
  int postHook2CallCount = 0;

  auto preHandle1 = torchcomm->registerPreHook(
      [&preHook1CallCount](TorchComm::PreHookArgs) { preHook1CallCount++; });

  auto preHandle2 = torchcomm->registerPreHook(
      [&preHook2CallCount](TorchComm::PreHookArgs) { preHook2CallCount++; });

  auto postHandle1 = torchcomm->registerPostHook(
      [&postHook1CallCount](TorchComm::PostHookArgs) { postHook1CallCount++; });

  auto postHandle2 = torchcomm->registerPostHook(
      [&postHook2CallCount](TorchComm::PostHookArgs) { postHook2CallCount++; });

  auto tensor = at::ones({2, 2}, at::kFloat);
  torchcomm->all_reduce(tensor, ReduceOp::SUM, true);

  EXPECT_EQ(preHook1CallCount, 1);
  EXPECT_EQ(preHook2CallCount, 1);
  EXPECT_EQ(postHook1CallCount, 1);
  EXPECT_EQ(postHook2CallCount, 1);
}

TEST_F(
    TorchCommHooksTest,
    PreAndPostHookOpIdIncreasesAcrossDifferentOperations) {
  at::Device device(at::kCPU);
  auto torchcomm = new_comm(kBackendName, device, "test_comm", {});
  ASSERT_NE(torchcomm, nullptr);

  std::vector<std::pair<OpName, size_t>> preHookCalls;
  std::vector<std::pair<OpName, size_t>> postHookCalls;

  auto preHandle =
      torchcomm->registerPreHook([&preHookCalls](TorchComm::PreHookArgs args) {
        preHookCalls.push_back({args.name, args.op_id});
      });

  auto postHandle = torchcomm->registerPostHook(
      [&postHookCalls](TorchComm::PostHookArgs args) {
        postHookCalls.push_back({args.name, args.op_id});
      });

  auto tensor = at::ones({2, 2}, at::kFloat);

  torchcomm->all_reduce(tensor, ReduceOp::SUM, true);
  torchcomm->barrier(true);
  torchcomm->broadcast(tensor, 0, true);

  ASSERT_EQ(preHookCalls.size(), 3);
  ASSERT_EQ(postHookCalls.size(), 3);

  EXPECT_EQ(preHookCalls[0].first, OpName::all_reduce);
  EXPECT_EQ(preHookCalls[1].first, OpName::barrier);
  EXPECT_EQ(preHookCalls[2].first, OpName::broadcast);

  EXPECT_EQ(postHookCalls[0].first, OpName::all_reduce);
  EXPECT_EQ(postHookCalls[1].first, OpName::barrier);
  EXPECT_EQ(postHookCalls[2].first, OpName::broadcast);

  EXPECT_LT(preHookCalls[0].second, preHookCalls[1].second);
  EXPECT_LT(preHookCalls[1].second, preHookCalls[2].second);

  EXPECT_LT(postHookCalls[0].second, postHookCalls[1].second);
  EXPECT_LT(postHookCalls[1].second, postHookCalls[2].second);

  EXPECT_EQ(preHookCalls[0].second, postHookCalls[0].second);
  EXPECT_EQ(preHookCalls[1].second, postHookCalls[1].second);
  EXPECT_EQ(preHookCalls[2].second, postHookCalls[2].second);
}

TEST_F(TorchCommHooksTest, AbortHookNotCalledAfterRemoval) {
  at::Device device(at::kCPU);
  auto torchcomm = new_comm(kBackendName, device, "test_comm", {});
  ASSERT_NE(torchcomm, nullptr);

  auto backend =
      std::dynamic_pointer_cast<TorchCommDummy>(torchcomm->unsafeGetBackend());
  ASSERT_NE(backend, nullptr);

  int abortHookCallCount = 0;
  auto handle = torchcomm->registerAbortHook(
      [&abortHookCallCount]() { abortHookCallCount++; });

  // Trigger abort - hook should be called
  backend->triggerAbort();
  EXPECT_EQ(abortHookCallCount, 1);

  // Remove the hook
  handle->remove();

  // Trigger abort again - hook should NOT be called
  backend->triggerAbort();
  EXPECT_EQ(abortHookCallCount, 1);
}

TEST_F(TorchCommHooksTest, AbortHookInvoked) {
  at::Device device(at::kCPU);
  auto torchcomm = new_comm(kBackendName, device, "test_comm", {});
  ASSERT_NE(torchcomm, nullptr);

  auto backend =
      std::dynamic_pointer_cast<TorchCommDummy>(torchcomm->unsafeGetBackend());
  ASSERT_NE(backend, nullptr);

  int abortHookCallCount = 0;
  torchcomm->registerAbortHook(
      [&abortHookCallCount]() { abortHookCallCount++; });

  EXPECT_EQ(abortHookCallCount, 0);

  // Trigger abort to invoke hooks
  backend->triggerAbort();

  EXPECT_EQ(abortHookCallCount, 1);

  // Trigger abort again - hook should be called again
  backend->triggerAbort();

  EXPECT_EQ(abortHookCallCount, 2);
}

TEST_F(TorchCommHooksTest, MultipleAbortHooksInvoked) {
  at::Device device(at::kCPU);
  auto torchcomm = new_comm(kBackendName, device, "test_comm", {});
  ASSERT_NE(torchcomm, nullptr);

  auto backend =
      std::dynamic_pointer_cast<TorchCommDummy>(torchcomm->unsafeGetBackend());
  ASSERT_NE(backend, nullptr);

  int hook1CallCount = 0;
  int hook2CallCount = 0;
  int hook3CallCount = 0;

  torchcomm->registerAbortHook([&hook1CallCount]() { hook1CallCount++; });
  torchcomm->registerAbortHook([&hook2CallCount]() { hook2CallCount++; });
  torchcomm->registerAbortHook([&hook3CallCount]() { hook3CallCount++; });

  EXPECT_EQ(hook1CallCount, 0);
  EXPECT_EQ(hook2CallCount, 0);
  EXPECT_EQ(hook3CallCount, 0);

  // Trigger abort - all hooks should be called
  backend->triggerAbort();

  EXPECT_EQ(hook1CallCount, 1);
  EXPECT_EQ(hook2CallCount, 1);
  EXPECT_EQ(hook3CallCount, 1);
}

} // namespace torch::comms
