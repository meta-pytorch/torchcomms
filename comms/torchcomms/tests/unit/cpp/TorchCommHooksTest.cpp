// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comms/torchcomms/TorchComm.hpp>
#include <comms/torchcomms/TorchCommFactory.hpp>
#include <comms/torchcomms/dummy/TorchCommDummy.hpp>
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
  int postHookCallCount = 0;

  auto preHandle = torchcomm->registerPreHook(
      [&preHookCalls](OpName name, size_t, const PreHookArgs&) {
        preHookCalls.push_back(name);
      });

  auto postHandle = torchcomm->registerPostHook(
      [&postHookCallCount](size_t, const PostHookArgs&) {
        postHookCallCount++;
      });

  auto tensor = at::ones({2, 2}, at::kFloat);
  torchcomm->all_reduce(tensor, ReduceOp::SUM, true);

  ASSERT_EQ(preHookCalls.size(), 1);
  EXPECT_EQ(preHookCalls[0], OpName::all_reduce);
  EXPECT_EQ(postHookCallCount, 1);
}

TEST_F(TorchCommHooksTest, PreAndPostHookOpIdIncreases) {
  at::Device device(at::kCPU);
  auto torchcomm = new_comm(kBackendName, device, "test_comm", {});
  ASSERT_NE(torchcomm, nullptr);

  std::vector<size_t> preOpIds;
  std::vector<size_t> postOpIds;

  auto preHandle = torchcomm->registerPreHook(
      [&preOpIds](OpName, size_t op_id, const PreHookArgs&) {
        preOpIds.push_back(op_id);
      });

  auto postHandle = torchcomm->registerPostHook(
      [&postOpIds](size_t op_id, const PostHookArgs&) {
        postOpIds.push_back(op_id);
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
      [&preHookCallCount](OpName, size_t, const PreHookArgs&) {
        preHookCallCount++;
      });

  auto postHandle = torchcomm->registerPostHook(
      [&postHookCallCount](size_t, const PostHookArgs&) {
        postHookCallCount++;
      });

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
      [&preHook1CallCount](OpName, size_t, const PreHookArgs&) {
        preHook1CallCount++;
      });

  auto preHandle2 = torchcomm->registerPreHook(
      [&preHook2CallCount](OpName, size_t, const PreHookArgs&) {
        preHook2CallCount++;
      });

  auto postHandle1 = torchcomm->registerPostHook(
      [&postHook1CallCount](size_t, const PostHookArgs&) {
        postHook1CallCount++;
      });

  auto postHandle2 = torchcomm->registerPostHook(
      [&postHook2CallCount](size_t, const PostHookArgs&) {
        postHook2CallCount++;
      });

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
  std::vector<size_t> postHookOpIds;

  auto preHandle = torchcomm->registerPreHook(
      [&preHookCalls](OpName name, size_t op_id, const PreHookArgs&) {
        preHookCalls.emplace_back(name, op_id);
      });

  auto postHandle = torchcomm->registerPostHook(
      [&postHookOpIds](size_t op_id, const PostHookArgs&) {
        postHookOpIds.push_back(op_id);
      });

  auto tensor = at::ones({2, 2}, at::kFloat);

  torchcomm->all_reduce(tensor, ReduceOp::SUM, true);
  torchcomm->barrier(true);
  torchcomm->broadcast(tensor, 0, true);

  ASSERT_EQ(preHookCalls.size(), 3);
  ASSERT_EQ(postHookOpIds.size(), 3);

  EXPECT_EQ(preHookCalls[0].first, OpName::all_reduce);
  EXPECT_EQ(preHookCalls[1].first, OpName::barrier);
  EXPECT_EQ(preHookCalls[2].first, OpName::broadcast);

  EXPECT_LT(preHookCalls[0].second, preHookCalls[1].second);
  EXPECT_LT(preHookCalls[1].second, preHookCalls[2].second);

  EXPECT_LT(postHookOpIds[0], postHookOpIds[1]);
  EXPECT_LT(postHookOpIds[1], postHookOpIds[2]);

  EXPECT_EQ(preHookCalls[0].second, postHookOpIds[0]);
  EXPECT_EQ(preHookCalls[1].second, postHookOpIds[1]);
  EXPECT_EQ(preHookCalls[2].second, postHookOpIds[2]);
}

TEST_F(TorchCommHooksTest, PreHookArgsContainCorrectVariantTypes) {
  at::Device device(at::kCPU);
  auto torchcomm = new_comm(kBackendName, device, "test_comm", {});
  ASSERT_NE(torchcomm, nullptr);

  std::vector<const PreHookArgs*> captured_args;
  // Store the variant index to verify the correct type was passed
  std::vector<size_t> variant_indices;

  auto preHandle = torchcomm->registerPreHook(
      [&variant_indices](OpName, size_t, const PreHookArgs& args) {
        variant_indices.push_back(args.index());
      });

  auto tensor = at::ones({2, 2}, at::kFloat);

  // all_reduce should produce AllReducePreHookArgs
  torchcomm->all_reduce(tensor, ReduceOp::SUM, true);
  ASSERT_EQ(variant_indices.size(), 1);
  // AllReducePreHookArgs is at index 3 in the variant (Send=0, Recv=1,
  // Broadcast=2, AllReduce=3)

  // barrier should produce BarrierPreHookArgs
  torchcomm->barrier(true);
  ASSERT_EQ(variant_indices.size(), 2);

  // broadcast should produce BroadcastPreHookArgs
  torchcomm->broadcast(tensor, 0, true);
  ASSERT_EQ(variant_indices.size(), 3);

  // Verify each call produced a different variant type
  // (all_reduce, barrier, and broadcast are different variant alternatives)
  EXPECT_NE(variant_indices[0], variant_indices[1]);
  EXPECT_NE(variant_indices[1], variant_indices[2]);
  EXPECT_NE(variant_indices[0], variant_indices[2]);
}

TEST_F(TorchCommHooksTest, PreHookAllReduceArgsAccessible) {
  at::Device device(at::kCPU);
  auto torchcomm = new_comm(kBackendName, device, "test_comm", {});
  ASSERT_NE(torchcomm, nullptr);

  bool hook_called = false;
  auto preHandle = torchcomm->registerPreHook(
      [&hook_called](OpName, size_t, const PreHookArgs& args) {
        auto* ar = std::get_if<AllReducePreHookArgs>(&args);
        ASSERT_NE(ar, nullptr);
        EXPECT_EQ(ar->tensor.numel(), 4); // 2x2 tensor
        EXPECT_TRUE(ar->async_op);
        hook_called = true;
      });

  auto tensor = at::ones({2, 2}, at::kFloat);
  torchcomm->all_reduce(tensor, ReduceOp::SUM, true);
  EXPECT_TRUE(hook_called);
}

TEST_F(TorchCommHooksTest, AbortHookNotCalledAfterRemoval) {
  at::Device device(at::kCPU);
  auto torchcomm = new_comm(kBackendName, device, "test_comm", {});
  ASSERT_NE(torchcomm, nullptr);

  auto backend =
      std::dynamic_pointer_cast<TorchCommDummy>(torchcomm->getBackendImpl());
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
      std::dynamic_pointer_cast<TorchCommDummy>(torchcomm->getBackendImpl());
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
      std::dynamic_pointer_cast<TorchCommDummy>(torchcomm->getBackendImpl());
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
