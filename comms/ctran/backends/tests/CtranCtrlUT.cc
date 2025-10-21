// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/ctran/backends/CtranCtrl.h"

class CtranCtrlTest : public ::testing::Test {
 public:
  CtranCtrlTest() = default;
};

namespace {
const int kPeerRank = 9;
bool testCbCtx = false;
constexpr int kTestCMsgType1 = 1;

commResult_t testCtrlMsgCb(int peer, void* msgPtr, void* ctx) {
  bool* ctxPtr = reinterpret_cast<bool*>(ctx);
  // Update context in CB to verify that callback is called with the correct
  // context
  *ctxPtr = true;
  EXPECT_EQ(peer, kPeerRank);

  auto msg = reinterpret_cast<ControlMsg*>(msgPtr);
  EXPECT_EQ(msg->type, kTestCMsgType1);
  return commSuccess;
}
} // namespace

TEST_F(CtranCtrlTest, regCb) {
  auto ctrlMgr = std::make_unique<CtranCtrlManager>();
  EXPECT_THAT(ctrlMgr, testing::NotNull());

  auto res = ctrlMgr->regCb(kTestCMsgType1, testCtrlMsgCb, &testCbCtx);
  EXPECT_EQ(res, commSuccess);
  EXPECT_TRUE(ctrlMgr->hasCb(kTestCMsgType1));
}

TEST_F(CtranCtrlTest, dupRegCb) {
  auto ctrlMgr = std::make_unique<CtranCtrlManager>();
  EXPECT_THAT(ctrlMgr, testing::NotNull());

  // First registration should succeed
  auto res = ctrlMgr->regCb(kTestCMsgType1, testCtrlMsgCb, &testCbCtx);
  EXPECT_EQ(res, commSuccess);
  EXPECT_TRUE(ctrlMgr->hasCb(kTestCMsgType1));

  // Duplicate registration should fail
  res = ctrlMgr->regCb(kTestCMsgType1, testCtrlMsgCb, &testCbCtx);
  EXPECT_EQ(res, commInternalError);
}

TEST_F(CtranCtrlTest, noCb) {
  auto ctrlMgr = std::make_unique<CtranCtrlManager>();
  EXPECT_THAT(ctrlMgr, testing::NotNull());

  EXPECT_FALSE(ctrlMgr->hasCb(kTestCMsgType1));
}

TEST_F(CtranCtrlTest, runCb) {
  auto ctrlMgr = std::make_unique<CtranCtrlManager>();
  EXPECT_THAT(ctrlMgr, testing::NotNull());

  auto res = ctrlMgr->regCb(kTestCMsgType1, testCtrlMsgCb, &testCbCtx);
  EXPECT_EQ(res, commSuccess);

  ControlMsg msg = ControlMsg(kTestCMsgType1);
  res = ctrlMgr->runCb(kPeerRank, kTestCMsgType1, &msg);
  EXPECT_EQ(res, commSuccess);

  // Expect callback has updated the registered context
  EXPECT_TRUE(testCbCtx);
}
