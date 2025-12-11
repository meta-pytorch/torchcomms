// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <gtest/gtest.h>
#include "comms/ctran/algos/SendRecv/SendRecvImpl.h"
#include "comms/ctran/tests/CtranTestUtils.h"

class CtranAlgoNameTest : public ::testing::Test {
 public:
  CtranAlgoNameTest() = default;

 protected:
  void SetUp() override {
    EXPECT_EQ(cudaSetDevice(cudaDev_), cudaSuccess);

    ncclCvarInit();
    dummyCommRAII = ctran::createDummyCtranComm();
    dummyComm_ = dummyCommRAII->ctranComm.get();
  }

 protected:
  int cudaDev_{0};
  cudaStream_t stream_{0};
  int dummyOpCount_{0};
  std::unique_ptr<ctran::TestCtranCommRAII> dummyCommRAII;
  CtranComm* dummyComm_{nullptr};
};

TEST_F(CtranAlgoNameTest, SendAlgoName) {
  auto op = OpElem(OpElem::opType::SEND, stream_, dummyComm_, dummyOpCount_);

  std::vector<OpElem*> opGroup;
  opGroup.push_back(&op);

  EXPECT_EQ(sendRecvAlgoName(NCCL_SENDRECV_ALGO::ctran, opGroup), "CtranSend");
}

TEST_F(CtranAlgoNameTest, RecvAlgoName) {
  cudaStream_t stream_ = 0;
  constexpr int dummyOpCount_ = 0;
  auto op = OpElem(OpElem::opType::RECV, stream_, dummyComm_, dummyOpCount_);

  std::vector<OpElem*> opGroup;
  opGroup.push_back(&op);

  EXPECT_EQ(sendRecvAlgoName(NCCL_SENDRECV_ALGO::ctran, opGroup), "CtranRecv");
}

TEST_F(CtranAlgoNameTest, SendRecvAlgoName) {
  cudaStream_t stream_ = 0;
  constexpr int dummyOpCount_ = 0;
  auto op1 = OpElem(OpElem::opType::SEND, stream_, dummyComm_, dummyOpCount_);

  auto op2 = OpElem(OpElem::opType::RECV, stream_, dummyComm_, dummyOpCount_);

  std::vector<OpElem*> opGroup;
  opGroup.push_back(&op1);
  opGroup.push_back(&op2);

  EXPECT_EQ(
      sendRecvAlgoName(NCCL_SENDRECV_ALGO::ctran, opGroup), "CtranSendRecv");
}
