// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/SendRecv/SendRecvImpl.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/ctran/utils/CommGroupUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::testing {

class SendRecvCtgraphSupportTest : public CtranStandaloneFixture {
 protected:
  void SetUp() override {
    CtranStandaloneFixture::SetUp();
    comm_ = makeCtranComm();
    ASSERT_NE(comm_, nullptr);
  }

  std::unique_ptr<CtranComm> comm_;
};

TEST_F(SendRecvCtgraphSupportTest, NullStream) {
  const int peer = 0;
  EXPECT_FALSE(ctranSendRecvSupport(
      peer, comm_.get(), NCCL_SENDRECV_ALGO::ctgraph, nullptr));
}

TEST_F(SendRecvCtgraphSupportTest, EagerAlgosUnaffected) {
  const int peer = 0;
  EXPECT_TRUE(ctranSendRecvSupport(
      peer, comm_.get(), NCCL_SENDRECV_ALGO::ctran, nullptr));
}

// Verify ctranGroupEndHook returns commInvalidUsage when ctgraph is called
// outside of CUDA graph capture. Uses cudaStreamDefault (stream 0) which is
// always valid and never in capture mode.
TEST_F(SendRecvCtgraphSupportTest, InvalidUsageOutsideCapture) {
  const int peer = 0;
  const size_t count = 1024;

  commGroupDepth++;
  ASSERT_EQ(
      ctranSend(
          nullptr, count, commInt32, peer, comm_.get(), cudaStreamDefault),
      commSuccess);
  ASSERT_EQ(
      ctranRecv(
          nullptr, count, commInt32, peer, comm_.get(), cudaStreamDefault),
      commSuccess);
  commGroupDepth--;

  auto result = ctranGroupEndHook(NCCL_SENDRECV_ALGO::ctgraph, std::nullopt);
  EXPECT_EQ(result, commInvalidUsage);
}

} // namespace ctran::testing
