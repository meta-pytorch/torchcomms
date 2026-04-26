// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::testing {

class AllGatherCtgraphSupportTest : public CtranStandaloneFixture {
 protected:
  void SetUp() override {
    CtranStandaloneFixture::SetUp();
    comm_ = makeCtranComm();
    ASSERT_NE(comm_, nullptr);
  }

  std::unique_ptr<CtranComm> comm_;
};

TEST_F(AllGatherCtgraphSupportTest, NullStream) {
  EXPECT_FALSE(ctranAllGatherSupport(
      comm_.get(), NCCL_ALLGATHER_ALGO::ctgraph, nullptr));
  EXPECT_FALSE(ctranAllGatherSupport(
      comm_.get(), NCCL_ALLGATHER_ALGO::ctgraph_pipeline, nullptr));
  EXPECT_FALSE(ctranAllGatherSupport(
      comm_.get(), NCCL_ALLGATHER_ALGO::ctgraph_ring, nullptr));
  EXPECT_FALSE(ctranAllGatherSupport(
      comm_.get(), NCCL_ALLGATHER_ALGO::ctgraph_rd, nullptr));
}

TEST_F(AllGatherCtgraphSupportTest, EagerAlgosUnaffected) {
  EXPECT_TRUE(ctranAllGatherSupport(
      comm_.get(), NCCL_ALLGATHER_ALGO::ctdirect, stream->get()));
  EXPECT_TRUE(ctranAllGatherSupport(
      comm_.get(), NCCL_ALLGATHER_ALGO::ctran, stream->get()));
}

} // namespace ctran::testing
