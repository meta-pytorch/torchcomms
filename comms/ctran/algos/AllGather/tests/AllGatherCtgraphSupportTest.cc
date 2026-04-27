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

// Verify ctranAllGather returns commInvalidUsage when a graph-aware algo is
// called outside of CUDA graph capture. Uses cudaStreamDefault (stream 0)
// which is always valid and never in capture mode.
TEST_F(AllGatherCtgraphSupportTest, InvalidUsageOutsideCapture) {
  const size_t count = 1024;
  void* sendbuf = nullptr;
  void* recvbuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendbuf, count * sizeof(int32_t)));
  CUDACHECK_TEST(cudaMalloc(&recvbuf, count * 8 * sizeof(int32_t)));

  auto result = ctranAllGather(
      sendbuf,
      recvbuf,
      count,
      commInt32,
      comm_.get(),
      cudaStreamDefault,
      NCCL_ALLGATHER_ALGO::ctgraph);
  EXPECT_EQ(result, commInvalidUsage);

  CUDACHECK_TEST(cudaFree(sendbuf));
  CUDACHECK_TEST(cudaFree(recvbuf));
}

} // namespace ctran::testing
