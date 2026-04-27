// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include "comm.h"
#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "meta/NcclxConfig.h" // @manual
#include "meta/hints/GlobalHints.h" // @manual
#include "meta/transport/NcclxIbNetCommConfig.h" // @manual
#include "nccl.h" // @manual

class CommSplitPerCommConfigTest : public NcclxBaseTestFixture {
 public:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp();
    ASSERT_EQ(
        ncclx::setGlobalHint(std::string(ncclx::HintKeys::kCommNoLocal), "1"),
        ncclSuccess);
    comm = ncclx::test::createNcclComm(
        globalRank, numRanks, localRank, bootstrap_.get());
    CUDACHECK_TEST(cudaStreamCreate(&stream));
    CUDACHECK_TEST(cudaMalloc(&dataBuf, sizeof(int) * dataCount));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaFree(dataBuf));
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    NCCLCHECK_TEST(ncclCommDestroy(comm));
    ncclx::resetGlobalHint(std::string(ncclx::HintKeys::kCommNoLocal));
    NcclxBaseTestFixture::TearDown();
  }

  void runAllReduce(ncclComm_t c) {
    int myRank, nRanks;
    NCCLCHECK_TEST(ncclCommUserRank(c, &myRank));
    NCCLCHECK_TEST(ncclCommCount(c, &nRanks));

    std::vector<int> initVals(dataCount);
    for (int i = 0; i < dataCount; i++) {
      initVals[i] = i * myRank;
    }
    CUDACHECK_TEST(cudaMemcpy(
        dataBuf,
        initVals.data(),
        sizeof(int) * dataCount,
        cudaMemcpyHostToDevice));

    NCCLCHECK_TEST(ncclAllReduce(
        dataBuf, dataBuf, dataCount, ncclInt, ncclSum, c, stream));
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    std::vector<int> result(dataCount);
    CUDACHECK_TEST(cudaMemcpy(
        result.data(),
        dataBuf,
        sizeof(int) * dataCount,
        cudaMemcpyDeviceToHost));

    const int sumRanks = nRanks * (nRanks - 1) / 2;
    int errs = 0;
    for (int i = 0; i < dataCount; i++) {
      if (result[i] != i * sumRanks) {
        errs++;
      }
    }
    EXPECT_EQ(errs, 0) << "AllReduce data mismatch on rank " << myRank;
  }

  ncclComm_t comm;
  cudaStream_t stream;
  int* dataBuf{nullptr};
  static constexpr int dataCount = 65536;
};

// --- ncclBuffSize tests ---

TEST_F(CommSplitPerCommConfigTest, NcclBuffSizeOverride) {
  constexpr int kCustomBuffSize = 8388608;
  int parentBuffSize = comm->buffSizes[NCCL_PROTO_SIMPLE];

  ncclConfig_t splitConfig = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints({{"ncclBuffSize", "8388608"}});
  splitConfig.hints = &hints;

  ncclx::test::NcclCommSplitRAII child(comm, 0, globalRank, &splitConfig);

  EXPECT_EQ(child->buffSizes[NCCL_PROTO_SIMPLE], kCustomBuffSize);
  EXPECT_EQ(comm->buffSizes[NCCL_PROTO_SIMPLE], parentBuffSize);

  runAllReduce(comm);
  runAllReduce(child.get());
}

// --- IB config tests ---

TEST_F(CommSplitPerCommConfigTest, IbConfigOverride) {
  ncclConfig_t splitConfig = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints({
      {"ibQpsPerConnection", "2"},
      {"ibSplitDataOnQps", "1"},
  });
  splitConfig.hints = &hints;

  ncclx::test::NcclCommSplitRAII child(comm, 0, globalRank, &splitConfig);

  auto* childCtx = static_cast<ncclx::NcclxIbNetCommConfig*>(child->netContext);
  ASSERT_NE(childCtx, nullptr);
  EXPECT_EQ(childCtx->ibQpsPerConnection, 2);
  EXPECT_EQ(childCtx->ibSplitDataOnQps, 1);

  auto* parentCtx = static_cast<ncclx::NcclxIbNetCommConfig*>(comm->netContext);
  ASSERT_NE(parentCtx, nullptr);
  EXPECT_EQ(parentCtx->ibQpsPerConnection, NCCL_CONFIG_UNDEF_INT);
  EXPECT_EQ(parentCtx->ibSplitDataOnQps, NCCL_CONFIG_UNDEF_INT);

  runAllReduce(comm);
  runAllReduce(child.get());
}

TEST_F(CommSplitPerCommConfigTest, DefaultHintsMatchParent) {
  int parentBuffSize = comm->buffSizes[NCCL_PROTO_SIMPLE];

  ncclx::test::NcclCommSplitRAII child(comm, 0, globalRank);

  EXPECT_EQ(child->buffSizes[NCCL_PROTO_SIMPLE], parentBuffSize);

  auto* childCtx = static_cast<ncclx::NcclxIbNetCommConfig*>(child->netContext);
  ASSERT_NE(childCtx, nullptr);
  EXPECT_EQ(childCtx->ibQpsPerConnection, NCCL_CONFIG_UNDEF_INT);
  EXPECT_EQ(childCtx->ibSplitDataOnQps, NCCL_CONFIG_UNDEF_INT);

  runAllReduce(child.get());
}

// --- splitShare=1 rejection tests ---

TEST_F(CommSplitPerCommConfigTest, SplitShareRejectsNcclBuffSize) {
  ncclConfig_t splitConfig = NCCL_CONFIG_INITIALIZER;
  splitConfig.splitShare = 1;
  ncclx::Hints hints({{"ncclBuffSize", "8388608"}});
  splitConfig.hints = &hints;

  ncclComm_t child = nullptr;
  ncclResult_t res = ncclCommSplit(comm, 0, globalRank, &child, &splitConfig);
  EXPECT_EQ(res, ncclInvalidArgument);
}

TEST_F(CommSplitPerCommConfigTest, SplitShareRejectsIbSplitDataOnQps) {
  ncclConfig_t splitConfig = NCCL_CONFIG_INITIALIZER;
  splitConfig.splitShare = 1;
  ncclx::Hints hints({{"ibSplitDataOnQps", "1"}});
  splitConfig.hints = &hints;

  ncclComm_t child = nullptr;
  ncclResult_t res = ncclCommSplit(comm, 0, globalRank, &child, &splitConfig);
  EXPECT_EQ(res, ncclInvalidArgument);
}

TEST_F(CommSplitPerCommConfigTest, SplitShareRejectsIbQpsPerConnection) {
  ncclConfig_t splitConfig = NCCL_CONFIG_INITIALIZER;
  splitConfig.splitShare = 1;
  ncclx::Hints hints({{"ibQpsPerConnection", "2"}});
  splitConfig.hints = &hints;

  ncclComm_t child = nullptr;
  ncclResult_t res = ncclCommSplit(comm, 0, globalRank, &child, &splitConfig);
  EXPECT_EQ(res, ncclInvalidArgument);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
