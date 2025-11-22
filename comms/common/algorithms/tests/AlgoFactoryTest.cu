// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include "comms/common/algorithms/AlgoFactory.cuh"
#include "comms/common/tests/TestBaselineBootstrap.h"
#include "comms/rcclx/develop/meta/lib/tests/RcclxTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::rcclx;
using namespace meta::comms;

namespace {
constexpr int maxBlocks = 128;
constexpr int ddaSendbufSizeBytes = 4096;
// always use a threshold smaller than the ddaSendbufSizeBytes
constexpr int ddaThresholdBytes = ddaSendbufSizeBytes / 4;
constexpr int ddaFlatThresholdBytes = ddaSendbufSizeBytes / 4;
constexpr int ddaTreeThresholdBytes = ddaSendbufSizeBytes / 2;
constexpr commDataType_t dataType = commBfloat16;
constexpr int maxCnt = ddaThresholdBytes / sizeof(__nv_bfloat16);
constexpr int maxOneshotCnt = ddaFlatThresholdBytes / sizeof(__nv_bfloat16);
constexpr int maxTwoshotCnt = ddaTreeThresholdBytes / sizeof(__nv_bfloat16);
} // namespace

class AlgoFactoryTest : public RcclxBaseTestFixture {
 public:
  void SetUp() override {
    RcclxBaseTestFixture::SetUp();
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclComm_t comm{nullptr};
    NCCL_CHECK(
        ncclCommInitRankConfig(&comm, numRanks, commId, globalRank, &config));
    XLOGF(INFO, "rank {} init done; total ranks: {}", globalRank, numRanks);

    ASSERT_EQ(numRanks, 8);
    CUDA_CHECK(cudaSetDevice(localRank));

    auto bootstrap = std::make_shared<TestBaselineBootstrap>(comm);
    factory = std::make_unique<AlgoFactory>(
        bootstrap,
        numRanks,
        localRank,
        maxBlocks,
        AlgoFactory::AllReduceOptions{
            .enableDda = true,
            .ddaSendbufSizeBytes = ddaSendbufSizeBytes,
            .ddaFlatMaxThresholdBytes = ddaFlatThresholdBytes,
            .ddaTreeMaxThresholdBytes = ddaTreeThresholdBytes},
        AlgoFactory::AllGatherOptions{
            .enableDda = true,
            .ddaSendbufSizeBytes = ddaSendbufSizeBytes,
            .ddaMaxThresholdBytes = ddaThresholdBytes},
        AlgoFactory::ReduceScatterOptions{
            .enableDda = true,
            .ddaSendbufSizeBytes = ddaSendbufSizeBytes,
            .ddaMaxThresholdBytes = ddaThresholdBytes},
        AlgoFactory::AllToAllOptions{
            .enableDda = true,
            .ddaSendbufSizeBytes = ddaSendbufSizeBytes,
            .ddaMaxThresholdBytes = ddaThresholdBytes});
  }

 protected:
  std::unique_ptr<AlgoFactory> factory;
  DeviceBuffer sendbuf{ddaSendbufSizeBytes};
  DeviceBuffer recvbuf{ddaSendbufSizeBytes};

  // Stress test by launching multiple allReduce calls consecutively.
  // This validates that there is no synchronization bug in the kernel. For
  // example, if a rank's second allReduce call overrides the buffer that's
  // still being used by the first allReduce call on another rank, then it
  // will be caught by this test.
  void stressTest(int cnt, cudaStream_t stream) {
    // Stress test by launching multiple allReduce calls consecutively.
    // This validates that there is no synchronization bug in the kernel. For
    // example, if a rank's second allReduce call overrides the buffer that's
    // still being used by the first allReduce call on another rank, then it
    // will be caught by this test.
    constexpr int stressCnt = 1000;
    // prepare all the buffers and init values outside of the kernel launch loop
    // so that we can launch kernels consecutively as fast as possible
    std::vector<std::unique_ptr<DeviceBuffer>> sendbufs;
    std::vector<std::unique_ptr<DeviceBuffer>> recvbufs;
    for (int stressIdx = 0; stressIdx < stressCnt; ++stressIdx) {
      sendbufs.emplace_back(
          std::make_unique<DeviceBuffer>(ddaSendbufSizeBytes));
      recvbufs.emplace_back(
          std::make_unique<DeviceBuffer>(ddaSendbufSizeBytes));
      __nv_bfloat16 sendVals[cnt];
      // on each rank, the send buf is an array of
      // [selfRank*stressIdx, selfRank*stressIdx, ...]
      for (int i = 0; i < cnt; ++i) {
        sendVals[i] = localRank * stressIdx;
      }
      CUDA_CHECK(cudaMemcpy(
          sendbufs.at(stressIdx)->get(),
          sendVals,
          sizeof(__nv_bfloat16) * cnt,
          cudaMemcpyDefault));
    }

    for (int stressIdx = 0; stressIdx < stressCnt; ++stressIdx) {
      auto algo = factory->getAllReduceAlgo(
          sendbufs.at(stressIdx)->get(),
          recvbufs.at(stressIdx)->get(),
          cnt,
          dataType,
          stream);
      algo->allReduce();
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int stressIdx = 0; stressIdx < stressCnt; ++stressIdx) {
      __nv_bfloat16 results[cnt];
      CUDA_CHECK(cudaMemcpy(
          results,
          recvbufs.at(stressIdx)->get(),
          sizeof(__nv_bfloat16) * cnt,
          cudaMemcpyDefault));

      __nv_bfloat16 expectedRes[cnt];
      for (int i = 0; i < cnt; ++i) {
        __nv_bfloat16 sum = 0;
        for (int r = 0; r < numRanks; ++r) {
          sum += r * stressIdx;
        }
        expectedRes[i] = sum;
      }
      for (int i = 0; i < cnt; ++i) {
        EXPECT_EQ(results[i], expectedRes[i]);
      }
    }
  }
};

TEST_F(AlgoFactoryTest, allReduceAlgoSelection) {
  CudaStream stream;
  // if msg size is within the dda threshold, we will use DDA
  EXPECT_NE(
      factory->getAllReduceAlgo(
          sendbuf.get(), recvbuf.get(), maxOneshotCnt, dataType, stream.get()),
      nullptr);
  // if msg size exceeds the dda threshold, we won't use DDA
  EXPECT_EQ(
      factory->getAllReduceAlgo(
          sendbuf.get(),
          recvbuf.get(),
          maxTwoshotCnt + 16,
          dataType,
          stream.get()),
      nullptr);
  // if sendbuf is not aligned with 16-byte, we won't do DDA
  EXPECT_EQ(
      factory->getAllReduceAlgo(
          sendbuf.get(),
          recvbuf.get(),
          maxOneshotCnt - 1,
          dataType,
          stream.get()),
      nullptr);
  EXPECT_EQ(
      factory->getAllReduceAlgo(
          sendbuf.get(),
          recvbuf.get(),
          maxTwoshotCnt - 1,
          dataType,
          stream.get()),
      nullptr);
  // DDA currently doesn't support int type
  // DDA only supports SUM op
  EXPECT_EQ(
      factory->getAllReduceAlgo(
          sendbuf.get(), recvbuf.get(), maxOneshotCnt, commInt8, stream.get()),
      nullptr);

  // validate one-shot algo selection
  auto algo = factory->getAllReduceAlgo(
      sendbuf.get(), recvbuf.get(), maxOneshotCnt, dataType, stream.get());
  EXPECT_NE(dynamic_cast<AlgoAllReduceDdaFlatIpc*>(algo.get()), nullptr);

  // validate two-shot algo selection
  algo = factory->getAllReduceAlgo(
      sendbuf.get(), recvbuf.get(), maxTwoshotCnt, dataType, stream.get());
  EXPECT_NE(dynamic_cast<AlgoAllReduceDdaTreeIpc*>(algo.get()), nullptr);
}

TEST_F(AlgoFactoryTest, allReduceFlatDdaStressTest) {
  CudaStream stream;
  stressTest(maxOneshotCnt, stream.get());
}

TEST_F(AlgoFactoryTest, allReduceTreeDdaStressTest) {
  CudaStream stream;
  stressTest(maxTwoshotCnt, stream.get());
}

TEST_F(AlgoFactoryTest, allGatherAlgoSelection) {
  CudaStream stream;
  // if msg size is within the dda threshold, we will use DDA
  EXPECT_NE(
      factory->getAllGatherAlgo(
          sendbuf.get(), recvbuf.get(), maxOneshotCnt, dataType, stream.get()),
      nullptr);
  // if msg size exceeds the dda threshold, we won't use DDA
  EXPECT_EQ(
      factory->getAllGatherAlgo(
          sendbuf.get(), recvbuf.get(), maxCnt + 16, dataType, stream.get()),
      nullptr);
  // if sendbuf is not aligned with 16-byte, we won't do DDA
  EXPECT_EQ(
      factory->getAllGatherAlgo(
          sendbuf.get(), recvbuf.get(), maxCnt - 1, dataType, stream.get()),
      nullptr);
  EXPECT_EQ(
      factory->getAllGatherAlgo(
          sendbuf.get(), recvbuf.get(), maxCnt - 1, dataType, stream.get()),
      nullptr);
  // DDA currently doesn't support int type
  // DDA only supports SUM op
  EXPECT_EQ(
      factory->getAllGatherAlgo(
          sendbuf.get(), recvbuf.get(), maxCnt, commInt8, stream.get()),
      nullptr);

  // validate dda algo selection
  auto algo = factory->getAllGatherAlgo(
      sendbuf.get(), recvbuf.get(), maxCnt, dataType, stream.get());
  EXPECT_NE(dynamic_cast<AlgoAllGatherDdaIpc*>(algo.get()), nullptr);
}

TEST_F(AlgoFactoryTest, reduceScatterAlgoSelection) {
  CudaStream stream;
  // if msg size is within the dda threshold, we will use DDA
  EXPECT_NE(
      factory->getReduceScatterAlgo(
          sendbuf.get(), recvbuf.get(), maxOneshotCnt, dataType, stream.get()),
      nullptr);
  // if msg size exceeds the dda threshold, we won't use DDA
  EXPECT_EQ(
      factory->getReduceScatterAlgo(
          sendbuf.get(), recvbuf.get(), maxCnt + 16, dataType, stream.get()),
      nullptr);
  // if sendbuf is not aligned with 16-byte, we won't do DDA
  EXPECT_EQ(
      factory->getReduceScatterAlgo(
          sendbuf.get(), recvbuf.get(), maxCnt - 1, dataType, stream.get()),
      nullptr);
  EXPECT_EQ(
      factory->getReduceScatterAlgo(
          sendbuf.get(), recvbuf.get(), maxCnt - 1, dataType, stream.get()),
      nullptr);
  // DDA currently doesn't support int type
  // DDA only supports SUM op
  EXPECT_EQ(
      factory->getReduceScatterAlgo(
          sendbuf.get(), recvbuf.get(), maxCnt, commInt8, stream.get()),
      nullptr);

  // validate dda algo selection
  auto algo = factory->getReduceScatterAlgo(
      sendbuf.get(), recvbuf.get(), maxCnt, dataType, stream.get());
  EXPECT_NE(dynamic_cast<AlgoReduceScatterDdaIpc*>(algo.get()), nullptr);
}

TEST_F(AlgoFactoryTest, allToAllAlgoSelection) {
  CudaStream stream;
  // if msg size is within the dda threshold, we will use DDA
  EXPECT_NE(
      factory->getAllToAllAlgo(
          sendbuf.get(), recvbuf.get(), maxOneshotCnt, dataType, stream.get()),
      nullptr);
  // if msg size exceeds the dda threshold, we won't use DDA
  EXPECT_EQ(
      factory->getAllToAllAlgo(
          sendbuf.get(), recvbuf.get(), maxCnt + 16, dataType, stream.get()),
      nullptr);
  // if sendbuf is not aligned with 16-byte, we won't do DDA
  EXPECT_EQ(
      factory->getAllToAllAlgo(
          sendbuf.get(), recvbuf.get(), maxCnt - 1, dataType, stream.get()),
      nullptr);
  EXPECT_EQ(
      factory->getAllToAllAlgo(
          sendbuf.get(), recvbuf.get(), maxCnt - 1, dataType, stream.get()),
      nullptr);
  // DDA currently doesn't support int type
  // DDA only supports SUM op
  EXPECT_EQ(
      factory->getAllToAllAlgo(
          sendbuf.get(), recvbuf.get(), maxCnt, commInt8, stream.get()),
      nullptr);

  // validate dda algo selection
  auto algo = factory->getAllToAllAlgo(
      sendbuf.get(), recvbuf.get(), maxCnt, dataType, stream.get());
  EXPECT_NE(dynamic_cast<AlgoAllToAllDdaIpc*>(algo.get()), nullptr);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
