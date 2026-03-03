// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include "checks.h"
#include "comm.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "nccl.h"

void printCommStateX(const ncclComm& comm) {
  const auto statex = comm.ctranComm_->statex_.get();
  VLOG(1) << "=== CommStateX ===";
  VLOG(1) << "CommStateX " << &comm << ": ";
  VLOG(1) << "rank: " << statex->rank();
  VLOG(1) << "nRanks: " << statex->nRanks();
  VLOG(1) << "=================";
}

void validateCtranInitialization(
    ncclComm_t comm,
    int expectedRank,
    int expectedNRanks,
    int expectedCudaDev) {
  EXPECT_EQ(comm->rank, expectedRank);
  EXPECT_EQ(comm->nRanks, expectedNRanks);
  EXPECT_EQ(comm->cudaDev, expectedCudaDev);

  ASSERT_NE(nullptr, comm->ctranComm_);
  ASSERT_NE(nullptr, comm->ctranComm_->statex_);
  ASSERT_NE(nullptr, comm->ctranComm_->bootstrap_);
  ASSERT_NE(nullptr, comm->ctranComm_->colltraceNew_);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);
  EXPECT_TRUE(ctranInitialized(comm->ctranComm_.get()));
  EXPECT_EQ(comm->commHash, comm->ctranComm_->statex_->commHash());
}

ncclConfig_t ncclConfigInitHelper(bool enableFastInitConfig) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  if (enableFastInitConfig) {
    config.fastInitMode = NCCL_FAST_INIT_MODE_RING;
  }
  return config;
}

ncclResult_t ncclCommInitRankConfigHelper(
    ncclComm_t* comm,
    int nRanks,
    ncclUniqueId commId,
    int myRank,
    bool enableFastInitConfig) {
  if (!enableFastInitConfig) {
    return ncclCommInitRankConfig(comm, nRanks, commId, myRank, nullptr);
  } else {
    ncclConfig_t config = ncclConfigInitHelper(enableFastInitConfig);
    return ncclCommInitRankConfig(comm, nRanks, commId, myRank, &config);
  }
}

TEST_P(NcclxBaseTestFixture, NcclCommInitWorldAndDestroy) {
  ncclComm_t rootComm = nullptr;
  ncclUniqueId commId;
  NCCLCHECK_TEST(ncclCommInitRankConfigHelper(
      &rootComm, numRanks, commId, globalRank, enableFastInitConfig));

  ASSERT_NE(nullptr, rootComm);
  validateCtranInitialization(rootComm, globalRank, numRanks, localRank);

  const auto statex = rootComm->ctranComm_->statex_.get();
  if (statex->nNodes() == 1) {
    NCCLCHECK_TEST(ncclCommDestroy(rootComm));
    GTEST_SKIP() << "Skip test since only one node provided";
  }

  EXPECT_EQ(statex->rank(), globalRank);
  EXPECT_EQ(statex->nRanks(), numRanks);

  // distributed run strict checks
  EXPECT_EQ(statex->nNodes(), 2);
  EXPECT_EQ(statex->nLocalRanks(), localSize);
  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

TEST_P(NcclxBaseTestFixture, NcclCommInitWorldAndAbort) {
  ncclComm_t rootComm = nullptr;
  ncclUniqueId commId;
  NCCLCHECK_TEST(ncclCommInitRankConfigHelper(
      &rootComm, numRanks, commId, globalRank, enableFastInitConfig));
  ASSERT_NE(nullptr, rootComm);
  validateCtranInitialization(rootComm, globalRank, numRanks, localRank);

  void* sendBuf;
  void* recvBuf;
  ncclDataType_t dataType = ncclBfloat16;
  size_t count = 8192;
  size_t sendBytes = count * ncclTypeSize(dataType);
  size_t recvBytes = sendBytes * numRanks;
  cudaStream_t stream;

  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendBytes));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, recvBytes));
  CUDACHECK_TEST(cudaMemset(sendBuf, globalRank, sendBytes));
  CUDACHECK_TEST(cudaMemset(recvBuf, 0, recvBytes));
  // Ensure value has been set before colletive runs on nonblocking stream
  CUDACHECK_TEST(cudaDeviceSynchronize());
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // run baseline allgather
  auto res = ncclAllGather(sendBuf, recvBuf, count, dataType, rootComm, stream);
  EXPECT_EQ(res, ncclSuccess);

  // kill collective and abort
  NCCLCHECK_TEST(ncclCommAbort(rootComm));
}

void compareComm(const ncclComm& comm1, const ncclComm& comm2) {
  const auto state1 = comm1.ctranComm_->statex_.get();
  const auto state2 = comm2.ctranComm_->statex_.get();
  EXPECT_EQ(state1->rank(), state2->rank());
  EXPECT_EQ(state1->nRanks(), state2->nRanks());
  EXPECT_EQ(state1->cudaDev(), state2->cudaDev());

  // TODO: baseline ncclComm does not maintain commRankToWorldRank mapping
  // populate them to support statex->gRank() API
}

TEST_P(NcclxBaseTestFixture, NcclCommSplit) {
  ncclComm_t rootComm = nullptr;
  ncclUniqueId commId;
  NCCLCHECK_TEST(ncclCommInitRankConfigHelper(
      &rootComm, numRanks, commId, globalRank, enableFastInitConfig));
  ASSERT_NE(nullptr, rootComm);
  validateCtranInitialization(rootComm, globalRank, numRanks, localRank);

  ncclComm_t childComm = nullptr;
  ncclConfig_t childCommConfig = ncclConfigInitHelper(enableFastInitConfig);
  int color = globalRank % 2;
  std::string childCommDesc = "child_communicator_" + std::to_string(color);
  childCommConfig.commDesc = childCommDesc.c_str();
  int groupSize = rootComm->ctranComm_->statex_.get()->nRanks() / 2;
  int* groupRanks = new int[groupSize];
  for (int i = 0; i < groupSize; ++i) {
    *(groupRanks + i) = 2 * i + globalRank % 2;
  }
  childCommConfig.splitGroupRanks = groupRanks;
  childCommConfig.splitGroupSize = groupSize;
  NCCLCHECK_TEST(ncclCommSplit(
      rootComm, color, globalRank / 2, &childComm, &childCommConfig));
  ASSERT_NE(nullptr, childComm);

  const auto statex1 = childComm->ctranComm_->statex_.get();
  if (statex1->nNodes() == 1) {
    NCCLCHECK_TEST(ncclCommDestroy(childComm));
    NCCLCHECK_TEST(ncclCommDestroy(rootComm));
    GTEST_SKIP() << "Skip test since only one node provided";
  }
  EXPECT_EQ(statex1->nRanks(), numRanks / 2);
  EXPECT_NE(statex1->commHash(), 0);
  // distributed run strict checks
  EXPECT_EQ(statex1->nNodes(), 2);
  EXPECT_EQ(statex1->nLocalRanks(), localSize / 2);

  ncclComm expectedComm;
  expectedComm.config.commDesc = childCommConfig.commDesc;
  setCtranCommBase(&expectedComm);

  expectedComm.ctranComm_->statex_ = std::make_unique<ncclx::CommStateX>(
      globalRank / 2,
      numRanks / 2,
      localRank,
      rootComm->ctranComm_->statex_->cudaArch(), // cudaArch of H100
      25, // busId
      0,
      std::vector<ncclx::RankTopology>{},
      std::vector<int>{});

  printCommStateX(*childComm);
  printCommStateX(expectedComm);
  compareComm(*childComm, expectedComm);

  NCCLCHECK_TEST(ncclCommDestroy(childComm));
  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

// we can split the same group rank multiple times
// we should expect unique hash for each communicator
TEST_P(NcclxBaseTestFixture, NcclCommSplitDuplicateGroups) {
  ncclComm_t rootComm = nullptr;
  ncclUniqueId commId;
  NCCLCHECK_TEST(ncclCommInitRankConfigHelper(
      &rootComm, numRanks, commId, globalRank, enableFastInitConfig));
  ASSERT_NE(nullptr, rootComm);
  validateCtranInitialization(rootComm, globalRank, numRanks, localRank);
  const auto statex = rootComm->ctranComm_->statex_.get();

  if (statex->nNodes() == 1) {
    NCCLCHECK_TEST(ncclCommDestroy(rootComm));
    GTEST_SKIP() << "Skip test since only one node provided";
  }

  // child comm config
  ncclConfig_t childCommConfig = ncclConfigInitHelper(enableFastInitConfig);
  int color = globalRank % 2;
  std::string childCommDesc = "child_communicator_" + std::to_string(color);
  childCommConfig.commDesc = childCommDesc.c_str();
  int groupSize = rootComm->ctranComm_->statex_.get()->nRanks() / 2;
  int* groupRanks = new int[groupSize];
  for (int i = 0; i < groupSize; ++i) {
    *(groupRanks + i) = 2 * i + globalRank % 2;
  }
  childCommConfig.splitGroupRanks = groupRanks;
  childCommConfig.splitGroupSize = groupSize;

  ncclComm_t childComm1 = nullptr;
  NCCLCHECK_TEST(ncclCommSplit(
      rootComm, color, globalRank / 2, &childComm1, &childCommConfig));
  ASSERT_NE(nullptr, childComm1);

  // split again with same config
  ncclComm_t childComm2 = nullptr;
  NCCLCHECK_TEST(ncclCommSplit(
      rootComm, color, globalRank / 2, &childComm2, &childCommConfig));
  ASSERT_NE(nullptr, childComm2);

  const auto statex1 = childComm1->ctranComm_->statex_.get();
  const auto statex2 = childComm2->ctranComm_->statex_.get();

  // hash should be different
  EXPECT_NE(statex1->commHash(), statex2->commHash());

  NCCLCHECK_TEST(ncclCommDestroy(childComm2));
  NCCLCHECK_TEST(ncclCommDestroy(childComm1));
  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

TEST_P(NcclxBaseTestFixture, WorldCommAllGather) {
  ncclComm_t rootComm = nullptr;
  ncclUniqueId commId;
  NCCLCHECK_TEST(ncclCommInitRankConfigHelper(
      &rootComm, numRanks, commId, globalRank, enableFastInitConfig));
  ASSERT_NE(nullptr, rootComm);
  validateCtranInitialization(rootComm, globalRank, numRanks, localRank);
  const auto statex = rootComm->ctranComm_->statex_.get();

  if (statex->nNodes() == 1) {
    NCCLCHECK_TEST(ncclCommDestroy(rootComm));
    GTEST_SKIP() << "Skip test since only one node provided";
  }

  void* sendBuf;
  void* recvBuf;
  ncclDataType_t dataType = ncclBfloat16;
  size_t count = 8192;
  size_t sendBytes = count * ncclTypeSize(dataType);
  size_t recvBytes = sendBytes * numRanks;
  cudaStream_t stream;

  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendBytes));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, recvBytes));
  CUDACHECK_TEST(cudaMemset(sendBuf, globalRank, sendBytes));
  CUDACHECK_TEST(cudaMemset(recvBuf, 0, recvBytes));
  // Ensure value has been set before colletive runs on nonblocking stream
  CUDACHECK_TEST(cudaDeviceSynchronize());
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // run baseline allgather
  auto res = ncclAllGather(sendBuf, recvBuf, count, dataType, rootComm, stream);
  EXPECT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  for (int i = 0; i < numRanks; ++i) {
    std::vector<char> observedVals(sendBytes, rand());
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(),
        (char*)recvBuf + i * sendBytes,
        sendBytes,
        cudaMemcpyDefault));
    EXPECT_THAT(observedVals, testing::Each(i));
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

TEST_P(NcclxBaseTestFixture, ChildCommAllGather) {
  ncclComm_t rootComm = nullptr;
  ncclUniqueId commId;
  NCCLCHECK_TEST(ncclCommInitRankConfigHelper(
      &rootComm, numRanks, commId, globalRank, enableFastInitConfig));
  ASSERT_NE(nullptr, rootComm);
  validateCtranInitialization(rootComm, globalRank, numRanks, localRank);
  const auto statex = rootComm->ctranComm_->statex_.get();
  if (statex->nNodes() == 1) {
    NCCLCHECK_TEST(ncclCommDestroy(rootComm));
    GTEST_SKIP() << "Skip test since only one node provided";
  }

  ncclComm_t childComm = nullptr;
  ncclConfig_t childCommConfig = ncclConfigInitHelper(enableFastInitConfig);
  childCommConfig.commDesc = "child_communicator";
  int groupSize = rootComm->ctranComm_->statex_.get()->nRanks() / 2;
  int* groupRanks = new int[groupSize];
  for (int i = 0; i < groupSize; ++i) {
    *(groupRanks + i) = 2 * i + globalRank % 2;
  }
  childCommConfig.splitGroupRanks = groupRanks;
  childCommConfig.splitGroupSize = groupSize;
  NCCLCHECK_TEST(ncclCommSplit(
      rootComm, globalRank % 2, globalRank / 2, &childComm, &childCommConfig));
  ASSERT_NE(nullptr, childComm);

  void* sendBuf;
  void* recvBuf;
  ncclDataType_t dataType = ncclBfloat16;
  size_t count = 8192;
  size_t sendBytes = count * ncclTypeSize(dataType);
  size_t recvBytes = sendBytes * numRanks;
  cudaStream_t stream;

  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendBytes));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, recvBytes));
  CUDACHECK_TEST(cudaMemset(sendBuf, childComm->rank, sendBytes));
  CUDACHECK_TEST(cudaMemset(recvBuf, 0, recvBytes));
  // Ensure value has been set before colletive runs on nonblocking stream
  CUDACHECK_TEST(cudaDeviceSynchronize());
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // run baseline allgather
  auto res =
      ncclAllGather(sendBuf, recvBuf, count, dataType, childComm, stream);
  EXPECT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  for (int i = 0; i < childComm->nRanks; ++i) {
    std::vector<char> observedVals(sendBytes, rand());
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(),
        (char*)recvBuf + i * sendBytes,
        sendBytes,
        cudaMemcpyDefault));
    EXPECT_THAT(observedVals, testing::Each(i));
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));

  NCCLCHECK_TEST(ncclCommDestroy(childComm));
  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

TEST_P(NcclxBaseTestFixture, NcclCommSplitNoColor) {
  ncclComm_t rootComm = nullptr;
  ncclComm_t childComm = NCCL_COMM_NULL;
  ncclUniqueId commId;
  ncclConfig_t rootConfig = ncclConfigInitHelper(enableFastInitConfig);
  rootConfig.commDesc = "root_communicator";
  ncclConfig_t childConfig = ncclConfigInitHelper(enableFastInitConfig);
  childConfig.commDesc = "child_communicator";

  NCCLCHECK_TEST(ncclCommInitRankConfig(
      &rootComm, numRanks, commId, globalRank, &rootConfig));
  ASSERT_NE(nullptr, rootComm);
  validateCtranInitialization(rootComm, globalRank, numRanks, localRank);

  const auto statex = rootComm->ctranComm_->statex_.get();
  EXPECT_NE(statex, nullptr);

  EXPECT_EQ(statex->rank(), globalRank);
  EXPECT_EQ(statex->nRanks(), numRanks);

  // set up childConfig for split
  childConfig.splitGroupSize = numRanks / 2;
  std::vector<int> groupRanks(childConfig.splitGroupSize);
  childConfig.splitGroupRanks = groupRanks.data();
  for (int i = 0; i < childConfig.splitGroupSize; ++i) {
    groupRanks.at(i) = i * 2 + 1;
  }
  // do ncclCommSplit: even ranks have no color
  if (globalRank % 2 == 0) {
    NCCLCHECK_TEST(ncclCommSplit(
        rootComm,
        NCCL_SPLIT_NOCOLOR,
        this->globalRank,
        &childComm,
        &childConfig));
    EXPECT_EQ(childComm, (ncclComm_t)(NCCL_COMM_NULL));
  } else {
    NCCLCHECK_TEST(
        ncclCommSplit(rootComm, 1, this->globalRank, &childComm, &childConfig));
    EXPECT_NE(childComm, (ncclComm_t)(NCCL_COMM_NULL));

    // check if statex is valid
    const auto splitStatex = childComm->ctranComm_->statex_.get();
    EXPECT_EQ(splitStatex->nRanks(), this->numRanks / 2);
    EXPECT_EQ(splitStatex->rank(), this->globalRank / 2);

    // check if regular APIs provide correct results
    int nChildRanks, childRank;
    NCCLCHECK_TEST(ncclCommCount(childComm, &nChildRanks));
    EXPECT_EQ(nChildRanks, this->numRanks / 2);
    NCCLCHECK_TEST(ncclCommUserRank(childComm, &childRank));
    EXPECT_EQ(childRank, this->globalRank / 2);
  }

  NCCLCHECK_TEST(ncclCommDestroy(childComm));
  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

TEST_P(NcclxBaseTestFixture, NcclCommInitWithDifferentCommDesc) {
  ncclComm_t comm1 = nullptr, comm2 = nullptr;
  ncclUniqueId commId1, commId2;

  // Create first comm with commDesc "comm_desc_1"
  ncclConfig_t config1 = ncclConfigInitHelper(enableFastInitConfig);
  config1.commDesc = "comm_desc_1";
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm1, numRanks, commId1, globalRank, &config1));
  ASSERT_NE(nullptr, comm1);
  validateCtranInitialization(comm1, globalRank, numRanks, localRank);

  // Create second comm with commDesc "comm_desc_2"
  ncclConfig_t config2 = ncclConfigInitHelper(enableFastInitConfig);
  config2.commDesc = "comm_desc_2";
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm2, numRanks, commId2, globalRank, &config2));
  ASSERT_NE(nullptr, comm2);
  validateCtranInitialization(comm2, globalRank, numRanks, localRank);

  // Verify both comms are valid and have correct properties
  const auto statex1 = comm1->ctranComm_->statex_.get();
  const auto statex2 = comm2->ctranComm_->statex_.get();

  EXPECT_EQ(statex1->rank(), globalRank);
  EXPECT_EQ(statex1->nRanks(), numRanks);
  EXPECT_EQ(statex2->rank(), globalRank);
  EXPECT_EQ(statex2->nRanks(), numRanks);

  // Verify the comms have different hashes (indicating they are separate
  // communicators)
  EXPECT_NE(comm1->commHash, comm2->commHash);
  EXPECT_NE(statex1->commHash(), statex2->commHash());

  // Clean up both communicators
  NCCLCHECK_TEST(ncclCommDestroy(comm1));
  NCCLCHECK_TEST(ncclCommDestroy(comm2));
}

INSTANTIATE_TEST_SUITE_P(
    MyTestSuite,
    NcclxBaseTestFixture,
    testing::ValuesIn(
        {NcclxEnvs({
             {"NCCL_FASTINIT_MODE", "ring_hybrid"},
             {"NCCL_CTRAN_ENABLE", "1"},
             {"NCCL_COLLTRACE", "trace"},
             {"NCCL_COLLTRACE_USE_NEW_COLLTRACE", "1"},
         }),
         NcclxEnvs({
             {"TEST_ENABLE_FASTINIT_CONFIG", "1"},
             {"NCCL_FASTINIT_MODE", "ring_hybrid"},
             {"NCCL_CTRAN_ENABLE", "1"},
             {"NCCL_COLLTRACE", "trace"},
             {"NCCL_COLLTRACE_USE_NEW_COLLTRACE", "1"},
         }),
         NcclxEnvs({
             {"TEST_ENABLE_FASTINIT_CONFIG", "1"},
             {"NCCL_FASTINIT_MODE", "none"},
             {"NCCL_CTRAN_ENABLE", "1"},
             {"NCCL_COLLTRACE", "trace"},
             {"NCCL_COLLTRACE_USE_NEW_COLLTRACE", "1"},
         })}),
    [](const testing::TestParamInfo<NcclxBaseTestFixture::ParamType>& info) {
      // generate test-name for a given NcclxEnvs
      std::string name = "";
      for (const auto& [key, val] : info.param) {
        if (key == "NCCL_FASTINIT_MODE" ||
            key == "TEST_ENABLE_FASTINIT_CONFIG") {
          name += key;
          name += "_";
          name += val;
        }
      }
      return name;
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
