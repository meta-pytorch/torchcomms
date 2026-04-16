// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <stdlib.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

class CtranTest : public ctran::CtranDistTestFixture {
 public:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    ctran::CtranDistTestFixture::SetUp();
    ctranComm_ = makeCtranComm();
    comm = ctranComm_.get();
  }

 protected:
  std::unique_ptr<CtranComm> ctranComm_;
  CtranComm* comm{nullptr};
};

TEST_F(CtranTest, CtranAllToAllvDynamicHints) {
  EXPECT_TRUE(ctranInitialized(comm));

  meta::comms::Hints hints;
  hints.set("ncclx_alltoallv_dynamic_sendbuffs_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_recvbuffs_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_sendcounts_location", "gpu");
  hints.set("ncclx_alltoallv_dynamic_max_sendcounts_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_max_recvcounts_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_actual_recvcounts_location", "gpu");

  const size_t elemSize = commTypeSize(commInt);
  size_t maxSendcounts =
      (CTRAN_MIN_REGISTRATION_SIZE + elemSize - 1) / elemSize;
  size_t maxRecvcounts = maxSendcounts;

  auto res = ctranAllToAllvDynamicSupport(
      comm, hints, maxSendcounts, maxRecvcounts, commInt);
  EXPECT_EQ(commSuccess, res);
}

TEST_F(CtranTest, GpeNotInitialized) {
  comm->ctran_->gpe.reset();
  ASSERT_FALSE(ctranInitialized(comm));
  ASSERT_FALSE(comm->ctran_->isInitialized());
}

TEST_F(CtranTest, AlgoDeviceState) {
  const uint64_t kTestP2pDevBufSize = 65536;
  const uint64_t kTestBcastDevBufSize = 1048576;
  const bool kTestDevTraceLog = true;

  // Set env vars before creating comm so they take effect during ctranInit
  EnvRAII env1(NCCL_CTRAN_ENABLE_DEV_TRACE_LOG, kTestDevTraceLog);
  EnvRAII env2(NCCL_CTRAN_P2P_NVL_SHARED_DEVBUF_SIZE, kTestP2pDevBufSize);
  EnvRAII env3(NCCL_CTRAN_BCAST_NVL_SHARED_DEVBUF_SIZE, kTestBcastDevBufSize);

  // Create a fresh comm after env overrides (fixture's comm was created with
  // defaults)
  auto testComm = makeCtranComm();
  ASSERT_NE(nullptr, testComm->ctran_->algo->getDevState());

  // check contents of devState_d to make sure it is initialized correctly
  CtranAlgoDeviceState devState;
  CUDACHECK_TEST(cudaMemcpy(
      &devState,
      testComm->ctran_->algo->getDevState(),
      sizeof(devState),
      cudaMemcpyDeviceToHost));

  int nLocalRanks = testComm->statex_->nLocalRanks();
  const auto& statexDev = devState.statex;
  EXPECT_EQ(statexDev.rank(), testComm->statex_->rank());
  EXPECT_EQ(statexDev.nLocalRanks(), nLocalRanks);
  EXPECT_EQ(statexDev.localRank(), testComm->statex_->localRank());
  EXPECT_EQ(devState.bufSize, kTestP2pDevBufSize);
  EXPECT_EQ(devState.bcastBufSize, kTestBcastDevBufSize);
  EXPECT_EQ(devState.enableTraceLog, kTestDevTraceLog);
  EXPECT_EQ(statexDev.pid(), getpid());

  for (int i = 0; i < nLocalRanks; i++) {
    if (i == testComm->statex_->localRank()) {
      EXPECT_EQ(devState.remoteSyncsMap[i], nullptr);
      EXPECT_EQ(devState.localSyncsMap[i], nullptr);
      EXPECT_EQ(devState.remoteStagingBufsMap[i], nullptr);
      EXPECT_EQ(devState.localStagingBufsMap[i], nullptr);
      EXPECT_EQ(devState.remoteChunkStatesMap[i], nullptr);
      EXPECT_EQ(devState.localChunkStatesMap[i], nullptr);
      continue;
    }
    EXPECT_NE(devState.remoteSyncsMap[i], nullptr);
    EXPECT_NE(devState.localSyncsMap[i], nullptr);
    EXPECT_NE(devState.remoteStagingBufsMap[i], nullptr);
    EXPECT_NE(devState.localStagingBufsMap[i], nullptr);
    EXPECT_NE(devState.remoteChunkStatesMap[i], nullptr);
    EXPECT_NE(devState.localChunkStatesMap[i], nullptr);

    // Copy buffer state to host and check values are reset to default
    struct CtranAlgoDeviceSync localStateVal, remoteStateVal;
    CUDACHECK_TEST(cudaMemcpy(
        &localStateVal,
        devState.localSyncsMap[i],
        sizeof(localStateVal),
        cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaMemcpy(
        &remoteStateVal,
        devState.remoteSyncsMap[i],
        sizeof(remoteStateVal),
        cudaMemcpyDeviceToHost));
    for (int k = 0; k < CTRAN_ALGO_MAX_THREAD_BLOCKS; k++) {
      EXPECT_EQ(
          localStateVal.syncs[k].stepOnSameBlockIdx, CTRAN_ALGO_STEP_RESET);
      EXPECT_EQ(
          remoteStateVal.syncs[k].stepOnSameBlockIdx, CTRAN_ALGO_STEP_RESET);
    }
  }
}

TEST_F(CtranTest, ExchangeTmpBufComms) {
  commResult_t res;

  COMMCHECK_TEST(comm->ctran_->algo->initTmpBufs());

  // Exchange tmpbuf with all ranks on remote nodes.
  for (int i = 0; i < numRanks; i++) {
    res = comm->ctran_->algo->exchangePeerTmpbuf(i);
  }

  // Make sure all ranks on remote nodes have impoted local tmpbuf before local
  // rank frees the buffer.
  {
    auto mapper = comm->ctran_->mapper.get();
    std::vector<CtranMapperRequest*> sReqs(numRanks, nullptr),
        rReqs(numRanks, nullptr);
    CtranMapperEpochRAII epochRAII(mapper);
    for (int i = 0; i < numRanks; i++) {
      if (i != comm->statex_->rank()) {
        COMMCHECK_TEST(mapper->isendCtrl(i, &sReqs[i]));
        COMMCHECK_TEST(mapper->irecvCtrl(i, &rReqs[i]));
      }
    }
    for (int i = 0; i < numRanks; i++) {
      if (i != comm->statex_->rank()) {
        COMMCHECK_TEST(mapper->waitRequest(sReqs[i]));
        COMMCHECK_TEST(mapper->waitRequest(rReqs[i]));
      }
    }
  }

  ASSERT_EQ(res, commSuccess);

  for (int i = 0; i < numRanks; i++) {
    if (i != comm->statex_->rank()) {
      auto [_, interNodeRemoteTmpAccessKey] =
          comm->ctran_->algo->getRemoteTmpBufInfo(i);
      ASSERT_NE(CtranMapperBackend::UNSET, interNodeRemoteTmpAccessKey.backend);
    }
  }
}

TEST_F(CtranTest, ExchangeInterNodeTmpBufComms) {
  commResult_t res;

  COMMCHECK_TEST(comm->ctran_->algo->initTmpBufs());

  // Exchange tmpbuf with all ranks on remote nodes.
  res = comm->ctran_->algo->exchangeInterNodeTmpbuf();

  // Make sure all ranks on remote nodes have impoted local tmpbuf before local
  // rank frees the buffer.
  {
    auto mapper = comm->ctran_->mapper.get();
    std::vector<CtranMapperRequest*> sReqs(numRanks, nullptr),
        rReqs(numRanks, nullptr);
    CtranMapperEpochRAII epochRAII(mapper);
    for (int i = 0; i < numRanks; i++) {
      if (!comm->statex_->isSameNode(i, globalRank)) {
        COMMCHECK_TEST(mapper->isendCtrl(i, &sReqs[i]));
        COMMCHECK_TEST(mapper->irecvCtrl(i, &rReqs[i]));
      }
    }
    for (int i = 0; i < numRanks; i++) {
      if (!comm->statex_->isSameNode(i, globalRank)) {
        COMMCHECK_TEST(mapper->waitRequest(sReqs[i]));
        COMMCHECK_TEST(mapper->waitRequest(rReqs[i]));
      }
    }
  }

  ASSERT_EQ(res, commSuccess);

  for (int i = 0; i < numRanks; i++) {
    if (!comm->statex_->isSameNode(i, globalRank)) {
      auto [_, interNodeRemoteTmpAccessKey] =
          comm->ctran_->algo->getInterNodeTmpBufInfo(i);
      ASSERT_NE(CtranMapperBackend::UNSET, interNodeRemoteTmpAccessKey.backend);
    }
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
