// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <nccl.h>
#include <stdlib.h>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/mapper/CtranMapperRegMem.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "meta/wrapper/MetaFactory.h"

#define dceil(x, y) ((x / y) + !!(x % y))

class CtranTest : public NcclxBaseTest, public CtranBaseTest {
 public:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    NcclxBaseTest::SetUp();
  }

  void TearDown() override {
    if (!Test::IsSkipped()) {
      verifyPostCommResourceLeak();
    }
    NcclxBaseTest::TearDown();
  };

  void verifyPostCommResourceLeak() {
    // Check that all local/remote handles have been deregistered by CommAbort
    CtranIbSingleton& s = CtranIbSingleton::getInstance();
    EXPECT_EQ(s.getActiveRegCount(), 0);
    EXPECT_EQ(ctran::utils::getActiveIpcMemCount(), 0);
    EXPECT_EQ(ctran::utils::getActiveIpcRemMemCount(), 0);
  }
};

TEST_F(CtranTest, Initialized) {
  NcclCommRAII comm(
      globalRank, numRanks, localRank, false, nullptr, server.get());

  ASSERT_NE(nullptr, static_cast<ncclComm_t>(comm));
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  EXPECT_NE(nullptr, comm->ctranComm_->ctran_->mapper);
  EXPECT_NE(nullptr, comm->ctranComm_->ctran_->gpe);
  EXPECT_TRUE(comm->ctranComm_->ctran_->isInitialized());
  EXPECT_TRUE(ctranInitialized(comm->ctranComm_.get()));

  EXPECT_NE(nullptr, comm->ctranComm_);
}

TEST_F(CtranTest, CtranCommInitialized) {
  NcclCommRAII comm(
      globalRank, numRanks, localRank, false, nullptr, server.get());

  const auto ncclComm = static_cast<ncclComm_t>(comm);
  ASSERT_NE(nullptr, ncclComm);
  ASSERT_NE(nullptr, ncclComm->ctranComm_);

  EXPECT_EQ(comm->ctranComm_->opCount_, &ncclComm->opCount);
  EXPECT_EQ(comm->ctranComm_->config_, makeCtranConfigFrom(ncclComm));
  EXPECT_EQ(comm->ctranComm_->logMetaData_, ncclComm->logMetaData);
  EXPECT_EQ(comm->ctranComm_->runtimeConn_, ncclComm->runtimeConn);
}

TEST_F(CtranTest, CTranDisabled) {
  EnvRAII env(NCCL_CTRAN_ENABLE, false);
  ncclComm_t comm = createNcclComm(
      globalRank, numRanks, localRank, false, nullptr, server.get());

  ASSERT_NE(nullptr, static_cast<ncclComm_t>(comm));
  ASSERT_EQ(nullptr, comm->ctranComm_->ctran_);
  EXPECT_FALSE(ctranInitialized(comm->ctranComm_.get()));

  ASSERT_NE(nullptr, comm->ctranComm_->opCount_);
  EXPECT_EQ(comm->ctranComm_->opCount_, &comm->opCount);

  ASSERT_NE(nullptr, comm->ctranComm_->statex_);
  EXPECT_EQ(comm->ctranComm_->statex_, comm->ctranComm_->statex_);

  EXPECT_EQ(comm->ctranComm_->config_, makeCtranConfigFrom(comm));
  EXPECT_EQ(comm->ctranComm_->logMetaData_, comm->logMetaData);

  EXPECT_EQ(comm->ctranComm_->collTrace_, comm->collTrace);
  EXPECT_EQ(comm->ctranComm_->colltraceNew_, nullptr);

  // Expect all CTran collective support to be false
  EXPECT_FALSE(
      ctranAllGatherSupport(comm->ctranComm_.get(), NCCL_ALLGATHER_ALGO));
  EXPECT_FALSE(ctranAllReduceSupport(
      comm->ctranComm_.get(), NCCL_ALLREDUCE_ALGO::ctran));
  EXPECT_FALSE(
      ctranBroadcastSupport(comm->ctranComm_.get(), NCCL_BROADCAST_ALGO));
  EXPECT_FALSE(ctranReduceScatterSupport(
      comm->ctranComm_.get(), NCCL_REDUCESCATTER_ALGO));
  EXPECT_FALSE(ctranSendRecvSupport(0, comm->ctranComm_.get()));
  EXPECT_FALSE(ctranAllToAllSupport(
      1048576, commInt, comm->ctranComm_.get(), NCCL_ALLTOALL_ALGO::ctran));
  EXPECT_FALSE(ctranAllToAllvSupport(comm->ctranComm_.get()));
  meta::comms::Hints hints;

  size_t maxSendcounts =
      dceil(CTRAN_MIN_REGISTRATION_SIZE, commTypeSize(commInt));
  size_t maxRecvcounts =
      dceil(CTRAN_MIN_REGISTRATION_SIZE, commTypeSize(commInt));

  auto res = ctranAllToAllvDynamicSupport(
      comm->ctranComm_.get(), hints, maxSendcounts, maxRecvcounts, commInt);
  EXPECT_EQ(commInvalidUsage, res);

  hints.set("ncclx_alltoallv_dynamic_sendbuffs_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_recvbuffs_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_sendcounts_location", "gpu");
  hints.set("ncclx_alltoallv_dynamic_max_sendcounts_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_max_recvcounts_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_actual_recvcounts_location", "gpu");

  maxSendcounts = CTRAN_MIN_REGISTRATION_SIZE / commTypeSize(commInt);
  maxRecvcounts = CTRAN_MIN_REGISTRATION_SIZE / commTypeSize(commInt);
  res = ctranAllToAllvDynamicSupport(
      comm->ctranComm_.get(), hints, maxSendcounts, maxRecvcounts, commInt);
  EXPECT_EQ(commInvalidUsage, res);
  // Expect ncclCommDestroy to succeed even ctran is not initialized

  finalizeNcclComm(globalRank, server.get());
  res = ncclToMetaComm(ncclCommDestroy(comm));
  ASSERT_EQ(res, commSuccess);
}

TEST_F(CtranTest, CtranAllToAllvDynamicHints) {
  NcclCommRAII comm(
      globalRank, numRanks, localRank, false, nullptr, server.get());
  ASSERT_NE(nullptr, static_cast<ncclComm_t>(comm));
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);
  EXPECT_TRUE(ctranInitialized(comm->ctranComm_.get()));

  meta::comms::Hints hints;
  hints.set("ncclx_alltoallv_dynamic_sendbuffs_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_recvbuffs_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_sendcounts_location", "gpu");
  hints.set("ncclx_alltoallv_dynamic_max_sendcounts_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_max_recvcounts_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_actual_recvcounts_location", "gpu");

  size_t maxSendcounts =
      dceil(CTRAN_MIN_REGISTRATION_SIZE, commTypeSize(commInt));
  size_t maxRecvcounts =
      dceil(CTRAN_MIN_REGISTRATION_SIZE, commTypeSize(commInt));

  auto res = ctranAllToAllvDynamicSupport(
      comm->ctranComm_.get(), hints, maxSendcounts, maxRecvcounts, commInt);
  EXPECT_EQ(commSuccess, res);
}

TEST_F(CtranTest, CTranAllGatherOverrideConfig) {
  EnvRAII env(NCCL_ALLGATHER_ALGO, NCCL_ALLGATHER_ALGO::orig);
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.ncclAllGatherAlgo = "ctring";
  ncclComm_t comm = createNcclComm(
      globalRank, numRanks, localRank, false, &config, server.get());

  ASSERT_NE(nullptr, static_cast<ncclComm_t>(comm));
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);
  EXPECT_TRUE(ctranInitialized(comm->ctranComm_.get()));

  auto algo = comm->ctranComm_->ctran_->algo->getAllGatherAlgo();
  EXPECT_TRUE(algo == NCCL_ALLGATHER_ALGO::ctring);

  finalizeNcclComm(globalRank, server.get());
  ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

TEST_F(CtranTest, CTranAllGatherOverrideConfigSplitComm) {
  EnvRAII env(NCCL_ALLGATHER_ALGO, NCCL_ALLGATHER_ALGO::orig);
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclComm_t comm;
  std::string algoStr = "ctring";
  config.ncclAllGatherAlgo = algoStr.c_str();
  comm = createNcclComm(
      globalRank, numRanks, localRank, false, &config, server.get());

  // change pointer to be an invalid value, should still work since
  // we copy the string to the config
  algoStr = "badconfig";
  ncclComm_t childComm;
  NCCLCHECK_TEST(
      ncclCommSplit(comm, globalRank % 2, globalRank / 2, &childComm, NULL));
  ASSERT_NE(nullptr, childComm);

  ASSERT_NE(nullptr, static_cast<ncclComm_t>(comm));
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);
  EXPECT_TRUE(ctranInitialized(comm->ctranComm_.get()));

  auto algo = comm->ctranComm_->ctran_->algo->getAllGatherAlgo();
  EXPECT_TRUE(algo == NCCL_ALLGATHER_ALGO::ctring);

  auto childAlgo = childComm->ctranComm_->ctran_->algo->getAllGatherAlgo();
  EXPECT_TRUE(childAlgo == NCCL_ALLGATHER_ALGO::ctring);

  finalizeNcclComm(globalRank, server.get());
  ASSERT_EQ(ncclCommDestroy(childComm), ncclSuccess);
  ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

TEST_F(CtranTest, GpeNotInitialized) {
  ncclComm_t comm = createNcclComm(
      globalRank, numRanks, localRank, false, nullptr, server.get());

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  comm->ctranComm_->ctran_->gpe.reset();
  ASSERT_FALSE(ctranInitialized(comm->ctranComm_.get()));
  ASSERT_FALSE(comm->ctranComm_->ctran_->isInitialized());

  finalizeNcclComm(globalRank, server.get());
  // Cleanup comm resource, expect ncclSuccess even gpe is not
  // initialized
  ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

TEST_F(CtranTest, PostCommDestroy) {
  ncclComm_t comm = createNcclComm(
      globalRank, numRanks, localRank, false, nullptr, server.get());

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  finalizeNcclComm(globalRank, server.get());
  NCCLCHECK_TEST(ncclCommDestroy(comm));

  ASSERT_FALSE(ctranInitialized(comm->ctranComm_.get()));
  // Do not check ctran->isInitialized() as it is already destroyed
}

TEST_F(CtranTest, AlgoDeviceState) {
  const uint64_t kTestP2pDevBufSize = 65536;
  const uint64_t kTestBcastDevBufSize = 1048576;
  const bool kTestDevTraceLog = true;

  EnvRAII env1(NCCL_CTRAN_ENABLE_DEV_TRACE_LOG, kTestDevTraceLog);
  EnvRAII env2(NCCL_CTRAN_P2P_NVL_SHARED_DEVBUF_SIZE, kTestP2pDevBufSize);
  EnvRAII env3(NCCL_CTRAN_BCAST_NVL_SHARED_DEVBUF_SIZE, kTestBcastDevBufSize);

  NcclCommRAII comm(
      globalRank, numRanks, localRank, false, nullptr, server.get());

  ASSERT_NE(nullptr, static_cast<ncclComm_t>(comm));
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  ASSERT_NE(nullptr, comm->ctranComm_->ctran_->algo->getDevState());

  // check contents of devState_d to make sure it is initialized correctly
  CtranAlgoDeviceState devState;
  CUDACHECK_TEST(cudaMemcpy(
      &devState,
      comm->ctranComm_->ctran_->algo->getDevState(),
      sizeof(devState),
      cudaMemcpyDeviceToHost));

  int nLocalRanks = comm->ctranComm_->statex_->nLocalRanks();
  const auto& statexDev = devState.statex;
  EXPECT_EQ(statexDev.rank(), comm->ctranComm_->statex_->rank());
  EXPECT_EQ(statexDev.nLocalRanks(), nLocalRanks);
  EXPECT_EQ(statexDev.localRank(), comm->ctranComm_->statex_->localRank());
  EXPECT_EQ(devState.bufSize, kTestP2pDevBufSize);
  EXPECT_EQ(devState.bcastBufSize, kTestBcastDevBufSize);
  EXPECT_EQ(devState.enableTraceLog, kTestDevTraceLog);
  EXPECT_EQ(statexDev.pid(), getpid());

  for (int i = 0; i < nLocalRanks; i++) {
    if (i == comm->ctranComm_->statex_->localRank()) {
      EXPECT_EQ(devState.remoteSyncsMap[i], nullptr);
      EXPECT_EQ(devState.localSyncsMap[i], nullptr);
      EXPECT_EQ(devState.remoteStagingBufsMap[i], nullptr);
      EXPECT_EQ(devState.localStagingBufsMap[i], nullptr);
      continue;
    }
    EXPECT_NE(devState.remoteSyncsMap[i], nullptr);
    EXPECT_NE(devState.localSyncsMap[i], nullptr);
    EXPECT_NE(devState.remoteStagingBufsMap[i], nullptr);
    EXPECT_NE(devState.localStagingBufsMap[i], nullptr);

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

TEST_F(CtranTest, RegMemReuseInMultiComms) {
  constexpr int numComms = 5;

  std::vector<ncclComm_t> comms(numComms, NCCL_COMM_NULL);
  std::vector<cudaStream_t> streams(numComms, nullptr);
  for (int c = 0; c < numComms; c++) {
    // In ring_hybrid mode, only create the first communicator using
    // createNcclComm; the rest are created using ncclCommSplit to leverage the
    // info from the first communicator.
    if (c == 0) {
      comms[c] = createNcclComm(
          globalRank, numRanks, localRank, false, nullptr, server.get());
    } else {
      NCCLCHECK_TEST(ncclCommSplit(
          comms[0], 0, globalRank, &comms[c], nullptr /* config */));
    }

    CUDACHECK_TEST(cudaStreamCreate(&streams[c]));

    ASSERT_NE(nullptr, comms[c]);
    ASSERT_NE(nullptr, comms[c]->ctranComm_->ctran_);

    if (!ctranAllGatherSupport(
            comms[c]->ctranComm_.get(), NCCL_ALLGATHER_ALGO)) {
      finalizeNcclComm(globalRank, server.get());
      ncclCommAbort(comms[c]);
      GTEST_SKIP() << "ctranAllGather is not supported. Skip test";
    }
  }

  auto regCache = CtranMapperRegCache::getInstance();
  ASSERT_NE(regCache, nullptr);

  auto allHandles = regCache->getSegments();
  size_t commNumHdls = allHandles.size();

  void* buf = nullptr;
  constexpr size_t count = 4096;
  size_t bufSize = count * sizeof(int) * numRanks * numComms;
  NCCLCHECK_TEST(ncclMemAlloc(&buf, bufSize));
  ASSERT_NE(buf, nullptr);

  // Assign value and register buffer with all comms
  std::vector<void*> hdls(numComms, nullptr);
  std::vector<void*> recvBufs(numComms, nullptr);
  std::vector<void*> sendBufs(numComms, nullptr);

  CUDACHECK_TEST(cudaMemset(buf, 0, bufSize));
  for (int c = 0; c < numComms; c++) {
    recvBufs[c] = (char*)buf + c * count * sizeof(int) * numRanks;
    sendBufs[c] = (char*)recvBufs[c] + count * sizeof(int) * globalRank;

    // Assign different value for each comm
    std::vector<int> vals(count, 0);
    for (int i = 0; i < count; i++) {
      vals[i] = globalRank + c;
    }

    CUDACHECK_TEST(cudaMemcpy(
        sendBufs[c], vals.data(), count * sizeof(int), cudaMemcpyHostToDevice));

    // Register entire buffer range with all comms
    NCCLCHECK_TEST(ncclCommRegister(comms[c], buf, bufSize, &hdls[c]));
    ASSERT_NE(hdls[c], nullptr);
  }
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Run collective on each communicator concurrently, each will be handled by a
  // separate GPE thread
  for (int c = 0; c < numComms; c++) {
    COMMCHECK_TEST(ctranAllGather(
        sendBufs[c],
        recvBufs[c],
        count,
        commInt,
        comms[c]->ctranComm_.get(),
        streams[c],
        NCCL_ALLGATHER_ALGO));
  }

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Expect all communicators shared the same buffer registration
  // Thus, we'd see only 1 additional handle in the cache
  allHandles = regCache->getSegments();
  EXPECT_EQ(allHandles.size(), 1 + commNumHdls);

  // Check that all values are correct
  for (int c = 0; c < numComms; c++) {
    std::vector<int> expVals(count * numRanks, 0);
    std::vector<int> vals(count * numRanks, -1);
    CUDACHECK_TEST(cudaMemcpy(
        vals.data(),
        recvBufs[c],
        count * sizeof(int) * numRanks,
        cudaMemcpyDeviceToHost));

    for (int i = 0; i < numRanks; i++) {
      for (int j = 0; j < count; j++) {
        expVals[i * count + j] = i + c;
      }
    }

    EXPECT_THAT(vals, ::testing::ElementsAreArray(expVals))
        << " at comm " << c << " buf " << recvBufs[c];
  }

  finalizeNcclComm(globalRank, server.get());
  // Deregister buffer and destroy communicator
  for (int c = 0; c < numComms; c++) {
    NCCLCHECK_TEST(ncclCommDeregister(comms[c], hdls[c]));
    NCCLCHECK_TEST(ncclCommDestroy(comms[c]));
    CUDACHECK_TEST(cudaStreamDestroy(streams[c]));
  }

  NCCLCHECK_TEST(ncclMemFree(buf));
}

TEST_F(CtranTest, CommAbort) {
  ncclResult_t res;
  ncclComm_t comm = createNcclComm(
      globalRank, numRanks, localRank, false, nullptr, server.get());

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  finalizeNcclComm(globalRank, server.get());
  // Expect shared resource has been released properly
  res = ncclCommAbort(comm);
  ASSERT_EQ(res, ncclSuccess);
}

TEST_F(CtranTest, CommAbortWithRegMem) {
  ncclResult_t res;
  ncclComm_t comm = createNcclComm(
      globalRank, numRanks, localRank, false, nullptr, server.get());
  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  CtranMapperRegCache* regCache = CtranMapperRegCache::getInstance().get();
  ASSERT_NE(regCache, nullptr);

  auto allHandles = regCache->getSegments();
  size_t commNumHdls = allHandles.size();

  void* buf = nullptr;
  constexpr size_t count = 8192;
  size_t bufSize = count * sizeof(int) * numRanks;
  NCCLCHECK_TEST(ncclMemAlloc(&buf, bufSize));
  ASSERT_NE(buf, nullptr);
  void* hdl = nullptr;

  ncclCommRegister(comm, buf, bufSize, &hdl);
  ASSERT_NE(hdl, nullptr);

  if (!ctranAllGatherSupport(comm->ctranComm_.get(), NCCL_ALLGATHER_ALGO)) {
    finalizeNcclComm(globalRank, server.get());
    ncclCommAbort(comm);
    GTEST_SKIP() << "ctranAllGather is not supported. Skip test";
  }
  // Run a collective to ensure buffer has been remote registered
  COMMCHECK_TEST(ctranAllGather(
      buf,
      buf,
      count,
      commInt,
      comm->ctranComm_.get(),
      0,
      NCCL_ALLGATHER_ALGO));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  allHandles = regCache->getSegments();
  EXPECT_EQ(allHandles.size(), 1 + commNumHdls);

  finalizeNcclComm(globalRank, server.get());
  // Do not deregister buffer; expect commAbort to finish properly.
  // Resource leak is checked in TearDown()
  res = ncclCommAbort(comm);
  ASSERT_EQ(res, ncclSuccess);

  // Explicit trigger regCache destruction before check
  EXPECT_EQ(regCache->destroy(), commSuccess);

  NCCLCHECK_TEST(ncclMemFree(buf));
}

/*
 * This test has been commented out as NCCL_COMM_ABORT_SCOPE is not
 * supported in v2_25. We can re-enable this test once we have the
 * functionality or remove it if we decide to deprecate it.
TEST_F(CtranTest, CommAbortScopeNone) {
  ncclResult_t res;
  ncclComm_t comm = createNcclComm(globalRank, numRanks, localRank);
  EnvRAII env(NCCL_COMM_ABORT_SCOPE, NCCL_COMM_ABORT_SCOPE::none);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  // Expect no op for commAbort at none scope
  res = ncclCommAbort(comm);
  ASSERT_EQ(res, ncclSuccess);

  CtranIbSingleton& s = CtranIbSingleton::getInstance();

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  ASSERT_EQ(s.destroy(), ncclSuccess);

  // Except no warning nor error message is printed
  std::string output = testing::internal::GetCapturedStdout();
  std::string error = testing::internal::GetCapturedStderr();
  EXPECT_THAT(output, testing::Not(testing::HasSubstr("WARN")));
  EXPECT_THAT(output, testing::Not(testing::HasSubstr("ERROR")));
  EXPECT_THAT(error, testing::Not(testing::HasSubstr("WARN")));
  EXPECT_THAT(error, testing::Not(testing::HasSubstr("ERROR")));
}
*/

TEST_F(CtranTest, ExchangeTmpBufComms) {
  commResult_t res;
  ncclComm_t comm = createNcclComm(
      globalRank, numRanks, localRank, false, nullptr, server.get());
  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  COMMCHECK_TEST(comm->ctranComm_->ctran_->algo->initTmpBufs());

  // Exchange tmpbuf with all ranks on remote nodes.
  for (int i = 0; i < numRanks; i++) {
    res = comm->ctranComm_->ctran_->algo->exchangePeerTmpbuf(i);
  }

  // Make sure all ranks on remote nodes have impoted local tmpbuf before local
  // rank frees the buffer.
  {
    auto mapper = comm->ctranComm_->ctran_->mapper.get();
    std::vector<CtranMapperRequest*> sReqs(numRanks, nullptr),
        rReqs(numRanks, nullptr);
    CtranMapperEpochRAII epochRAII(mapper);
    for (int i = 0; i < numRanks; i++) {
      if (i != comm->ctranComm_->statex_->rank()) {
        COMMCHECK_TEST(mapper->isendCtrl(i, &sReqs[i]));
        COMMCHECK_TEST(mapper->irecvCtrl(i, &rReqs[i]));
      }
    }
    for (int i = 0; i < numRanks; i++) {
      if (i != comm->ctranComm_->statex_->rank()) {
        COMMCHECK_TEST(mapper->waitRequest(sReqs[i]));
        COMMCHECK_TEST(mapper->waitRequest(rReqs[i]));
      }
    }
  }

  ASSERT_EQ(res, commSuccess);

  for (int i = 0; i < numRanks; i++) {
    if (i != comm->ctranComm_->statex_->rank()) {
      auto [_, interNodeRemoteTmpAccessKey] =
          comm->ctranComm_->ctran_->algo->getRemoteTmpBufInfo(i);
      ASSERT_NE(CtranMapperBackend::UNSET, interNodeRemoteTmpAccessKey.backend);
    }
  }

  finalizeNcclComm(globalRank, server.get());
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(CtranTest, ExchangeInterNodeTmpBufComms) {
  commResult_t res;
  ncclComm_t comm = createNcclComm(
      globalRank, numRanks, localRank, false, nullptr, server.get());
  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  COMMCHECK_TEST(comm->ctranComm_->ctran_->algo->initTmpBufs());

  // Exchange tmpbuf with all ranks on remote nodes.
  res = comm->ctranComm_->ctran_->algo->exchangeInterNodeTmpbuf();

  // Make sure all ranks on remote nodes have impoted local tmpbuf before local
  // rank frees the buffer.
  {
    auto mapper = comm->ctranComm_->ctran_->mapper.get();
    std::vector<CtranMapperRequest*> sReqs(numRanks, nullptr),
        rReqs(numRanks, nullptr);
    CtranMapperEpochRAII epochRAII(mapper);
    for (int i = 0; i < numRanks; i++) {
      if (!comm->ctranComm_->statex_->isSameNode(i, globalRank)) {
        COMMCHECK_TEST(mapper->isendCtrl(i, &sReqs[i]));
        COMMCHECK_TEST(mapper->irecvCtrl(i, &rReqs[i]));
      }
    }
    for (int i = 0; i < numRanks; i++) {
      if (!comm->ctranComm_->statex_->isSameNode(i, globalRank)) {
        COMMCHECK_TEST(mapper->waitRequest(sReqs[i]));
        COMMCHECK_TEST(mapper->waitRequest(rReqs[i]));
      }
    }
  }

  ASSERT_EQ(res, commSuccess);

  for (int i = 0; i < numRanks; i++) {
    if (!comm->ctranComm_->statex_->isSameNode(i, globalRank)) {
      auto [_, interNodeRemoteTmpAccessKey] =
          comm->ctranComm_->ctran_->algo->getInterNodeTmpBufInfo(i);
      ASSERT_NE(CtranMapperBackend::UNSET, interNodeRemoteTmpAccessKey.backend);
    }
  }

  finalizeNcclComm(globalRank, server.get());
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(CtranTest, CommFailureWithInvalidTopology) {
  const std::string invalidTopoFilepath = "/tmp/invalid_comm_topology.txt";
  std::ofstream invalidFile(invalidTopoFilepath);
  invalidFile << "DEVICE_NAME=testhost.facebook.com" << std::endl;
  // Invalid topology format - should have exactly 4 parts separated by '/'
  invalidFile << "DEVICE_BACKEND_NETWORK_TOPOLOGY=part1/part2/part3"
              << std::endl;
  invalidFile.close();

  EnvRAII topoFileEnv(NCCL_TOPO_FILE_PATH, invalidTopoFilepath);
  // Disable NCCL_IGNORE_TOPO_LOAD_FAILURE to ensure topology loading failure
  // causes comm creation to fail
  EnvRAII ignoreTopoFailureEnv(NCCL_IGNORE_TOPO_LOAD_FAILURE, false);

  ncclComm_t comm = nullptr;
  EXPECT_THROW(
      {
        try {
          createNcclComm(
              globalRank, numRanks, localRank, true, nullptr, server.get());
        } catch (const std::runtime_error& e) {
          EXPECT_THAT(
              e.what(),
              testing::HasSubstr("Failed, NCCL error: internal error"));
          throw;
        }
      },
      std::runtime_error);

  if (comm != nullptr) {
    finalizeNcclComm(globalRank, server.get());
    ncclCommDestroy(comm);
  }

  // Clean up temporary files
  unlink(invalidTopoFilepath.c_str());
}

TEST_F(CtranTest, IgnoreCommFailureWithInvalidTopology) {
  EnvRAII ignoreTopoFailureEnv(NCCL_IGNORE_TOPO_LOAD_FAILURE, true);
  const std::string dummyTopoFilepath = "/tmp/invalid_comm_topology.txt";
  std::ofstream dummyFile(dummyTopoFilepath);
  dummyFile << "DEVICE_NAME=testhost.facebook.com" << std::endl;
  dummyFile << "DEVICE_BACKEND_NETWORK_TOPOLOGY=part1/part2/part3" << std::endl;
  dummyFile.close();

  SysEnvRAII emptyTopoFileEnv("NCCL_TOPO_FILE_PATH", dummyTopoFilepath);

  ncclComm_t comm = nullptr;
  comm = createNcclComm(
      globalRank, numRanks, localRank, true, nullptr, server.get());

  EXPECT_FALSE(comm == nullptr)
      << "Communicator creation should not fail when NCCL_IGNORE_TOPO_LOAD_FAILURE=1";

  if (comm != nullptr) {
    finalizeNcclComm(globalRank, server.get());
    ncclCommDestroy(comm);
  }

  // Clean up temporary files
  unlink(dummyTopoFilepath.c_str());
}

// FIXME: This test can be better covered in CtranCommTest.cc but the current
// standalone CtranComm has issue. Move the test once fix.
TEST_F(CtranTest, CtranOpCount) {
  ncclComm_t comm = nullptr;
  comm = createNcclComm(
      globalRank, numRanks, localRank, true, nullptr, server.get());

  // Update opCount mimic Ctran collectives
  constexpr int kNumOps = 10;
  auto ctranComm = comm->ctranComm_.get();
  for (int i = 0; i < kNumOps; ++i) {
    ctranComm->ctran_->updateOpCount();
  }

  // Expect that both external NCCL opCount and ctranOpCount have been updated
  EXPECT_EQ(ctranComm->ctran_->getOpCount(), kNumOps);
  EXPECT_EQ(ctranComm->ctran_->getCtranOpCount(), kNumOps);

  // Update opCount mimic baseline collectives
  for (int i = 0; i < kNumOps; ++i) {
    comm->opCount++;
  }

  // Expect only external NCCL opCount has been updated
  EXPECT_EQ(ctranComm->ctran_->getOpCount(), kNumOps * 2);
  EXPECT_EQ(ctranComm->ctran_->getCtranOpCount(), kNumOps);

  if (comm != nullptr) {
    finalizeNcclComm(globalRank, server.get());
    ncclCommDestroy(comm);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
