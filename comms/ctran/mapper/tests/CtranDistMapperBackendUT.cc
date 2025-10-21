// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "nccl.h"

// Note that The bufSize should be less than NCCL_CTRAN_IB_QP_SCALING_THRESHOLD
// and putTimes should be less than NCCL_CTRAN_IB_QP_MAX_MSGS to allow fast path
// to be enabled.
constexpr size_t bufSize = 8192;
// In issuePutsAndWaitForCompletion, we issue putTimes puts to each peer to test
// different iput interfaces. If ibFastPath=false, only the last put will notify
// the peer; otherwise, all puts will notify the peer.
constexpr int putTimes = 3;
using ctran::algos::GpeKernelSync;
extern __global__ void
waitSigTestKernel(GpeKernelSync* sync, uint64_t* data, int cmpVal);

class CtranDistMapperBackendTest : public NcclxBaseTest {
 public:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    NcclxBaseTest::SetUp();
    commDeprecated_ = createNcclComm(
        globalRank, numRanks, localRank, false, nullptr, server.get());
    comm_ = commDeprecated_->ctranComm_.get();

    if (!ctranInitialized(comm_) || !comm_->ctran_->mapper->hasBackend()) {
      GTEST_SKIP()
          << "Ctran is not initialized or backend is not available.  Skip test.";
    }

    mapper = comm_->ctran_->mapper.get();
    NCCLCHECK_TEST(ncclMemAlloc(&buf, bufSize));
    COMMCHECK_TEST(comm_->ctran_->commRegister(buf, bufSize, &handle));

    bool localReg = false;
    COMMCHECK_TEST(mapper->searchRegHandle(buf, bufSize, &sendHdl, &localReg));
    ASSERT_EQ(localReg, false);
  }

  void TearDown() override {
    finalizeNcclComm(globalRank, server.get());
    NCCLCHECK_TEST(ncclCommDestroy(commDeprecated_));
    NcclxBaseTest::TearDown();
  }

  void PreConnectAllPeers() {
    std::unordered_set<int> peers;
    for (int peer = 0; peer < comm_->statex_->nRanks(); peer++) {
      if (peer != comm_->statex_->rank()) {
        peers.insert(peer);
      }
    }
    ASSERT_EQ(mapper->preConnect(peers), commSuccess);
  }

  void mockNvlPutComplete(KernelElem* kElem) {
    // Directly complete posted kElem to mock NVL put completion, since we don't
    // launch kernel to really handle the NVL put.
    if (kElem) {
      kElem->setStatus(KernelElem::ElemStatus::DONE);
    }
  }

  void issuePutsAndWaitForCompletion(
      const void* buf,
      void* sendHdl,
      const std::vector<int>& ranks,
      std::vector<void*>& remoteBufs,
      std::vector<struct CtranMapperRemoteAccessKey>& remoteAccessKeys,
      CtranIbConfig* config,
      const bool lowlatency,
      const bool ibFastPath = false) {
    const auto myRank = comm_->statex_.get()->rank();

    for (auto peer : ranks) {
      if (peer == myRank) {
        continue; // skip self (local
      }

      KernelElem* kElem = nullptr;
      if (remoteAccessKeys[peer].backend == CtranMapperBackend::NVL) {
        kElem = new KernelElem();
        kElem->ngroups = 1;
      }

      // Test pass null request pointer (used in a2av dynamic)
      ASSERT_EQ(
          mapper->iput(
              buf,
              remoteBufs[peer],
              bufSize,
              peer,
              CtranMapperConfig{
                  .memHdl_ = sendHdl,
                  .remoteAccessKey_ = remoteAccessKeys[peer],
                  .notify_ = ibFastPath,
                  .kernElem_ = kElem,
                  .ibConfig_ = config,
                  .ibFastPath_ = ibFastPath},
              static_cast<CtranMapperRequest*>(nullptr)),
          commSuccess);

      // Test pass nullptr address (request will be dynamically allocated in
      // iput)
      CtranMapperRequest* req = nullptr;
      ASSERT_EQ(
          mapper->iput(
              buf,
              remoteBufs[peer],
              bufSize,
              peer,
              CtranMapperConfig{
                  .memHdl_ = sendHdl,
                  .remoteAccessKey_ = remoteAccessKeys[peer],
                  .notify_ = ibFastPath,
                  .kernElem_ = kElem,
                  .ibConfig_ = config,
                  .ibFastPath_ = ibFastPath},
              &req),
          commSuccess);
      ASSERT_NE(req, nullptr);

      mockNvlPutComplete(kElem);
      ASSERT_EQ(mapper->waitRequest(req), commSuccess);
      // req is dynamically allocated in iput, so we need to delete it.
      delete req;

      // Test pass user allocated request
      CtranMapperRequest user_allocated_req;
      if (lowlatency) {
        ASSERT_EQ(
            mapper->iput<LowLatencyCollConfig>(
                buf,
                remoteBufs[peer],
                bufSize,
                peer,
                CtranMapperConfig{
                    .memHdl_ = sendHdl,
                    .remoteAccessKey_ = remoteAccessKeys[peer],
                    .notify_ = true,
                    .kernElem_ = kElem,
                    .ibConfig_ = config,
                    .ibFastPath_ = ibFastPath},
                &user_allocated_req),
            commSuccess);
      } else {
        ASSERT_EQ(
            mapper->iput<DefaultPerfCollConfig>(
                buf,
                remoteBufs[peer],
                bufSize,
                peer,
                CtranMapperConfig{
                    .memHdl_ = sendHdl,
                    .remoteAccessKey_ = remoteAccessKeys[peer],
                    .notify_ = true,
                    .kernElem_ = kElem,
                    .ibConfig_ = config,
                    .ibFastPath_ = ibFastPath},
                &user_allocated_req),
            commSuccess);
      }
      mockNvlPutComplete(kElem);
      ASSERT_EQ(mapper->waitRequest(&user_allocated_req), commSuccess);
    }
  }

  void waitSignal(const void* buf, uint64_t cmpVal) {
    void* syncPtr = nullptr;
    cudaHostAlloc(&syncPtr, sizeof(GpeKernelSync), cudaHostAllocDefault);
    GpeKernelSync* sync = reinterpret_cast<GpeKernelSync*>(syncPtr);
    new (sync) GpeKernelSync(1);
    std::array<void*, 3> kernArgs;
    dim3 grid = {1, 1, 1};
    dim3 blocks = {1, 1, 1};
    kernArgs.at(0) = &sync;
    kernArgs.at(1) = &buf;
    kernArgs.at(2) = &cmpVal;
    cudaLaunchKernel(
        (const void*)waitSigTestKernel, grid, blocks, kernArgs.data(), 0, 0);
    while (!sync->isComplete(0)) {
      std::this_thread::yield();
    }
    cudaFreeHost(syncPtr);
  }

  CtranMapper* mapper{nullptr};
  void* buf{nullptr};
  void* handle{nullptr};
  void* sendHdl{nullptr};

 protected:
  ncclComm_t commDeprecated_{nullptr};
  CtranComm* comm_{nullptr};
};

class CtranDistMapperBackendPerfConfigTestParam
    : public CtranDistMapperBackendTest,
      public ::testing::WithParamInterface<bool> {};
/*
 * Send a message to all local ranks using the IB backend.
 */
TEST_P(CtranDistMapperBackendPerfConfigTestParam, IntraNodeUseIb) {
  const bool lowlatency = GetParam();
  const auto& statex = comm_->statex_.get();
  const int nLocalRanks = statex->nLocalRanks();
  const int myRank = statex->rank();
  std::vector<int> ranks(nLocalRanks);

  std::vector<void*> remoteBufs(statex->nRanks(), nullptr);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(
      statex->nRanks());

  // get local peer ranks
  for (int i = 0; i < nLocalRanks; i++) {
    ranks[i] = statex->localRankToRank(i);
  }

  {
    CtranMapperEpochRAII epochRAII(mapper);

    if (lowlatency) {
      PreConnectAllPeers();
    }

    // Exchange control messages with local ranks.
    // Force issue all puts via IB
    ASSERT_EQ(
        mapper->allGatherCtrl(
            buf,
            sendHdl,
            ranks,
            remoteBufs,
            remoteAccessKeys,
            CtranMapperBackend::IB),
        commSuccess);

    intraNodeBarrier(commDeprecated_);

    ASSERT_EQ(mapper->iPutCount[CtranMapperBackend::IB], 0);
    ASSERT_EQ(mapper->iPutCount[CtranMapperBackend::NVL], 0);

    // Remove myself from the list of peers
    for (auto it = ranks.begin(); it != ranks.end();) {
      if (*it == myRank) {
        it = ranks.erase(it);
        break;
      } else {
        it++;
      }
    }

    std::vector<CtranMapperNotify> notifies(ranks.size());
    ASSERT_EQ(mapper->initNotifyBatchIB(ranks, notifies), commSuccess);
    issuePutsAndWaitForCompletion(
        buf, sendHdl, ranks, remoteBufs, remoteAccessKeys, nullptr, lowlatency);

    ASSERT_EQ(
        mapper->iPutCount[CtranMapperBackend::IB], ranks.size() * putTimes);
    ASSERT_EQ(mapper->iPutCount[CtranMapperBackend::NVL], 0);

    if (lowlatency) {
      ASSERT_EQ(
          mapper->waitAllNotifies<LowLatencyCollConfig>(notifies), commSuccess);
    } else {
      ASSERT_EQ(
          mapper->waitAllNotifies<DefaultPerfCollConfig>(notifies),
          commSuccess);
    }
  }

  // Ensure remote rank have finished iput access before local deregister
  ASSERT_EQ(mapper->intraBarrier(), commSuccess);

  COMMCHECK_TEST(comm_->ctran_->commDeregister(handle));
  NCCLCHECK_TEST(ncclMemFree(buf));
}

/*
 * Send a message to all local ranks using the NVL backend. Disabled because
 * CTRAN doesn't support NVL iput yet.
 */
TEST_P(CtranDistMapperBackendPerfConfigTestParam, IntraNodeUseNvl) {
  const auto perfconfig = GetParam();
  const auto& statex = comm_->statex_.get();
  const int nLocalRanks = statex->nLocalRanks();
  std::vector<int> ranks(nLocalRanks);

  std::vector<void*> remoteBufs(statex->nRanks(), nullptr);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(
      statex->nRanks());

  // get local ranks
  for (int i = 0; i < nLocalRanks; i++) {
    ranks[i] = statex->localRankToRank(i);
  }

  {
    CtranMapperEpochRAII epochRAII(mapper);

    // Exchange control messages with local ranks
    // Force issue all puts via NVL
    ASSERT_EQ(
        mapper->allGatherCtrl(
            buf,
            sendHdl,
            ranks,
            remoteBufs,
            remoteAccessKeys,
            CtranMapperBackend::NVL),
        commSuccess);

    intraNodeBarrier(commDeprecated_);

    ASSERT_EQ(mapper->iPutCount[CtranMapperBackend::IB], 0);
    ASSERT_EQ(mapper->iPutCount[CtranMapperBackend::NVL], 0);

    issuePutsAndWaitForCompletion(
        buf, sendHdl, ranks, remoteBufs, remoteAccessKeys, nullptr, perfconfig);
  }

  ASSERT_EQ(mapper->iPutCount[CtranMapperBackend::IB], 0);
  ASSERT_EQ(
      mapper->iPutCount[CtranMapperBackend::NVL], (nLocalRanks - 1) * putTimes);

  // Ensure remote rank have finished iput access before local deregister
  ASSERT_EQ(mapper->intraBarrier(), commSuccess);

  COMMCHECK_TEST(comm_->ctran_->commDeregister(handle));
  NCCLCHECK_TEST(ncclMemFree(buf));
}

/*
 * Send a message to all ranks using the IB backend (for remote ranks) and the
 * NVL backend (for local ranks).
 */

TEST_P(
    CtranDistMapperBackendPerfConfigTestParam,
    IntraAndInterNodeUseIbAndNvl) {
  const bool lowlatency = GetParam();
  const auto& statex = comm_->statex_.get();
  const int nRanks = statex->nRanks();
  const int nLocalRanks = statex->nLocalRanks();
  std::vector<int> ranks(nRanks);

  std::vector<void*> remoteBufs(statex->nRanks(), nullptr);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(
      statex->nRanks());

  // get ranks
  for (int i = 0; i < nRanks; i++) {
    ranks[i] = i;
  }

  {
    CtranMapperEpochRAII epochRAII(mapper);

    if (lowlatency) {
      PreConnectAllPeers();
    }

    // exchange control messages with all ranks
    ASSERT_EQ(
        mapper->allGatherCtrl(
            buf, sendHdl, ranks, remoteBufs, remoteAccessKeys),
        commSuccess);

    intraNodeBarrier(commDeprecated_);

    ASSERT_EQ(mapper->iPutCount[CtranMapperBackend::IB], 0);
    ASSERT_EQ(mapper->iPutCount[CtranMapperBackend::NVL], 0);

    // issue puts to all ranks, they should use NVL for intra-node and IB for
    // inter-node
    issuePutsAndWaitForCompletion(
        buf, sendHdl, ranks, remoteBufs, remoteAccessKeys, nullptr, lowlatency);
  }

  ASSERT_EQ(
      mapper->iPutCount[CtranMapperBackend::IB],
      (nRanks - nLocalRanks) * putTimes);
  ASSERT_EQ(
      mapper->iPutCount[CtranMapperBackend::NVL], (nLocalRanks - 1) * putTimes);

  // Ensure remote rank have finished iput access before local deregister
  ASSERT_EQ(mapper->barrier(), commSuccess);

  COMMCHECK_TEST(comm_->ctran_->commDeregister(handle));
  NCCLCHECK_TEST(ncclMemFree(buf));
}

TEST_P(CtranDistMapperBackendPerfConfigTestParam, InteNodeUseIbFastPut) {
  const bool lowlatency = GetParam();
  const auto& statex = comm_->statex_.get();
  const int nRanks = statex->nRanks();
  const int myRank = statex->rank();
  std::vector<int> ranks(nRanks);

  std::vector<void*> remoteBufs(statex->nRanks(), nullptr);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(
      statex->nRanks());

  // get ranks
  for (int i = 0; i < nRanks; i++) {
    ranks[i] = i;
  }

  {
    CtranMapperEpochRAII epochRAII(mapper);

    if (lowlatency) {
      PreConnectAllPeers();
    }

    // exchange control messages with all ranks
    ASSERT_EQ(
        mapper->allGatherCtrl(
            buf, sendHdl, ranks, remoteBufs, remoteAccessKeys),
        commSuccess);

    intraNodeBarrier(commDeprecated_);

    ASSERT_EQ(mapper->iPutCount[CtranMapperBackend::IB], 0);
    // Remove local ranks from the list of peers
    for (auto it = ranks.begin(); it != ranks.end();) {
      if (statex->isSameNode(*it, myRank)) {
        it = ranks.erase(it);
      } else {
        it++;
      }
    }

    std::vector<std::unique_ptr<CtranMapperNotify>> notifies;
    for (auto peer : ranks) {
      auto notify = std::make_unique<CtranMapperNotify>();
      ASSERT_EQ(
          mapper->initNotify(
              peer,
              sendHdl,
              notify.get(),
              putTimes // all fast puts will notify the peer
              ),
          commSuccess);
      notifies.push_back(std::move(notify));
    }

    // issue puts to all remote ranks, they should use IB for inter-node
    issuePutsAndWaitForCompletion(
        buf,
        sendHdl,
        ranks,
        remoteBufs,
        remoteAccessKeys,
        nullptr,
        lowlatency,
        /*ibFastPath=*/true);

    if (lowlatency) {
      for (auto& notify : notifies) {
        ASSERT_EQ(
            mapper->waitNotify<LowLatencyCollConfig>(notify.get()),
            commSuccess);
      }
    } else {
      for (auto& notify : notifies) {
        ASSERT_EQ(
            mapper->waitNotify<DefaultPerfCollConfig>(notify.get()),
            commSuccess);
      }
    }
  }

  ASSERT_EQ(mapper->iPutCount[CtranMapperBackend::IB], ranks.size() * putTimes);

  // Ensure remote rank have finished iput access before local deregister
  ASSERT_EQ(mapper->barrier(), commSuccess);

  COMMCHECK_TEST(comm_->ctran_->commDeregister(handle));
  NCCLCHECK_TEST(ncclMemFree(buf));
}

// Tests for PerfConfig
INSTANTIATE_TEST_SUITE_P(
    CtranDistMapperBackendTest,
    CtranDistMapperBackendPerfConfigTestParam,
    ::testing::Values(true, false),
    [&](const testing::TestParamInfo<
        CtranDistMapperBackendPerfConfigTestParam::ParamType>& info) {
      if (info.param) {
        return "low_latency_perfconfig";
      } else {
        return "default_perfconfig";
      }
    });

/*
 * Send a signal to all local ranks using the IB backend.
 */
TEST_F(CtranDistMapperBackendTest, InterNodeIbSignal) {
  if (!mapper->hasBackend(
          comm_->statex_.get()->rank(), CtranMapperBackend::IB)) {
    GTEST_SKIP() << "IB backend is not available.  Skip test.";
  }
  const auto& statex = comm_->statex_.get();
  std::vector<int> ranks(statex->nRanks());
  std::vector<void*> remoteBufs(statex->nRanks(), nullptr);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(
      statex->nRanks());
  for (int i = 0; i < statex->nRanks(); i++) {
    ranks[i] = i;
  }

  {
    CtranMapperEpochRAII epochRAII(mapper);
    // Exchange control messages with every rank using IB
    ASSERT_EQ(
        mapper->allGatherCtrl(
            buf,
            sendHdl,
            ranks,
            remoteBufs,
            remoteAccessKeys,
            CtranMapperBackend::IB),
        commSuccess);

    ASSERT_EQ(mapper->barrier(), commSuccess);
    // Issue signals to its neighbor
    const auto myRank = comm_->statex_.get()->rank();
    const int peerRank = (myRank + 1) % statex->nRanks();
    uint64_t cmpVal = (myRank + statex->nRanks() - 1) % statex->nRanks();
    CtranMapperRequest req;
    ASSERT_EQ(
        mapper->atomicSet(
            remoteBufs[peerRank],
            (uint64_t)myRank,
            peerRank,
            CtranMapperConfig{.remoteAccessKey_ = remoteAccessKeys[peerRank]},
            &req),
        commSuccess);
    ASSERT_EQ(mapper->waitRequest(&req), commSuccess);
    // check values
    waitSignal(buf, cmpVal);
  }

  COMMCHECK_TEST(comm_->ctran_->commDeregister(handle));
  NCCLCHECK_TEST(ncclMemFree(buf));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
