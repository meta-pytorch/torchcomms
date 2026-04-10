// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include "comms/ctran/Ctran.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranNcclTestUtils.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/testinfra/TestUtils.h"
#include "nccl.h"

class CtranDistMapperTest : public ctran::CtranDistTestFixture {
 public:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);

#ifdef CTRAN_TEST_SOCKET_ONLY_BACKEND
    setenv("NCCL_CTRAN_BACKENDS", "socket, nvl", 1);
#endif
    ctran::CtranDistTestFixture::SetUp();

    // Turn on CTran for the entire test
    NCCL_CTRAN_ENABLE = true;

    comm_ = makeCtranComm();

    if (!ctranInitialized(comm_.get()) ||
        !comm_->ctran_->mapper->hasBackend()) {
      GTEST_SKIP()
          << "Ctran is not initialized or backend is not available.  Skip test.";
    }
  }

  void TearDown() override {
    comm_.reset();
    ctran::CtranDistTestFixture::TearDown();
  }

  void PreConnectAllPeers() {
    std::unordered_set<int> peers;
    for (int peer = 0; peer < comm_->statex_->nRanks(); peer++) {
      if (peer != comm_->statex_->rank()) {
        peers.insert(peer);
      }
    }
    ASSERT_EQ(comm_->ctran_->mapper->preConnect(peers), commSuccess);
  }

 protected:
  std::unique_ptr<CtranComm> comm_;
};

class CtranDistMapperBackendParam
    : public CtranDistMapperTest,
      public ::testing::WithParamInterface<std::tuple<CtranMapperBackend>> {};

TEST_P(CtranDistMapperBackendParam, intraAllGatherCtrl) {
  auto& [backend] = GetParam();
#ifdef CTRAN_TEST_SOCKET_ONLY_BACKEND
  if (backend == CtranMapperBackend::IB) {
    GTEST_SKIP() << "IB backend is not available.  Skip test.";
  }
#endif

  void* buf = nullptr;
  void* handle = nullptr;
  constexpr size_t bufSize = 8192;
  auto mapper = comm_->ctran_->mapper.get();
  const auto& statex = comm_->statex_.get();

  NCCLCHECK_TEST(ncclMemAlloc(&buf, bufSize));
  COMMCHECK_TEST(comm_->ctran_->commRegister(buf, bufSize, &handle));

  void* sendHdl = nullptr;
  bool localReg = false;
  COMMCHECK_TEST(mapper->searchRegHandle(buf, bufSize, &sendHdl, &localReg));
  ASSERT_NE(sendHdl, nullptr);
  ASSERT_EQ(localReg, false);

  // Reserve space for all ranks in comm, but exchange only within the node
  std::vector<void*> remoteBufs(statex->nRanks(), nullptr);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(
      statex->nRanks());
  for (auto& key : remoteAccessKeys) {
    key.backend = CtranMapperBackend::UNSET;
  }

  {
    CtranMapperEpochRAII epochRAII(mapper);
    ASSERT_EQ(
        mapper->intraAllGatherCtrl(
            buf, sendHdl, remoteBufs, remoteAccessKeys, backend),
        commSuccess);
  }

  const int nodeId = statex->node();
  for (int i = 0; i < statex->nRanks(); i++) {
    // Check self rank or other ranks on the same node have received value
    if (i == statex->rank()) {
      ASSERT_EQ(remoteBufs[i], buf);
      ASSERT_EQ(remoteAccessKeys[i].backend, CtranMapperBackend::UNSET);
    } else if (statex->node(i) == nodeId) {
      ASSERT_NE(remoteBufs[i], nullptr);
      if (backend == CtranMapperBackend::UNSET) {
        // If backend is not specified, the default backend of the peer rank
        // should be used
        ASSERT_NE(remoteAccessKeys[i].backend, CtranMapperBackend::UNSET);
      } else {
        ASSERT_EQ(remoteAccessKeys[i].backend, backend);
      }
    } else {
      // For ranks on remote node, they should not be set
      ASSERT_EQ(remoteBufs[i], nullptr);
      ASSERT_EQ(remoteAccessKeys[i].backend, CtranMapperBackend::UNSET);
    }
  }

  // Ensure all local ranks have finished importing remote NVL buffer before
  // deregister
  barrierNvlDomain(comm_.get());

  COMMCHECK_TEST(comm_->ctran_->commDeregister(handle));
  NCCLCHECK_TEST(ncclMemFree(buf));
}

TEST_P(CtranDistMapperBackendParam, allGatherCtrl) {
  auto& [backend] = GetParam();
#ifdef CTRAN_TEST_SOCKET_ONLY_BACKEND
  if (backend == CtranMapperBackend::IB) {
    GTEST_SKIP() << "IB backend is not available.  Skip test.";
  }
#endif

  void* buf = nullptr;
  void* handle = nullptr;
  constexpr size_t bufSize = 8192;
  auto mapper = comm_->ctran_->mapper.get();
  const auto& statex = comm_->statex_.get();

  if (statex->nLocalRanks() < 2) {
    GTEST_SKIP()
        << "Test requires at least 2 localRanks on each node.  Skip test.";
  }

  NCCLCHECK_TEST(ncclMemAlloc(&buf, bufSize));
  COMMCHECK_TEST(comm_->ctran_->commRegister(buf, bufSize, &handle));

  void* sendHdl = nullptr;
  bool localReg = false;
  COMMCHECK_TEST(mapper->searchRegHandle(buf, bufSize, &sendHdl, &localReg));
  ASSERT_NE(sendHdl, nullptr);
  ASSERT_EQ(localReg, false);

  // Reserve space for all ranks in comm, but exchange only the first half from
  // each node
  std::vector<void*> remoteBufs(statex->nRanks(), nullptr);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(
      statex->nRanks());

  // Exchange with the first half of the ranks on each node
  std::vector<int> ranks;
  for (int n = 0; n < statex->nNodes(); n++) {
    for (int r = 0; r < statex->nLocalRanks(n) / 2; r++) {
      int peer = statex->localRankToRank(r, n);
      ranks.push_back(peer);
    }
  }

  {
    CtranMapperEpochRAII epochRAII(mapper);

    auto res = mapper->allGatherCtrl(
        buf, sendHdl, ranks, remoteBufs, remoteAccessKeys, backend);
    ASSERT_EQ(res, commSuccess);
  }

  bool excluded =
      std::find(ranks.begin(), ranks.end(), statex->rank()) == ranks.end();

  for (int peer = 0; peer < statex->nRanks(); peer++) {
    bool peerExcluded =
        std::find(ranks.begin(), ranks.end(), peer) == ranks.end();
    if (excluded || peerExcluded) {
      // If either local rank or the peer is excluded, values should not be set
      ASSERT_EQ(remoteBufs[peer], nullptr);
      ASSERT_EQ(remoteAccessKeys[peer].backend, CtranMapperBackend::UNSET);
    } else if (peer == statex->rank()) {
      // Check self rank or others in the ranks list have received value
      ASSERT_EQ(remoteBufs[peer], buf);
      ASSERT_EQ(remoteAccessKeys[peer].backend, CtranMapperBackend::UNSET);
    } else {
      ASSERT_NE(remoteBufs[peer], nullptr);
      if (backend == CtranMapperBackend::UNSET) {
        // If backend is not specified, the default backend of the peer rank
        // should be used
        ASSERT_NE(remoteAccessKeys[peer].backend, CtranMapperBackend::UNSET);
      } else {
        ASSERT_EQ(remoteAccessKeys[peer].backend, backend);
      }
    }
  }

  // Ensure all local ranks have finished importing remote NVL buffer before
  // deregister
  barrierNvlDomain(comm_.get());

  COMMCHECK_TEST(comm_->ctran_->commDeregister(handle));
  NCCLCHECK_TEST(ncclMemFree(buf));
}

TEST_F(CtranDistMapperTest, allGatherCtrlNRanks) {
  void* buf = nullptr;
  void* handle = nullptr;
  constexpr size_t bufSize = 8192;
  auto mapper = comm_->ctran_->mapper.get();
  const auto& statex = comm_->statex_.get();

  if (statex->nLocalRanks() < 2) {
    GTEST_SKIP()
        << "Test requires at least 2 localRanks on each node.  Skip test.";
  }

  NCCLCHECK_TEST(ncclMemAlloc(&buf, bufSize));
  COMMCHECK_TEST(comm_->ctran_->commRegister(buf, bufSize, &handle));

  void* sendHdl = nullptr;
  bool localReg = false;
  COMMCHECK_TEST(mapper->searchRegHandle(buf, bufSize, &sendHdl, &localReg));
  ASSERT_NE(sendHdl, nullptr);
  ASSERT_EQ(localReg, false);

  std::vector<void*> remoteBufs(statex->nRanks(), nullptr);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(
      statex->nRanks());

  {
    CtranMapperEpochRAII epochRAII(mapper);

    auto res =
        mapper->allGatherCtrl(buf, sendHdl, remoteBufs, remoteAccessKeys);
    ASSERT_EQ(res, commSuccess);
  }

  for (int peer = 0; peer < statex->nRanks(); peer++) {
    if (peer == statex->rank()) {
      // Check self rank or others in the ranks list have received value
      ASSERT_EQ(remoteBufs[peer], buf);
      ASSERT_EQ(remoteAccessKeys[peer].backend, CtranMapperBackend::UNSET);
    } else {
      ASSERT_NE(remoteBufs[peer], nullptr);
      // Backend chosen isendCtrl also considers buffer attribute, which
      // may be different from the default backend of the peer rank.
      ASSERT_NE(remoteAccessKeys[peer].backend, CtranMapperBackend::UNSET);
    }
  }

  // Ensure all local ranks have finished importing remote NVL buffer before
  // deregister
  barrierNvlDomain(comm_.get());

  COMMCHECK_TEST(comm_->ctran_->commDeregister(handle));
  NCCLCHECK_TEST(ncclMemFree(buf));
}

class CtranDistMapperPerfConfigTestParam
    : public CtranDistMapperTest,
      public ::testing::WithParamInterface<bool> {};
TEST_P(CtranDistMapperPerfConfigTestParam, CtrlWithUserAllocatedReq) {
  const bool lowlatency = GetParam();
  void* buf = nullptr;
  void* handle = nullptr;
  constexpr size_t bufSize = 8192;
  auto mapper = comm_->ctran_->mapper.get();
  const auto& statex = comm_->statex_.get();

  NCCLCHECK_TEST(ncclMemAlloc(&buf, bufSize));
  COMMCHECK_TEST(comm_->ctran_->commRegister(buf, bufSize, &handle));

  if (lowlatency) {
    PreConnectAllPeers();
  }

  void* sendHdl = nullptr;
  bool localReg = false;
  COMMCHECK_TEST(mapper->searchRegHandle(buf, bufSize, &sendHdl, &localReg));
  ASSERT_NE(sendHdl, nullptr);
  ASSERT_EQ(localReg, false);

  std::vector<void*> remoteBufs(statex->nRanks(), nullptr);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(
      statex->nRanks());

  {
    CtranMapperEpochRAII epochRAII(mapper);
    std::vector<CtranMapperRequest> requests((statex->nRanks() - 1) * 2);
    int idx = 0;
    for (int peer = 0; peer < statex->nRanks(); peer++) {
      if (peer == statex->rank()) {
        remoteBufs[peer] = const_cast<void*>(buf);
        remoteAccessKeys[peer].backend = CtranMapperBackend::UNSET;
        continue;
      }
      if (lowlatency) {
        ASSERT_EQ(
            mapper->irecvCtrl<LowLatencyCollConfig>(
                &remoteBufs[peer],
                &remoteAccessKeys[peer],
                peer,
                &requests[idx++]),
            commSuccess);
        ASSERT_EQ(
            mapper->isendCtrl<LowLatencyCollConfig>(
                buf, sendHdl, peer, &requests[idx++]),
            commSuccess);
      } else {
        ASSERT_EQ(
            mapper->irecvCtrl<DefaultPerfCollConfig>(
                &remoteBufs[peer],
                &remoteAccessKeys[peer],
                peer,
                &requests[idx++]),
            commSuccess);
        ASSERT_EQ(
            mapper->isendCtrl<DefaultPerfCollConfig>(
                buf, sendHdl, peer, &requests[idx++]),
            commSuccess);
      }
    }
    if (lowlatency) {
      ASSERT_EQ(
          mapper->waitAllRequests<LowLatencyCollConfig>(requests), commSuccess);
    } else {
      ASSERT_EQ(
          mapper->waitAllRequests<DefaultPerfCollConfig>(requests),
          commSuccess);
    }
  }

  for (int peer = 0; peer < statex->nRanks(); peer++) {
    if (peer == statex->rank()) {
      // Check self rank or others in the ranks list have received value
      ASSERT_EQ(remoteBufs[peer], buf);
      ASSERT_EQ(remoteAccessKeys[peer].backend, CtranMapperBackend::UNSET);
    } else {
      ASSERT_NE(remoteBufs[peer], nullptr);
      // Backend chosen isendCtrl also considers buffer attribute, which
      // may be different from the default backend of the peer rank.
      ASSERT_NE(remoteAccessKeys[peer].backend, CtranMapperBackend::UNSET);
    }
  }

  // Ensure all local ranks have finished importing remote NVL buffer before
  // deregister
  barrierNvlDomain(comm_.get());

  COMMCHECK_TEST(comm_->ctran_->commDeregister(handle));
  NCCLCHECK_TEST(ncclMemFree(buf));
}

TEST_P(CtranDistMapperPerfConfigTestParam, isendCtrlBatchToAllPeers) {
  const bool lowlatency = GetParam();
  void* buf = nullptr;
  void* handle = nullptr;
  constexpr size_t bufSize = 8192;
  auto mapper = comm_->ctran_->mapper.get();
  const auto& statex = comm_->statex_.get();

  NCCLCHECK_TEST(ncclMemAlloc(&buf, bufSize));
  COMMCHECK_TEST(comm_->ctran_->commRegister(buf, bufSize, &handle));

  if (lowlatency) {
    PreConnectAllPeers();
  }

  void* sendHdl = nullptr;
  bool localReg = false;
  COMMCHECK_TEST(mapper->searchRegHandle(buf, bufSize, &sendHdl, &localReg));
  ASSERT_NE(sendHdl, nullptr);
  ASSERT_EQ(localReg, false);

  std::vector<void*> localBufs(statex->nRanks(), buf),
      remoteBufs(statex->nRanks(), nullptr);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(
      statex->nRanks());
  std::vector<int> ibPeers, otherPeers;
  for (int peer = 0; peer < statex->nRanks(); peer++) {
    if (!statex->isSameNode(statex->rank(), peer) &&
        mapper->ctranIbPtr() != nullptr) {
      ibPeers.push_back(peer);
    } else {
      if (peer != statex->rank()) {
        otherPeers.push_back(peer);
      }
    }
  }

  {
    CtranMapperEpochRAII epochRAII(mapper);
    std::vector<CtranMapperRequest> ibSendCtrlRequests(ibPeers.size()),
        otherSendCtrlRequests(otherPeers.size()),
        recvCtrlRequests(statex->nRanks() - 1);
    int idx = 0;
    for (int peer = 0; peer < statex->nRanks(); peer++) {
      if (peer == statex->rank()) {
        continue;
      }
      if (lowlatency) {
        ASSERT_EQ(
            mapper->irecvCtrl<LowLatencyCollConfig>(
                &remoteBufs[peer],
                &remoteAccessKeys[peer],
                peer,
                &recvCtrlRequests[idx++]),
            commSuccess);
      } else {
        ASSERT_EQ(
            mapper->irecvCtrl<DefaultPerfCollConfig>(
                &remoteBufs[peer],
                &remoteAccessKeys[peer],
                peer,
                &recvCtrlRequests[idx++]),
            commSuccess);
      }
    }

    if (lowlatency) {
      ASSERT_EQ(
          mapper->isendCtrlBatch<LowLatencyCollConfig>(
              localBufs,
              sendHdl,
              ibPeers,
              ibSendCtrlRequests,
              CtranMapperBackend::IB),
          commSuccess);
      ASSERT_EQ(
          mapper->isendCtrlBatch<LowLatencyCollConfig>(
              localBufs,
              sendHdl,
              otherPeers,
              otherSendCtrlRequests,
              CtranMapperBackend::UNSET),
          commSuccess);

      ASSERT_EQ(
          mapper->waitAllRequests<LowLatencyCollConfig>(ibSendCtrlRequests),
          commSuccess);
      ASSERT_EQ(
          mapper->waitAllRequests<LowLatencyCollConfig>(otherSendCtrlRequests),
          commSuccess);
      ASSERT_EQ(
          mapper->waitAllRequests<LowLatencyCollConfig>(recvCtrlRequests),
          commSuccess);
    } else {
      ASSERT_EQ(
          mapper->isendCtrlBatch<DefaultPerfCollConfig>(
              localBufs,
              sendHdl,
              ibPeers,
              ibSendCtrlRequests,
              CtranMapperBackend::IB),
          commSuccess);
      ASSERT_EQ(
          mapper->isendCtrlBatch<DefaultPerfCollConfig>(
              localBufs,
              sendHdl,
              otherPeers,
              otherSendCtrlRequests,
              CtranMapperBackend::UNSET),
          commSuccess);

      ASSERT_EQ(
          mapper->waitAllRequests<DefaultPerfCollConfig>(ibSendCtrlRequests),
          commSuccess);
      ASSERT_EQ(
          mapper->waitAllRequests<DefaultPerfCollConfig>(otherSendCtrlRequests),
          commSuccess);
      ASSERT_EQ(
          mapper->waitAllRequests<DefaultPerfCollConfig>(recvCtrlRequests),
          commSuccess);
    }
  }

  for (int peer : ibPeers) {
    ASSERT_NE(remoteBufs[peer], nullptr);
    ASSERT_EQ(remoteAccessKeys[peer].backend, CtranMapperBackend::IB);
  }
  for (int peer : otherPeers) {
    ASSERT_NE(remoteBufs[peer], nullptr);
    ASSERT_NE(remoteAccessKeys[peer].backend, CtranMapperBackend::UNSET);
  }

  // Ensure all local ranks have finished importing remote NVL buffer before
  // deregister
  barrierNvlDomain(comm_.get());

  COMMCHECK_TEST(comm_->ctran_->commDeregister(handle));
  NCCLCHECK_TEST(ncclMemFree(buf));
}

TEST_F(CtranDistMapperTest, SyncCtrl) {
  auto mapper = comm_->ctran_->mapper.get();
  const auto& statex = comm_->statex_.get();

  const int rank = statex->rank();
  const int nRanks = statex->nRanks();
  int sendPeer = (rank + 1) % nRanks;
  int recvPeer = (rank - 1 + nRanks) % nRanks;

  CtranMapperEpochRAII epochRAII(mapper);

  std::vector<std::unique_ptr<CtranMapperRequest>> requests;
  for (int x = 0; x < 10; x++) {
    CtranMapperRequest* req = nullptr;
    COMMCHECK_TEST(mapper->isendCtrl(sendPeer, &req));
    requests.push_back(std::unique_ptr<CtranMapperRequest>(req));

    COMMCHECK_TEST(mapper->irecvCtrl(recvPeer, &req));
    requests.push_back(std::unique_ptr<CtranMapperRequest>(req));
  }

  while (!requests.empty()) {
    COMMCHECK_TEST(mapper->testSomeRequests(requests));
  }
}

TEST_F(CtranDistMapperTest, SyncCtrlWithReq) {
  auto mapper = comm_->ctran_->mapper.get();
  const auto& statex = comm_->statex_.get();

  const int rank = statex->rank();
  const int nRanks = statex->nRanks();
  int sendPeer = (rank + 1) % nRanks;
  int recvPeer = (rank - 1 + nRanks) % nRanks;

  CtranMapperEpochRAII epochRAII(mapper);

  constexpr auto nIters = 10;
  std::vector<CtranMapperRequest> reqs(nIters * 2);
  for (int x = 0; x < nIters; x++) {
    COMMCHECK_TEST(mapper->isendCtrl(sendPeer, &reqs.at(x * 2)));
    COMMCHECK_TEST(mapper->irecvCtrl(recvPeer, &reqs.at(x * 2 + 1)));
  }

  for (auto& req : reqs) {
    COMMCHECK_TEST(mapper->waitRequest(&req));
  }
}

TEST_F(CtranDistMapperTest, intraBarrier) {
  auto mapper = comm_->ctran_->mapper.get();
  CtranMapperEpochRAII epochRAII(mapper);

  constexpr int nIters = 10;
  // Test barrier multiple times and ensure no deadlock or error
  for (int x = 0; x < nIters; x++) {
    ASSERT_EQ(mapper->intraBarrier(), commSuccess);
  }
}

TEST_F(CtranDistMapperTest, intraBarrierWithCtrl) {
  auto mapper = comm_->ctran_->mapper.get();
  const auto& statex = comm_->statex_.get();
  void* buf = nullptr;
  void* handle = nullptr;
  constexpr size_t bufSize = 8192;
  void* sendHdl = nullptr;
  bool localReg = false;

  // exchange buffer with the next local rank
  const auto nLocalRanks = statex->nLocalRanks();
  const auto myLocalRank = statex->localRank();
  const auto sendPeerRank =
      statex->localRankToRank((myLocalRank + 1) % nLocalRanks);
  const auto recvPeerRank =
      statex->localRankToRank((myLocalRank - 1 + nLocalRanks) % nLocalRanks);

  constexpr int nIters = 10;
  std::array<std::unique_ptr<CtranMapperRequest>, 2> requests;

  // Test barrier followed with regular ctrl msg multiple times, to ensure no
  // deadlock or mismatch
  for (int x = 0; x < nIters; x++) {
    void* remoteBufs = nullptr;
    struct CtranMapperRemoteAccessKey remoteAccessKey;

    NCCLCHECK_TEST(ncclMemAlloc(&buf, bufSize));
    COMMCHECK_TEST(comm_->ctran_->commRegister(buf, bufSize, &handle));
    COMMCHECK_TEST(mapper->searchRegHandle(buf, bufSize, &sendHdl, &localReg));
    ASSERT_NE(sendHdl, nullptr);
    ASSERT_EQ(localReg, false);

    {
      CtranMapperEpochRAII epochRAII(mapper);
      CtranMapperRequest* req = nullptr;
      ASSERT_EQ(
          mapper->isendCtrl(buf, sendHdl, sendPeerRank, &req), commSuccess);
      requests.at(0) = std::unique_ptr<CtranMapperRequest>(req);

      ASSERT_EQ(
          mapper->irecvCtrl(&remoteBufs, &remoteAccessKey, recvPeerRank, &req),
          commSuccess);
      requests.at(1) = std::unique_ptr<CtranMapperRequest>(req);

      ASSERT_EQ(mapper->intraBarrier(), commSuccess);

      // Blocking barrier must ensure all preceding send/recv ctrls have
      // finished, thus test once only
      for (auto& req_ : requests) {
        bool complete = false;
        ASSERT_EQ(mapper->testRequest(req_.get(), &complete), commSuccess);
        ASSERT_TRUE(complete);
      }

      // We must barrier again after ranks completed the request, since NVL
      // buffer importing happens at request completion.
      ASSERT_EQ(mapper->intraBarrier(), commSuccess);
    }

    // Immediately free buffer to check receiver side has finished importing
    // buffer, otherwise it may fail if import after local free.
    COMMCHECK_TEST(comm_->ctran_->commDeregister(handle));
    NCCLCHECK_TEST(ncclMemFree(buf));
  }

  // Add barrier to ensure that ranks doesn't exit before all local ranks have
  // finished the deregistration API which is asynchronous by design.
  ASSERT_EQ(mapper->intraBarrier(), commSuccess);
}

TEST_F(CtranDistMapperTest, barrier) {
  auto mapper = comm_->ctran_->mapper.get();
  CtranMapperEpochRAII epochRAII(mapper);

  constexpr int nIters = 10;
  // Test barrier multiple times and ensure no deadlock or error
  for (int x = 0; x < nIters; x++) {
    ASSERT_EQ(mapper->barrier(), commSuccess);
  }
}

TEST_F(CtranDistMapperTest, intraNodeDynamicRegistration) {
  void* buf = nullptr;
  constexpr size_t bufSize = 8192;
  auto mapper = comm_->ctran_->mapper.get();
  const auto& statex = comm_->statex_.get();

  if (statex->nLocalRanks() < 2) {
    GTEST_SKIP()
        << "Test requires at least 2 localRanks on each node.  Skip test.";
  }

  COMMCHECK_TEST(
      ctran::utils::commCudaMalloc(
          reinterpret_cast<char**>(&buf),
          bufSize,
          nullptr,
          "CtranDistMapperTest.intraNodeDynamicRegistration"));

  void* sendHdl = nullptr;
  bool isDynamicReg = false;
  // trigger dynamic registration: searchRegHandle has a side effect to perform
  // dynamic registration is a buffer is not pre-registered
  COMMCHECK_TEST(
      mapper->searchRegHandle(buf, bufSize, &sendHdl, &isDynamicReg));
  ASSERT_NE(sendHdl, nullptr);
  ASSERT_EQ(isDynamicReg, true);

  std::vector<void*> remoteBufs(statex->nRanks(), nullptr);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(
      statex->nRanks());

  {
    CtranMapperEpochRAII epochRAII(mapper);

    auto res =
        mapper->allGatherCtrl(buf, sendHdl, remoteBufs, remoteAccessKeys);
    ASSERT_EQ(res, commSuccess);
  }

  for (int peer = 0; peer < statex->nRanks(); peer++) {
    if (peer == statex->rank()) {
      // Check self rank or others in the ranks list have received value
      ASSERT_EQ(remoteBufs[peer], buf);
      ASSERT_EQ(remoteAccessKeys[peer].backend, CtranMapperBackend::UNSET);
    } else {
      ASSERT_NE(remoteBufs[peer], nullptr);
      // Backend chosen isendCtrl also considers buffer attribute, which
      // may be different from the default backend of the peer rank.
      ASSERT_NE(remoteAccessKeys[peer].backend, CtranMapperBackend::UNSET);
    }
  }

  // Ensure all local ranks have finished importing remote NVL buffer before
  // deregister
  barrierNvlDomain(comm_.get());

  COMMCHECK_TEST(comm_->ctran_->mapper->deregDynamic(sendHdl));

  COMMCHECK_TEST(ctran::utils::commCudaFree(buf));
}

TEST_F(CtranDistMapperTest, CtrlMsgWithRawPayload) {
  auto mapper = comm_->ctran_->mapper.get();
  const auto& statex = comm_->statex_.get();

  if (statex->nRanks() < 2) {
    GTEST_SKIP() << "Test requires at least 2 ranks. Skip test.";
  }

  PreConnectAllPeers();

  std::vector<int> ibPeers;
  for (int peer = 0; peer < statex->nRanks(); peer++) {
    if (!statex->isSameNode(statex->rank(), peer) &&
        mapper->ctranIbPtr() != nullptr) {
      ibPeers.push_back(peer);
    }
  }

  if (ibPeers.empty()) {
    GTEST_SKIP() << "No IB peers available. Skip test.";
  }

  const int rank = statex->rank();

  // Create test payload
  int sendMsg = 111000 + rank;
  std::vector<int> recvMsgs(ibPeers.size(), 0);
  std::vector<CtranMapperRequest> sendReqs(ibPeers.size()),
      recvReqs(ibPeers.size());
  {
    CtranMapperEpochRAII epochRAII(mapper);

    // Send messages to all IB peers and receive from them
    for (size_t i = 0; i < ibPeers.size(); i++) {
      int peer = ibPeers[i];

      COMMCHECK_TEST(
          mapper->isendCtrlMsg(&sendMsg, sizeof(int), peer, &sendReqs[i]));
      COMMCHECK_TEST(
          mapper->irecvCtrlMsg(&recvMsgs[i], sizeof(int), peer, &recvReqs[i]));
    }

    for (auto& req : recvReqs) {
      mapper->waitRequest(&req);
    }
    for (auto& req : sendReqs) {
      mapper->waitRequest(&req);
    }

    // Verify received messages
    for (size_t i = 0; i < ibPeers.size(); i++) {
      int expectedMsg = 111000 + ibPeers[i];
      ASSERT_EQ(recvMsgs[i], expectedMsg);
    }
  }
}

class CtranDistMapperBufExportParam : public CtranDistMapperTest,
                                      public ::testing::WithParamInterface<
                                          // managed, skipRemRelease
                                          std::tuple<bool, bool>> {};

TEST_P(CtranDistMapperBufExportParam, BufExportCtrl) {
  auto [managed, skipRemRelease] = GetParam();

  auto mapper = comm_->ctran_->mapper.get();
  const auto& statex = comm_->statex_.get();

  const int rank = statex->rank();
  const int nRanks = statex->nRanks();
  int sendPeer = (rank + 1) % nRanks;
  int recvPeer = (rank - 1 + nRanks) % nRanks;

  void* buf = nullptr;
  void* segHandle = nullptr;
  constexpr size_t bufSize = 8192;

  NCCLCHECK_TEST(ncclMemAlloc(&buf, bufSize));
  COMMCHECK_TEST(mapper->regMem(
      buf, bufSize, &segHandle, false /* forceRegist */, managed));

  void* sendHdl = nullptr;
  bool localReg = false;
  COMMCHECK_TEST(mapper->searchRegHandle(buf, bufSize, &sendHdl, &localReg));
  ASSERT_NE(sendHdl, nullptr);
  ASSERT_EQ(localReg, false);
  void* remoteBufs = nullptr;
  auto remoteAccessKeys = CtranMapperRemoteAccessKey();

  std::vector<std::unique_ptr<CtranMapperRequest>> requests;
  for (int x = 0; x < 10; x++) {
    CtranMapperRequest* req = nullptr;

    CtranMapperEpochRAII epochRAII(mapper);

    COMMCHECK_TEST(mapper->isendCtrl(buf, sendHdl, sendPeer, &req));
    requests.push_back(std::unique_ptr<CtranMapperRequest>(req));

    COMMCHECK_TEST(
        mapper->irecvCtrl(&remoteBufs, &remoteAccessKeys, recvPeer, &req));
    requests.push_back(std::unique_ptr<CtranMapperRequest>(req));

    while (!requests.empty()) {
      COMMCHECK_TEST(mapper->testSomeRequests(requests));
    }

    ASSERT_NE(remoteBufs, nullptr);
    ASSERT_EQ(remoteAccessKeys.backend, mapper->getBackend(recvPeer));
    if (remoteAccessKeys.backend == CtranMapperBackend::IB) {
      ASSERT_NE(remoteAccessKeys.ibKey.rkeys[0], 0);
    } else if (remoteAccessKeys.backend == CtranMapperBackend::NVL) {
      ASSERT_EQ(remoteAccessKeys.nvlKey.peerId, statex->gPid(recvPeer));
      ASSERT_NE(remoteAccessKeys.nvlKey.basePtr, nullptr);
    }

    if (skipRemRelease) {
      ASSERT_EQ(mapper->deregRemReg(&remoteAccessKeys), commSuccess);
    }
  }

  // Finally sync if sendPeer side has finished all importing before local
  // rank frees the buffer
  {
    CtranMapperEpochRAII epochRAII(mapper);

    CtranMapperRequest *sReq = nullptr, *rReq = nullptr;
    COMMCHECK_TEST(mapper->irecvCtrl(sendPeer, &sReq));
    COMMCHECK_TEST(mapper->isendCtrl(recvPeer, &rReq));
    COMMCHECK_TEST(mapper->waitRequest(sReq));
    COMMCHECK_TEST(mapper->waitRequest(rReq));
  }

  COMMCHECK_TEST(mapper->deregMem(segHandle, skipRemRelease));
  if (!skipRemRelease) {
    auto ipcRegCache = ctran::IpcRegCache::getInstance();
    if (remoteAccessKeys.backend == CtranMapperBackend::NVL) {
      while (ipcRegCache->getNumRemReg(remoteAccessKeys.nvlKey.peerId) > 0) {
        std::this_thread::yield();
      }
      ASSERT_EQ(ipcRegCache->getNumRemReg(remoteAccessKeys.nvlKey.peerId), 0);
    }
  }
  NCCLCHECK_TEST(ncclMemFree(buf));
}

// Tests for PerfConfig

// Parameterized test class for allGatherCtrl with different segment counts.
// Tests both the inline path (numSegments <= CTRAN_IPC_INLINE_SEGMENTS)
// and the multi-packet path (numSegments > CTRAN_IPC_INLINE_SEGMENTS).
class CtranDistMapperMultiSegmentParam
    : public CtranDistMapperTest,
      public ::testing::WithParamInterface<int> {};

// E2E test: allGatherCtrl exercising a mix of IB (inter-node) and NVL
// (intra-node) backends. When numSegments > CTRAN_IPC_INLINE_SEGMENTS (2),
// extra segments are packed densely as raw CtranIpcSegDesc data (Phase 2).
// When numSegments <= 2, the inline path is used. By using the all-ranks
// allGatherCtrl (backend=UNSET), intra-node peers use NVL while
// inter-node peers use IB, validating the mixed-backend path end-to-end.
TEST_P(CtranDistMapperMultiSegmentParam, allGatherCtrlMultiSegment) {
  auto mapper = comm_->ctran_->mapper.get();
  const auto& statex = comm_->statex_.get();

  if (statex->nLocalRanks() < 2) {
    GTEST_SKIP()
        << "Test requires at least 2 localRanks on each node.  Skip test.";
  }

  if (!ncclIsCuMemSupported()) {
    GTEST_SKIP() << "CuMem not supported, skip multi-segment test";
  }

  const int numSegments = GetParam();
  constexpr size_t segSize = 1 * 1024; // 1KB
  size_t totalSize = segSize * numSegments;

  std::vector<TestMemSegment> segments;
  void* buf = ctran::CtranNcclTestHelpers::prepareBuf(
      totalSize, kCuMemAllocDisjoint, segments, numSegments);
  ASSERT_NE(buf, nullptr);

  // Fill local buffer with a rank-specific pattern so we can verify
  // remote imports. Each rank writes its rank value to every int element.
  const size_t count = totalSize / sizeof(int);
  std::vector<int> fillVals(count, statex->rank());
  CUDACHECK_TEST(
      cudaMemcpy(buf, fillVals.data(), totalSize, cudaMemcpyHostToDevice));

  void* handle = nullptr;
  COMMCHECK_TEST(comm_->ctran_->commRegister(buf, totalSize, &handle));

  void* sendHdl = nullptr;
  bool localReg = false;
  COMMCHECK_TEST(mapper->searchRegHandle(buf, totalSize, &sendHdl, &localReg));
  ASSERT_NE(sendHdl, nullptr);

  std::vector<void*> remoteBufs(statex->nRanks(), nullptr);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(
      statex->nRanks());
  for (auto& key : remoteAccessKeys) {
    key.backend = CtranMapperBackend::UNSET;
  }

  // Use the all-ranks allGatherCtrl (backend=UNSET) so that intra-node
  // peers are exchanged via NVL and inter-node peers via IB.
  {
    CtranMapperEpochRAII epochRAII(mapper);
    ASSERT_EQ(
        mapper->allGatherCtrl(buf, sendHdl, remoteBufs, remoteAccessKeys),
        commSuccess);
  }

  const int nodeId = statex->node();
  bool hasIbPeer = false;
  bool hasNvlPeer = false;

  for (int i = 0; i < statex->nRanks(); i++) {
    if (i == statex->rank()) {
      ASSERT_EQ(remoteBufs[i], buf);
      ASSERT_EQ(remoteAccessKeys[i].backend, CtranMapperBackend::UNSET);
    } else {
      ASSERT_NE(remoteBufs[i], nullptr);
      ASSERT_NE(remoteAccessKeys[i].backend, CtranMapperBackend::UNSET);

      if (statex->node(i) == nodeId) {
        // Intra-node peer: NVL backend expected for cuMem buffers
        ASSERT_EQ(remoteAccessKeys[i].backend, CtranMapperBackend::NVL)
            << "Intra-node peer " << i << " should use NVL backend";
        hasNvlPeer = true;
      } else {
        // Inter-node peer: IB backend expected
        ASSERT_EQ(remoteAccessKeys[i].backend, CtranMapperBackend::IB)
            << "Inter-node peer " << i << " should use IB backend";
        hasIbPeer = true;
      }
    }
  }

  // Verify imported remote memory contents for NVL (intra-node) peers only.
  // NVL peers have locally-mapped memory via CUDA IPC, so we can read back
  // directly. IB (inter-node) peers only provide a remote virtual address
  // for RDMA operations; it is not mapped into local address space.
  for (int i = 0; i < statex->nRanks(); i++) {
    if (i == statex->rank()) {
      continue;
    }
    if (remoteAccessKeys[i].backend != CtranMapperBackend::NVL) {
      continue;
    }
    std::vector<int> readBack(count);
    CUDACHECK_TEST(cudaMemcpy(
        readBack.data(), remoteBufs[i], totalSize, cudaMemcpyDeviceToHost));
    const std::vector<int> expected(count, i);
    ASSERT_EQ(readBack, expected)
        << "Remote buf from rank " << i << " (node " << statex->node(i)
        << ", backend " << static_cast<int>(remoteAccessKeys[i].backend)
        << ") has unexpected contents";
  }

  if (statex->nNodes() > 1) {
    ASSERT_TRUE(hasIbPeer)
        << "Multi-node test should have at least one IB peer";
    ASSERT_TRUE(hasNvlPeer)
        << "Multi-node test should have at least one NVL peer";
    LOG(INFO) << "Verified mixed IB+NVL backends across " << statex->nNodes()
              << " nodes";
  } else {
    ASSERT_TRUE(hasNvlPeer)
        << "Single-node test should have at least one NVL peer";
    LOG(INFO) << "Single-node run: only NVL backend exercised. "
              << "Run with multiple nodes to test mixed IB+NVL path.";
  }

  // Ensure all local ranks have finished importing remote NVL buffer before
  // deregister
  barrierNvlDomain(comm_.get());

  COMMCHECK_TEST(comm_->ctran_->commDeregister(handle));
  ctran::CtranNcclTestHelpers::releaseBuf(
      buf, totalSize, kCuMemAllocDisjoint, numSegments);
}

// Parameterize allGatherCtrlMultiSegment with different segment counts:
// - 2 segments: Tests the inline path (numSegments <=
// CTRAN_IPC_INLINE_SEGMENTS)
// - 100 segments: Tests the multi-packet path (numSegments > 2)
INSTANTIATE_TEST_SUITE_P(
    CtranDistMapperTest,
    CtranDistMapperMultiSegmentParam,
    ::testing::Values(2, 100),
    [](const testing::TestParamInfo<
        CtranDistMapperMultiSegmentParam::ParamType>& info) {
      return "numSegments_" + std::to_string(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    CtranDistMapperTest,
    CtranDistMapperPerfConfigTestParam,
    ::testing::Values(true, false),
    [&](const testing::TestParamInfo<
        CtranDistMapperPerfConfigTestParam::ParamType>& info) {
      if (info.param) {
        return "low_latency_perfconfig";
      } else {
        return "default_perfconfig";
      }
    });

// Tests for UInt64
INSTANTIATE_TEST_SUITE_P(
    CtranDistMapperTest,
    CtranDistMapperBufExportParam,
    ::testing::Combine(
        ::testing::Values(true, false), // managed
        ::testing::Values(true, false)), // skipRemRelease
    [&](const testing::TestParamInfo<CtranDistMapperBufExportParam::ParamType>&
            info) {
      std::string name;
      if (std::get<0>(info.param)) {
        name = "managed";
      } else {
        name = "unmanaged";
      }
      if (std::get<1>(info.param)) {
        name += "_skipRemRelease";
      } else {
        name += "_waitRemRelease";
      }
      return name;
    });

INSTANTIATE_TEST_SUITE_P(
    CtranDistMapperTest,
    CtranDistMapperBackendParam,
    ::testing::Values(CtranMapperBackend::IB, CtranMapperBackend::UNSET),
    [&](const testing::TestParamInfo<CtranDistMapperBackendParam::ParamType>&
            info) {
      auto backend = std::get<0>(info.param);
      if (backend == CtranMapperBackend::UNSET) {
        return "DEFAULT";
      } else {
        return "IB";
      }
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
