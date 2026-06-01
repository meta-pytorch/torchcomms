// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/interfaces/ICtran.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/transport/IP2pHostTransport.h"

class CtranMapperHostTransportDistTest : public ctran::CtranDistTestFixture {
 public:
  void SetUp() override {
    CtranDistTestFixture::SetUp();
    comm_ = makeCtranComm();
  }
  void TearDown() override {
    comm_.reset();
    CtranDistTestFixture::TearDown();
  }

 protected:
  std::unique_ptr<CtranComm> comm_{nullptr};
};

TEST_F(CtranMapperHostTransportDistTest, MapperHasGetP2pTransport) {
  ASSERT_NE(comm_, nullptr);
  ASSERT_NE(comm_->ctran_, nullptr);
  ASSERT_NE(comm_->ctran_->mapper, nullptr);
}

TEST_F(CtranMapperHostTransportDistTest, GetZcTransportCachesPerPeer) {
  ASSERT_NE(comm_->ctran_->mapper, nullptr);
  if (comm_->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "IB backend not available";
  }
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }

  // After the eager-connectVcs change (see HostZcTransport ctor),
  // getP2pTransport(peer, kZeroCopy) drives the per-peer rendezvous
  // synchronously. For deadlock-free progress every rank must call
  // it for every other rank — otherwise larger-rank-side spinners
  // hang waiting on a peer that never initiated. We do an
  // all-to-all warm-up here, then run the caching assertions on a
  // single peer.
  for (int p = 0; p < numRanks; ++p) {
    if (p == globalRank) {
      continue;
    }
    ASSERT_NE(
        comm_->ctran_->mapper->getP2pTransport(
            p, ctran::transport::HostTransportMode::kZeroCopy),
        nullptr);
  }

  const int peer = (globalRank + 1) % numRanks;
  auto* first = comm_->ctran_->mapper->getP2pTransport(
      peer, ctran::transport::HostTransportMode::kZeroCopy);
  auto* second = comm_->ctran_->mapper->getP2pTransport(
      peer, ctran::transport::HostTransportMode::kZeroCopy);

  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first, second);
  EXPECT_EQ(first->peerRank(), peer);
  EXPECT_EQ(first->mode(), ctran::transport::HostTransportMode::kZeroCopy);
  // Pure-ZC: no staging buffer is allocated by the transport ctor.
  EXPECT_EQ(first->pipelineDepth(), 0);
  EXPECT_EQ(first->chunkSize(), 0u);
}

TEST_F(CtranMapperHostTransportDistTest, GetCbTransportCachesPerPeer) {
  ASSERT_NE(comm_->ctran_->mapper, nullptr);
  if (comm_->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "IB backend not available";
  }
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }

  // HostCbTransport ctor (like HostZcTransport) eagerly calls
  // ctranIb->connectVcs(peer) for the per-peer rendezvous. For
  // deadlock-free progress every rank must initiate to every other
  // peer; otherwise the larger-rank-side spinner hangs waiting on a
  // peer that never initiated.
  for (int p = 0; p < numRanks; ++p) {
    if (p == globalRank) {
      continue;
    }
    ASSERT_NE(
        comm_->ctran_->mapper->getP2pTransport(
            p, ctran::transport::HostTransportMode::kCopyBased),
        nullptr);
  }

  const int peer = (globalRank + 1) % numRanks;
  auto* t = comm_->ctran_->mapper->getP2pTransport(
      peer, ctran::transport::HostTransportMode::kCopyBased);
  ASSERT_NE(t, nullptr);
  EXPECT_EQ(t->mode(), ctran::transport::HostTransportMode::kCopyBased);
  EXPECT_GT(t->pipelineDepth(), 0);
  EXPECT_GT(t->chunkSize(), 0u);
}

TEST_F(CtranMapperHostTransportDistTest, DifferentPeersGetDifferentTransports) {
  if (comm_->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "IB backend not available";
  }
  if (numRanks < 3) {
    GTEST_SKIP() << "Requires >= 3 ranks for two distinct non-self peers";
  }
  // All-to-all warm-up — eager connectVcs in the transport ctor
  // requires every rank to drive every pair, otherwise larger-rank
  // spinners deadlock.
  for (int p = 0; p < numRanks; ++p) {
    if (p == globalRank) {
      continue;
    }
    ASSERT_NE(
        comm_->ctran_->mapper->getP2pTransport(
            p, ctran::transport::HostTransportMode::kZeroCopy),
        nullptr);
  }

  const int peerA = (globalRank + 1) % numRanks;
  const int peerB = (globalRank + 2) % numRanks;
  ASSERT_NE(peerA, peerB);

  auto* tA = comm_->ctran_->mapper->getP2pTransport(
      peerA, ctran::transport::HostTransportMode::kZeroCopy);
  auto* tB = comm_->ctran_->mapper->getP2pTransport(
      peerB, ctran::transport::HostTransportMode::kZeroCopy);
  ASSERT_NE(tA, nullptr);
  ASSERT_NE(tB, nullptr);
  EXPECT_NE(tA, tB);
  EXPECT_EQ(tA->peerRank(), peerA);
  EXPECT_EQ(tB->peerRank(), peerB);
}

TEST_F(CtranMapperHostTransportDistTest, ModeMismatchAborts) {
  if (comm_->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "IB backend not available";
  }
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }

  // All-to-all warm-up to drive the eager connectVcs rendezvous
  // for every peer pair before the EXPECT_DEATH fork. Without this,
  // the asymmetric single-peer call below races against the
  // bootstrap channel and intermittently deadlocks one rank when
  // run in sequence with other tests (see
  // GetCbTransportCachesPerPeer for the same root cause).
  for (int p = 0; p < numRanks; ++p) {
    if (p == globalRank) {
      continue;
    }
    ASSERT_NE(
        comm_->ctran_->mapper->getP2pTransport(
            p, ctran::transport::HostTransportMode::kZeroCopy),
        nullptr);
  }

  const int peer = (globalRank + 1) % numRanks;
  auto* zc = comm_->ctran_->mapper->getP2pTransport(
      peer, ctran::transport::HostTransportMode::kZeroCopy);
  ASSERT_NE(zc, nullptr);

  // Asking for the same peer with a different mode must abort
  // (FB_CHECKABORT). gtest's death-test matcher inspects only the
  // child process's captured stderr; FB_CHECKABORT writes its message
  // through CLOGF (which goes to the logger sink, not raw stderr), so
  // the message itself isn't visible to EXPECT_DEATH. The signal
  // handler's "*** Aborted at ..." banner IS visible — match on that
  // so the test asserts only the death itself.
  EXPECT_DEATH_IF_SUPPORTED(
      {
        (void)comm_->ctran_->mapper->getP2pTransport(
            peer, ctran::transport::HostTransportMode::kCopyBased);
      },
      ".*Aborted.*");
}

TEST_F(CtranMapperHostTransportDistTest, CbCacheClearedBeforeGpeTeardown) {
  // Construct a CB transport for a peer (which pops GpeKernelSync
  // entries from gpe's pool). After the comm is destroyed,
  // numInUseGpeKernelSyncs MUST be zero — i.e., setAtDestruction()
  // cleared the per-peer cache before gpe was torn down so the CB
  // transport's dtor returned its borrowed syncs.
  if (comm_->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "IB backend not available";
  }
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }

  // HostCbTransport ctor eagerly rendezvous via ctranIb->connectVcs;
  // every rank must drive every other peer or larger-rank-side
  // spinners deadlock (see GetCbTransportCachesPerPeer).
  for (int p = 0; p < numRanks; ++p) {
    if (p == globalRank) {
      continue;
    }
    ASSERT_NE(
        comm_->ctran_->mapper->getP2pTransport(
            p, ctran::transport::HostTransportMode::kCopyBased),
        nullptr);
  }

  const int peer = (globalRank + 1) % numRanks;
  auto* cb = comm_->ctran_->mapper->getP2pTransport(
      peer, ctran::transport::HostTransportMode::kCopyBased);
  ASSERT_NE(cb, nullptr);

  // Snapshot the pool size while the CB transport holds borrowed
  // entries; expect non-zero.
  EXPECT_GT(comm_->ctran_->gpe->numInUseGpeKernelSyncs(), 0u);

  // Trigger the early cache-clear path.
  comm_->ctran_->mapper->setAtDestruction();
  EXPECT_EQ(comm_->ctran_->gpe->numInUseGpeKernelSyncs(), 0u);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
