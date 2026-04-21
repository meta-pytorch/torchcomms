// Copyright (c) Meta Platforms, Inc. and affiliates.

// SendRecv CUDA graph tests. Creates a full ncclComm via createNcclComm()
// for group API support (commGroupDepth / ctranGroupEndHook).

#include "comms/ctran/tests/cudagraph/CtranCudaGraphParamTest.h"
#include "comms/ctran/utils/CommGroupUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

static AlgoDescriptor makeSendRecv() {
  struct B : AlgoDescriptor::Buffers {
    ctran::TestDeviceBuffer send, recv;
    size_t bytes;
    int sendPeer, recvPeer;
    B(size_t c, int rank, int nR)
        : send(c * sizeof(int32_t)),
          recv(c * sizeof(int32_t)),
          bytes(c * sizeof(int32_t)),
          sendPeer((rank + 1) % nR),
          recvPeer((rank + nR - 1) % nR) {
      CtranCudaGraphTestBase::fillSendBuf(send.get(), c, rank);
    }
    void* sendbuf() override {
      return send.get();
    }
    void* recvbuf() override {
      return recv.get();
    }
    size_t recvBytes() override {
      return bytes;
    }
  };

  AlgoDescriptor desc;
  desc.name = "SendRecv_ctp2p";
  // ctp2p uses a kernel-only path for NVL peers (no GPE host callback).
  // For IB peers, sendRecvImpl runs on the GPE thread via a host node.
  desc.expectsHostNodes = [](CtranComm* comm, size_t) {
    int rank = comm->statex_->rank();
    int nRanks = comm->statex_->nRanks();
    int sendPeer = (rank + 1) % nRanks;
    int recvPeer = (rank + nRanks - 1) % nRanks;
    // If any peer uses IB, a GPE host node is needed.
    return comm->ctran_->mapper->getBackend(sendPeer) !=
        CtranMapperBackend::NVL ||
        comm->ctran_->mapper->getBackend(recvPeer) != CtranMapperBackend::NVL;
  };
  desc.isSupported = [](CtranComm* comm, size_t, int nRanks) {
    if (nRanks < 2) {
      return false;
    }
    int rank = comm->statex_->rank();
    int sendPeer = (rank + 1) % nRanks;
    return ctranSendRecvSupport(sendPeer, comm);
  };
  desc.makeBuffers = [](size_t c, int rank, int nR) {
    return std::make_shared<B>(c, rank, nR);
  };
  desc.capture = [](AlgoDescriptor::Buffers* base,
                    size_t count,
                    ctran::testing::CaptureContext& ctx) {
    auto* b = static_cast<B*>(base);
    commGroupDepth++;
    ASSERT_EQ(
        ctranSend(
            b->send.get(), count, commInt32, b->sendPeer, ctx.comm, ctx.stream),
        commSuccess);
    ASSERT_EQ(
        ctranRecv(
            b->recv.get(), count, commInt32, b->recvPeer, ctx.comm, ctx.stream),
        commSuccess);
    commGroupDepth--;
    ASSERT_EQ(ctranGroupEndHook(NCCL_SENDRECV_ALGO::ctp2p), commSuccess);
  };
  return desc;
}

class CudaGraphSendRecv : public CtranCudaGraphTestBase,
                          public ::testing::WithParamInterface<GraphTestParam> {
};

TEST_P(CudaGraphSendRecv, CudaGraphOp) {
  auto [desc, pattern, count, replayMult] = GetParam();
  int numReplays = baseReplays(pattern) * replayMult;

  // Create full ncclComm for group API support
  NcclCommRAII ncclComm(globalRank, numRanks, localRank, bootstrap_.get());
  auto* ctranComm = ncclComm->ctranComm_.get();

  if (!desc.isSupported(ctranComm, count, numRanks)) {
    GTEST_SKIP() << desc.name << " not supported";
  }

  runPattern(pattern, ctranComm, globalRank, numRanks, count, numReplays, desc);
}

std::string CudaGraphSendRecvTestName(
    const ::testing::TestParamInfo<GraphTestParam>& info) {
  auto& [desc, pattern, count, replayMult] = info.param;
  return desc.name + "_" + patternToString(pattern) + "_" +
      std::to_string(count) + "_x" + std::to_string(replayMult);
}

INSTANTIATE_TEST_SUITE_P(
    CudaGraphSendRecvTests,
    CudaGraphSendRecv,
    ::testing::Combine(
        ::testing::Values(makeSendRecv()),
        ::testing::Values(
            GraphPattern::Basic,
            GraphPattern::DestroyRecreate,
            GraphPattern::MixedEagerGraph,
            GraphPattern::MultiGraph,
            GraphPattern::InPlace,
            GraphPattern::MultipleSequential,
            GraphPattern::MultiStream),
        ::testing::Values(1024UL, 8192UL),
        ::testing::Values(1)),
    CudaGraphSendRecvTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranCudaGraphEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
