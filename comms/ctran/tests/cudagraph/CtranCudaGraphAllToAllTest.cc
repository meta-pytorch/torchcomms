// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/tests/cudagraph/CtranCudaGraphParamTest.h"

static AlgoDescriptor makeAllToAll() {
  struct B : AlgoDescriptor::Buffers {
    ctran::TestDeviceBuffer send, recv;
    size_t bytes;
    B(size_t c, int rank, int nR)
        : send(c * nR * sizeof(int32_t)),
          recv(c * nR * sizeof(int32_t)),
          bytes(c * nR * sizeof(int32_t)) {
      CtranCudaGraphTestBase::fillSendBuf(send.get(), c * nR, rank);
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
  desc.name = "AllToAll_ctran";
  desc.expectsHostNodes = [](CtranComm* comm, size_t) {
    return comm->statex_->nLocalRanks() < comm->statex_->nRanks();
  };
  desc.isSupported = [](CtranComm* comm, size_t count, int) {
    return ctranAllToAllSupport(
        count, commInt32, comm, NCCL_ALLTOALL_ALGO::ctran);
  };
  desc.makeBuffers = [](size_t c, int rank, int nR) {
    return std::make_shared<B>(c, rank, nR);
  };
  desc.capture = [](AlgoDescriptor::Buffers* base,
                    size_t count,
                    ctran::testing::CaptureContext& ctx) {
    auto* b = static_cast<B*>(base);
    ASSERT_EQ(
        ctranAllToAll(
            b->send.get(),
            b->recv.get(),
            count,
            commInt32,
            ctx.comm,
            ctx.stream,
            NCCL_ALLTOALL_ALGO::ctran),
        commSuccess);
  };
  return desc;
}

static AlgoDescriptor makeAllToAllv() {
  struct B : AlgoDescriptor::Buffers {
    std::vector<size_t> sendcounts, sdispls, recvcounts, rdispls;
    size_t totalSend, totalRecv;
    ctran::TestDeviceBuffer send, recv;

    B(size_t, int rank, int nRanks)
        : totalSend(static_cast<size_t>(rank + 1) * 100 * nRanks),
          totalRecv(100 * nRanks * (nRanks + 1) / 2),
          send(totalSend * sizeof(int32_t)),
          recv(totalRecv * sizeof(int32_t)) {
      size_t perPeerSend = (rank + 1) * 100;
      sendcounts.assign(nRanks, perPeerSend);
      sdispls.resize(nRanks);
      for (int i = 0; i < nRanks; ++i) {
        sdispls[i] = i * perPeerSend;
      }
      recvcounts.resize(nRanks);
      rdispls.resize(nRanks);
      size_t offset = 0;
      for (int p = 0; p < nRanks; ++p) {
        recvcounts[p] = (p + 1) * 100;
        rdispls[p] = offset;
        offset += recvcounts[p];
      }
      CtranCudaGraphTestBase::fillSendBuf(send.get(), totalSend, rank);
    }
    void* sendbuf() override {
      return send.get();
    }
    void* recvbuf() override {
      return recv.get();
    }
    size_t recvBytes() override {
      return totalRecv * sizeof(int32_t);
    }
  };

  AlgoDescriptor desc;
  desc.name = "AllToAllv";
  desc.expectsHostNodes = [](CtranComm* comm, size_t) {
    return comm->statex_->nLocalRanks() < comm->statex_->nRanks();
  };
  desc.isSupported = [](CtranComm* comm, size_t, int) {
    return ctranAllToAllvSupport(comm);
  };
  desc.makeBuffers = [](size_t c, int rank, int nR) {
    return std::make_shared<B>(c, rank, nR);
  };
  desc.capture = [](AlgoDescriptor::Buffers* base,
                    size_t,
                    ctran::testing::CaptureContext& ctx) {
    auto* b = static_cast<B*>(base);
    ASSERT_EQ(
        ctranAllToAllv(
            b->send.get(),
            b->sendcounts.data(),
            b->sdispls.data(),
            b->recv.get(),
            b->recvcounts.data(),
            b->rdispls.data(),
            commInt32,
            ctx.comm,
            ctx.stream),
        commSuccess);
  };
  return desc;
}

static meta::comms::Hints makeAllToAllvDynamicHints() {
  meta::comms::Hints hints;
  hints.set("ncclx_alltoallv_dynamic_sendbuffs_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_recvbuffs_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_sendcounts_location", "gpu");
  hints.set("ncclx_alltoallv_dynamic_max_sendcounts_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_max_recvcounts_location", "cpu");
  hints.set("ncclx_alltoallv_dynamic_actual_recvcounts_location", "gpu");
  return hints;
}

static AlgoDescriptor makeAllToAllvDynamic() {
  struct B : AlgoDescriptor::Buffers {
    int nRanks;
    size_t maxCount;
    std::vector<ctran::TestDeviceBuffer> sendBufs, recvBufs;
    std::vector<const void*> sendPtrs;
    std::vector<void*> recvPtrs;
    std::vector<size_t> sendcounts;
    std::vector<size_t> actualRecvcounts;

    B(size_t c, int rank, int nR) : nRanks(nR), maxCount(c) {
      sendcounts.resize(nR, c);
      actualRecvcounts.resize(nR, 0);
      for (int i = 0; i < nR; ++i) {
        sendBufs.emplace_back(c * sizeof(int32_t));
        recvBufs.emplace_back(c * sizeof(int32_t));
        CtranCudaGraphTestBase::fillSendBuf(sendBufs.back().get(), c, rank);
      }
      sendPtrs.resize(nR);
      recvPtrs.resize(nR);
      for (int i = 0; i < nR; ++i) {
        sendPtrs[i] = sendBufs[i].get();
        recvPtrs[i] = recvBufs[i].get();
      }
    }
    void* sendbuf() override {
      return sendBufs[0].get();
    }
    void* recvbuf() override {
      return recvBufs[0].get();
    }
    size_t recvBytes() override {
      return maxCount * sizeof(int32_t);
    }
  };

  AlgoDescriptor desc;
  desc.name = "AllToAllvDynamic";
  desc.expectsHostNodes = [](CtranComm* comm, size_t) {
    return comm->statex_->nLocalRanks() < comm->statex_->nRanks();
  };
  desc.isSupported = [](CtranComm* comm, size_t count, int) {
    return ctranAllToAllvDynamicSupport(
               comm, makeAllToAllvDynamicHints(), count, count, commInt32) ==
        commSuccess;
  };
  desc.makeBuffers = [](size_t c, int rank, int nR) {
    return std::make_shared<B>(c, rank, nR);
  };
  desc.capture = [](AlgoDescriptor::Buffers* base,
                    size_t count,
                    ctran::testing::CaptureContext& ctx) {
    auto* b = static_cast<B*>(base);
    ASSERT_EQ(
        ctranAllToAllvDynamic(
            b->sendPtrs.data(),
            b->sendcounts.data(),
            b->recvPtrs.data(),
            count,
            count,
            b->actualRecvcounts.data(),
            makeAllToAllvDynamicHints(),
            commInt32,
            ctx.comm,
            ctx.stream),
        commSuccess);
  };
  return desc;
}

DEFINE_CUDAGRAPH_PARAM_TEST(
    CudaGraphAllToAll,
    makeAllToAll(),
    makeAllToAllv(),
    makeAllToAllvDynamic());

// AllToAllDedup buffers with persistent lifecycle (like AllGatherP).
// Arrays are indexed by nNodes (not nRanks). Each local rank handles a
// split of each node's data chunk.
struct AllToAllDedupBuffers : AlgoDescriptor::Buffers {
  std::vector<size_t> splitSendCounts, splitSendDisps;
  std::vector<size_t> splitRecvCounts, splitRecvDisps;
  size_t maxSendCount, maxRecvCount;
  ctran::TestDeviceBuffer send;
  void* recv{nullptr};
  CtranPersistentRequest* request{nullptr};

  AllToAllDedupBuffers(
      size_t count,
      int rank,
      int /*nRanks*/,
      CtranComm* comm,
      cudaStream_t stream)
      : maxSendCount(count * comm->statex_->nNodes()),
        maxRecvCount(count * comm->statex_->nNodes()),
        send(count * comm->statex_->nNodes() * sizeof(int32_t)) {
    auto statex = comm->statex_.get();
    int nNodes = statex->nNodes();
    int nLocalRanks = statex->nLocalRanks();
    int localRank = statex->localRank();

    // Split per-node counts for this local rank
    splitSendCounts.resize(nNodes);
    splitSendDisps.resize(nNodes);
    splitRecvCounts.resize(nNodes);
    splitRecvDisps.resize(nNodes);
    for (int i = 0; i < nNodes; ++i) {
      size_t splitSize = count / nLocalRanks;
      splitSendCounts[i] = splitSize;
      splitRecvCounts[i] = splitSize;
      splitSendDisps[i] = i * count + splitSize * localRank;
      splitRecvDisps[i] = i * count + splitSize * localRank;
    }

    CtranCudaGraphTestBase::fillSendBuf(send.get(), maxSendCount, rank);

    auto res = ctranAllToAllDedupInit(
        send.get(),
        splitSendCounts.data(),
        splitSendDisps.data(),
        maxSendCount,
        recv,
        splitRecvCounts.data(),
        splitRecvDisps.data(),
        maxRecvCount,
        commInt32,
        comm,
        stream,
        request);
    if (res != commSuccess) {
      throw std::runtime_error("ctranAllToAllDedupInit failed");
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error("cudaStreamSynchronize after init failed");
    }
  }

  ~AllToAllDedupBuffers() override {
    if (request) {
      ctranAllToAllDedupDestroy(request);
    }
  }

  void* sendbuf() override {
    return send.get();
  }
  void* recvbuf() override {
    return recv;
  }
  size_t recvBytes() override {
    return maxRecvCount * sizeof(int32_t);
  }
};

static AlgoDescriptor makeAllToAllDedup() {
  AlgoDescriptor desc;
  desc.name = "AllToAllDedup";
  desc.expectsHostNodes = [](CtranComm* comm, size_t) {
    return comm->statex_->nLocalRanks() < comm->statex_->nRanks();
  };
  desc.isSupported = [](CtranComm* comm, size_t, int) {
    auto statex = comm->statex_.get();
    // AllToAllDedup requires real multi-node topology (setupGpeOp
    // creates op only when nLocalRanks < nRanks). Reject:
    //   - single-node (nNodes=1, nLocalRanks=nRanks)
    //   - nolocal simulation (nNodes=nRanks, nLocalRanks=1)
    // Accept real multi-node (nNodes>1, nLocalRanks>1).
    return ctranAllToAllDedupSupport(comm) && statex->nNodes() > 1 &&
        statex->nLocalRanks() > 1;
  };
  desc.makeBuffers = nullptr;
  desc.capture = [](AlgoDescriptor::Buffers* base,
                    size_t,
                    ctran::testing::CaptureContext& ctx) {
    auto* b = static_cast<AllToAllDedupBuffers*>(base);
    auto persistStream = b->request->stream;

    // AllToAllDedupExec uses the persistent request's stream (bound at
    // init), not ctx.stream. Fork into persistStream so its operations
    // are included in the captured graph, then join back.
    meta::comms::CudaEvent forkEv, joinEv;
    ASSERT_EQ(cudaEventRecord(forkEv.get(), ctx.stream), cudaSuccess);
    ASSERT_EQ(cudaStreamWaitEvent(persistStream, forkEv.get(), 0), cudaSuccess);

    ASSERT_EQ(ctranAllToAllDedupExec(b->request), commSuccess);

    ASSERT_EQ(cudaEventRecord(joinEv.get(), persistStream), cudaSuccess);
    ASSERT_EQ(cudaStreamWaitEvent(ctx.stream, joinEv.get(), 0), cudaSuccess);
  };
  return desc;
}

class CudaGraphAllToAllDedup
    : public CtranCudaGraphTestBase,
      public ::testing::WithParamInterface<GraphTestParam> {};

TEST_P(CudaGraphAllToAllDedup, CudaGraphOp) {
  auto [desc, pattern, count, replayMult] = GetParam();
  int numReplays = baseReplays(pattern) * replayMult;

  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  if (!desc.isSupported(comm.get(), count, numRanks)) {
    GTEST_SKIP() << desc.name << " not supported";
  }

  meta::comms::CudaStream initStream(cudaStreamNonBlocking);
  desc.makeBuffers = [&](size_t c, int rank, int nR) {
    return std::make_shared<AllToAllDedupBuffers>(
        c, rank, nR, comm.get(), initStream.get());
  };

  runPattern(
      pattern, comm.get(), globalRank, numRanks, count, numReplays, desc);
}

std::string CudaGraphAllToAllDedupTestName(
    const ::testing::TestParamInfo<GraphTestParam>& info) {
  auto& [desc, pattern, count, replayMult] = info.param;
  return desc.name + "_" + patternToString(pattern) + "_" +
      std::to_string(count) + "_x" + std::to_string(replayMult);
}

INSTANTIATE_TEST_SUITE_P(
    CudaGraphAllToAllDedupTests,
    CudaGraphAllToAllDedup,
    ::testing::Combine(
        ::testing::Values(makeAllToAllDedup()),
        ::testing::Values(CUDAGRAPH_TEST_PATTERN),
        ::testing::Values(1024UL, 8192UL),
        ::testing::Values(1)),
    CudaGraphAllToAllDedupTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranCudaGraphEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
