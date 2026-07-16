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

DEFINE_CUDAGRAPH_PARAM_TEST(CudaGraphAllToAll, makeAllToAll(), makeAllToAllv());

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranCudaGraphEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
