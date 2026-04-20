// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/ReduceScatter/ReduceScatterImpl.h"
#include "comms/ctran/tests/cudagraph/CtranCudaGraphParamTest.h"

static AlgoDescriptor makeReduceScatter(enum NCCL_REDUCESCATTER_ALGO algo) {
  struct B : AlgoDescriptor::Buffers {
    ctran::TestDeviceBuffer send, recv;
    size_t bytes;
    B(size_t c, int rank, int nR)
        : send(c * nR * sizeof(int32_t)),
          recv(c * sizeof(int32_t)),
          bytes(c * sizeof(int32_t)) {
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
  desc.name = std::string("ReduceScatter_") + reduceScatterAlgoName(algo);
  desc.expectsHostNodes = [](CtranComm* comm, size_t count) {
    if (comm->statex_->nLocalRanks() == 1) {
      return true;
    }
    size_t sendBytes = count * sizeof(int32_t) * comm->statex_->nRanks();
    return sendBytes > NCCL_CTRAN_BCAST_NVL_SHARED_DEVBUF_SIZE;
  };
  desc.isSupported = [algo](CtranComm* comm, size_t, int) {
    return ctranReduceScatterSupport(comm, algo);
  };
  desc.makeBuffers = [](size_t c, int rank, int nR) {
    return std::make_shared<B>(c, rank, nR);
  };
  desc.capture = [algo](
                     AlgoDescriptor::Buffers* base,
                     size_t count,
                     ctran::testing::CaptureContext& ctx) {
    auto* b = static_cast<B*>(base);
    ASSERT_EQ(
        ctranReduceScatter(
            b->send.get(),
            b->recv.get(),
            count,
            commInt32,
            commSum,
            ctx.comm,
            ctx.stream,
            algo),
        commSuccess);
  };
  return desc;
}

DEFINE_CUDAGRAPH_PARAM_TEST(
    CudaGraphReduceScatter,
    makeReduceScatter(NCCL_REDUCESCATTER_ALGO::ctran),
    makeReduceScatter(NCCL_REDUCESCATTER_ALGO::ctdirect),
    makeReduceScatter(NCCL_REDUCESCATTER_ALGO::ctring));

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranCudaGraphEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
