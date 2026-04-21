// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/tests/cudagraph/CtranCudaGraphParamTest.h"

static AlgoDescriptor makeAllGather(enum NCCL_ALLGATHER_ALGO algo) {
  struct B : AlgoDescriptor::Buffers {
    ctran::TestDeviceBuffer send, recv;
    size_t bytes;
    B(size_t c, int rank, int nR)
        : send(c * sizeof(int32_t)),
          recv(c * nR * sizeof(int32_t)),
          bytes(c * nR * sizeof(int32_t)) {
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
  desc.name = std::string("AllGather_") + allGatherAlgoName(algo);
  desc.isSupported = [algo](CtranComm* comm, size_t, int) {
    return ctranAllGatherSupport(comm, algo);
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
        ctranAllGather(
            b->send.get(),
            b->recv.get(),
            count,
            commInt32,
            ctx.comm,
            ctx.stream,
            algo),
        commSuccess);
  };
  return desc;
}

DEFINE_CUDAGRAPH_PARAM_TEST(
    CudaGraphAllGather,
    makeAllGather(NCCL_ALLGATHER_ALGO::ctran),
    makeAllGather(NCCL_ALLGATHER_ALGO::ctdirect),
    makeAllGather(NCCL_ALLGATHER_ALGO::ctring),
    makeAllGather(NCCL_ALLGATHER_ALGO::ctrd),
    makeAllGather(NCCL_ALLGATHER_ALGO::ctbrucks));

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranCudaGraphEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
