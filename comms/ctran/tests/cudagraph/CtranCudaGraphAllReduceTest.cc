// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/tests/cudagraph/CtranCudaGraphParamTest.h"

static AlgoDescriptor makeAllReduce(enum NCCL_ALLREDUCE_ALGO algo) {
  struct B : AlgoDescriptor::Buffers {
    ctran::TestDeviceBuffer send, recv;
    size_t bytes;
    B(size_t c, int rank)
        : send(c * sizeof(int32_t)),
          recv(c * sizeof(int32_t)),
          bytes(c * sizeof(int32_t)) {
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
  desc.name = std::string("AllReduce_") + allReduceAlgoName(algo);
  desc.isSupported = [algo](CtranComm* comm, size_t, int) {
    return ctranAllReduceSupport(comm, algo);
  };
  desc.makeBuffers = [](size_t c, int rank, int) {
    return std::make_shared<B>(c, rank);
  };
  desc.capture = [algo](
                     AlgoDescriptor::Buffers* base,
                     size_t count,
                     ctran::testing::CaptureContext& ctx) {
    auto* b = static_cast<B*>(base);
    ASSERT_EQ(
        ctranAllReduce(
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
    CudaGraphAllReduce,
    makeAllReduce(NCCL_ALLREDUCE_ALGO::ctran),
    makeAllReduce(NCCL_ALLREDUCE_ALGO::ctdirect),
    makeAllReduce(NCCL_ALLREDUCE_ALGO::ctring));

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranCudaGraphEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
