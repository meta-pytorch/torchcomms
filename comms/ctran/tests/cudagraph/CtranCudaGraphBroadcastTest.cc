// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/Broadcast/BroadcastImpl.h"
#include "comms/ctran/tests/cudagraph/CtranCudaGraphParamTest.h"

static AlgoDescriptor makeBroadcast(enum NCCL_BROADCAST_ALGO algo) {
  static constexpr int kRoot = 0;
  struct B : AlgoDescriptor::Buffers {
    ctran::TestDeviceBuffer send, recv;
    size_t bytes;
    B(size_t c)
        : send(c * sizeof(int32_t)),
          recv(c * sizeof(int32_t)),
          bytes(c * sizeof(int32_t)) {
      CtranCudaGraphTestBase::fillSendBuf(send.get(), c, kRoot);
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
  desc.name = std::string("Broadcast_") + broadcastAlgoName(algo);
  desc.isSupported = [algo](CtranComm* comm, size_t, int) {
    return ctranBroadcastSupport(comm, algo);
  };
  desc.makeBuffers = [](size_t c, int, int) { return std::make_shared<B>(c); };
  desc.capture = [algo](
                     AlgoDescriptor::Buffers* base,
                     size_t count,
                     ctran::testing::CaptureContext& ctx) {
    auto* b = static_cast<B*>(base);
    ASSERT_EQ(
        ctranBroadcast(
            b->send.get(),
            b->recv.get(),
            count,
            commInt32,
            kRoot,
            ctx.comm,
            ctx.stream,
            algo),
        commSuccess);
  };
  return desc;
}

DEFINE_CUDAGRAPH_PARAM_TEST(
    CudaGraphBroadcast,
    makeBroadcast(NCCL_BROADCAST_ALGO::ctran),
    makeBroadcast(NCCL_BROADCAST_ALGO::ctdirect),
    makeBroadcast(NCCL_BROADCAST_ALGO::ctbtree));

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranCudaGraphEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
