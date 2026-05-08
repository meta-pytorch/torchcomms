// Copyright (c) Meta Platforms, Inc. and affiliates.

// AllGatherP (persistent AllGather) CUDA graph tests.
// AllGatherP uses init/exec/destroy lifecycle: init exchanges handles once,
// exec is captured into the graph, destroy cleans up.

#include "comms/ctran/Ctran.h"
#include "comms/ctran/tests/cudagraph/CtranCudaGraphParamTest.h"
#include "comms/utils/cvars/nccl_cvars.h"

// Buffers struct that manages AllGatherP persistent lifecycle.
struct AllGatherPBuffers : AlgoDescriptor::Buffers {
  ctran::TestDeviceBuffer send, recv;
  size_t bytes;
  CtranComm* comm;
  CtranPersistentRequest* request{nullptr};
  void* recvHdl{nullptr};

  AllGatherPBuffers(
      size_t c,
      int rank,
      int nR,
      CtranComm* comm_,
      cudaStream_t stream)
      : send(c * sizeof(int32_t)),
        recv(c * nR * sizeof(int32_t)),
        bytes(c * nR * sizeof(int32_t)),
        comm(comm_) {
    CtranCudaGraphTestBase::fillSendBuf(send.get(), c, rank);

    auto res = comm->ctran_->commRegister(recv.get(), bytes, &recvHdl);
    if (res != commSuccess) {
      throw std::runtime_error("commRegister failed");
    }

    meta::comms::Hints hints;
    res = ctran::allGatherPInit(
        recv.get(), c * nR, hints, commInt32, comm, stream, request);
    if (res != commSuccess) {
      throw std::runtime_error("allGatherPInit failed");
    }

    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error("cudaStreamSynchronize after init failed");
    }
  }

  ~AllGatherPBuffers() override {
    if (request) {
      ctran::allGatherPDestroy(request);
      delete request;
    }
    if (recvHdl) {
      comm->ctran_->commDeregister(recvHdl);
    }
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

static AlgoDescriptor makeAllGatherP(enum NCCL_ALLGATHER_P_ALGO algo) {
  AlgoDescriptor desc;
  std::string algoName =
      (algo == NCCL_ALLGATHER_P_ALGO::ctdirect) ? "ctdirect" : "ctpipeline";
  desc.name = "AllGatherP_" + algoName;
  desc.expectsHostNodes = [](CtranComm*, size_t) { return false; };
  desc.isSupported = [](CtranComm* comm, size_t, int) {
    return ctran::allGatherPSupport(comm);
  };
  // makeBuffers is overridden in the test body (needs comm/stream for init).
  desc.makeBuffers = nullptr;
  desc.capture = [](AlgoDescriptor::Buffers* base,
                    size_t count,
                    ctran::testing::CaptureContext& ctx) {
    auto* b = static_cast<AllGatherPBuffers*>(base);
    auto persistStream = b->request->stream;

    // AllGatherP uses stream_ (bound at init). Fork it into the capture
    // so its operations are included in the graph, then join back.
    meta::comms::CudaEvent forkEv, joinEv;
    ASSERT_EQ(cudaEventRecord(forkEv.get(), ctx.stream), cudaSuccess);
    ASSERT_EQ(cudaStreamWaitEvent(persistStream, forkEv.get(), 0), cudaSuccess);

    ASSERT_EQ(
        ctran::allGatherPExec(b->send.get(), count, commInt32, b->request),
        commSuccess);

    ASSERT_EQ(cudaEventRecord(joinEv.get(), persistStream), cudaSuccess);
    ASSERT_EQ(cudaStreamWaitEvent(ctx.stream, joinEv.get(), 0), cudaSuccess);
  };
  return desc;
}

class CudaGraphAllGatherP
    : public CtranCudaGraphTestBase,
      public ::testing::WithParamInterface<GraphTestParam> {};

TEST_P(CudaGraphAllGatherP, CudaGraphOp) {
  auto [desc, pattern, count, replayMult] = GetParam();
  int numReplays = baseReplays(pattern) * replayMult;

  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  if (!desc.isSupported(comm.get(), count, numRanks)) {
    GTEST_SKIP() << desc.name << " not supported";
  }

  meta::comms::CudaStream initStream(cudaStreamNonBlocking);

  // Override makeBuffers to pass comm/stream for persistent init.
  desc.makeBuffers = [&](size_t c, int rank, int nR) {
    return std::make_shared<AllGatherPBuffers>(
        c, rank, nR, comm.get(), initStream.get());
  };

  runPattern(
      pattern, comm.get(), globalRank, numRanks, count, numReplays, desc);
}

std::string CudaGraphAllGatherPTestName(
    const ::testing::TestParamInfo<GraphTestParam>& info) {
  auto& [desc, pattern, count, replayMult] = info.param;
  return desc.name + "_" + patternToString(pattern) + "_" +
      std::to_string(count) + "_x" + std::to_string(replayMult);
}

INSTANTIATE_TEST_SUITE_P(
    CudaGraphAllGatherPTests,
    CudaGraphAllGatherP,
    ::testing::Combine(
        ::testing::Values(
            makeAllGatherP(NCCL_ALLGATHER_P_ALGO::ctdirect),
            makeAllGatherP(NCCL_ALLGATHER_P_ALGO::ctpipeline)),
        ::testing::Values(CUDAGRAPH_TEST_PATTERN),
        ::testing::Values(1024UL, 8192UL),
        ::testing::Values(1)),
    CudaGraphAllGatherPTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranCudaGraphEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
