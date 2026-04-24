// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Tests for the ctgraph AllGather algorithm.
// When algo=ctgraph, ctranAllGatherSupport returns true only during CUDA graph
// capture (and falls back to baseline otherwise). During capture,
// ctranAllGather transparently converts to the persistent window-based AGP
// algorithm.

#include <random>

#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/tests/cudagraph/CtranCudaGraphParamTest.h"

static AlgoDescriptor makeAllGatherCtgraph() {
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
  desc.name = "AllGatherCtgraph";
  desc.isSupported = [](CtranComm* comm, size_t, int) {
    return ctran::allGatherPSupport(comm);
  };
  desc.expectsHostNodes = [](CtranComm* comm, size_t) {
    auto statex = comm->statex_.get();
    return statex->nLocalRanks() < statex->nRanks();
  };
  desc.makeBuffers = [](size_t c, int rank, int nR) {
    return std::make_shared<B>(c, rank, nR);
  };
  desc.capture = [](AlgoDescriptor::Buffers* base,
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
            NCCL_ALLGATHER_ALGO::ctgraph),
        commSuccess);
  };
  return desc;
}

DEFINE_CUDAGRAPH_PARAM_TEST(CudaGraphAllGatherCtgraph, makeAllGatherCtgraph());

// Expandable segment test: verifies cudagraph-aware AllGather with
// kCuMemAllocDisjoint memory (multiple disjoint physical segments per buffer,
// 20MB each) and a random offset from the allocation base.
class CudaGraphAllGatherCtgraphExpandable
    : public CtranCudaGraphTestBase,
      public ::testing::WithParamInterface<size_t> {};

static constexpr size_t kSegmentSize = 20UL * 1024 * 1024;

static size_t segmentsNeeded(size_t bytes) {
  return (bytes + kSegmentSize - 1) / kSegmentSize;
}

TEST_P(CudaGraphAllGatherCtgraphExpandable, CaptureReplayVerify) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  if (!ctran::allGatherPSupport(comm.get())) {
    GTEST_SKIP() << "allGatherP not supported";
  }

  const size_t count = GetParam();
  const int nRanks = numRanks;

  std::mt19937 rng(globalRank);
  const size_t offsetElems = rng() % 4096 + 1;
  const size_t offsetBytes = offsetElems * sizeof(int32_t);

  const size_t sendDataBytes = count * sizeof(int32_t);
  const size_t recvDataBytes = count * nRanks * sizeof(int32_t);

  const size_t sendNumSeg = segmentsNeeded(sendDataBytes + offsetBytes);
  const size_t recvNumSeg = segmentsNeeded(recvDataBytes + offsetBytes);

  ctran::TestDeviceBuffer send(
      sendNumSeg * kSegmentSize, kCuMemAllocDisjoint, sendNumSeg);
  ctran::TestDeviceBuffer recv(
      recvNumSeg * kSegmentSize, kCuMemAllocDisjoint, recvNumSeg);

  auto* sendbuf = static_cast<char*>(send.get()) + offsetBytes;
  auto* recvbuf = static_cast<char*>(recv.get()) + offsetBytes;

  fillSendBuf(sendbuf, count, globalRank);

  meta::comms::CudaStream stream(cudaStreamNonBlocking);

  // Capture
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  ASSERT_EQ(
      cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal),
      cudaSuccess);
  ASSERT_EQ(
      ctranAllGather(
          sendbuf,
          recvbuf,
          count,
          commInt32,
          comm.get(),
          stream.get(),
          NCCL_ALLGATHER_ALGO::ctgraph),
      commSuccess);
  ASSERT_EQ(cudaStreamEndCapture(stream.get(), &graph), cudaSuccess);
  ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);

  // Replay
  ASSERT_EQ(cudaGraphLaunch(exec, stream.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);

  // Verify
  verifyAllGather(recvbuf, count, nRanks);

  ASSERT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(
    CudaGraphAllGatherCtgraphExpandableTests,
    CudaGraphAllGatherCtgraphExpandable,
    ::testing::Values(2097152UL, 10485760UL));

// Verifies that graph destruction cleans up without CUDA API errors.
// The retainUserObject destructor callback defers cleanup to comm destruction
// since CUDA APIs are forbidden in the callback context.
class CudaGraphAllGatherCtgraphDestroy : public CtranCudaGraphTestBase {};

TEST_F(CudaGraphAllGatherCtgraphDestroy, DestroyGraphCleanly) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  if (!ctran::allGatherPSupport(comm.get())) {
    GTEST_SKIP() << "allGatherP not supported";
  }

  const size_t count = 1024;
  const int nRanks = numRanks;
  ctran::TestDeviceBuffer send(count * sizeof(int32_t));
  ctran::TestDeviceBuffer recv(count * nRanks * sizeof(int32_t));
  fillSendBuf(send.get(), count, globalRank);

  meta::comms::CudaStream stream(cudaStreamNonBlocking);

  // Capture
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  ASSERT_EQ(
      cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal),
      cudaSuccess);
  ASSERT_EQ(
      ctranAllGather(
          send.get(),
          recv.get(),
          count,
          commInt32,
          comm.get(),
          stream.get(),
          NCCL_ALLGATHER_ALGO::ctgraph),
      commSuccess);
  ASSERT_EQ(cudaStreamEndCapture(stream.get(), &graph), cudaSuccess);
  ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);

  // Replay
  ASSERT_EQ(cudaGraphLaunch(exec, stream.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);

  // Destroy — triggers retainUserObject destructor callback.
  ASSERT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranCudaGraphEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
