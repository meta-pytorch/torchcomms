// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Tests for cudagraph-aware SendRecv with ctgraph algorithm.
// Pre-registers send/recv buffers during capture, delegates to the normal
// eager dispatch. Currently requires IB backend for the peer.
// TODO: Enable window-based global registration for NVL send/recv in graph
// mode, similar to AllGather ctgraph_pipeline.

#include <random>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/SendRecv/SendRecvImpl.h"
#include "comms/ctran/tests/VerifyAlgoStatsUtil.h"
#include "comms/ctran/tests/cudagraph/CtranCudaGraphParamTest.h"
#include "comms/ctran/utils/CommGroupUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

static AlgoDescriptor makeSendRecvCtgraph() {
  struct B : AlgoDescriptor::Buffers {
    ctran::TestDeviceBuffer send, recv;
    size_t bytes;
    int sendPeer, recvPeer;
    B(size_t c, int rank, int nR)
        : send(c * sizeof(int32_t)),
          recv(c * sizeof(int32_t)),
          bytes(c * sizeof(int32_t)),
          sendPeer((rank + 1) % nR),
          recvPeer((rank - 1 + nR) % nR) {
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
  desc.name = "SendRecvCtgraph";
  desc.isSupported = [](CtranComm* comm, size_t, int nRanks) {
    if (nRanks < 2) {
      return false;
    }
    const auto statex = comm->statex_.get();
    const int sendPeer = (statex->rank() + 1) % nRanks;
    return comm->ctran_->mapper->getBackend(sendPeer) == CtranMapperBackend::IB;
  };
  desc.expectsHostNodes = [](CtranComm*, size_t) { return true; };
  desc.makeBuffers = [](size_t c, int rank, int nR) {
    return std::make_shared<B>(c, rank, nR);
  };
  desc.capture = [](AlgoDescriptor::Buffers* base,
                    size_t count,
                    ctran::testing::CaptureContext& ctx) {
    auto* b = static_cast<B*>(base);
    cudaStreamCaptureStatus status;
    ASSERT_EQ(cudaStreamIsCapturing(ctx.stream, &status), cudaSuccess);
    const auto algo = (status == cudaStreamCaptureStatusActive)
        ? NCCL_SENDRECV_ALGO::ctgraph
        : NCCL_SENDRECV_ALGO::ctran;
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
    ASSERT_EQ(ctranGroupEndHook(algo, std::nullopt), commSuccess);
  };
  return desc;
}

DEFINE_CUDAGRAPH_PARAM_TEST(CudaGraphSendRecvCtgraph, makeSendRecvCtgraph());

// Expandable segment test: SendRecv ctgraph with kCuMemAllocDisjoint memory
// (multiple disjoint physical segments, 20MB each) and random offset.
class CudaGraphSendRecvCtgraphExpandable
    : public CtranCudaGraphTestBase,
      public ::testing::WithParamInterface<size_t> {
 protected:
  ctran::test::VerifyAlgoStatsHelper algoStats_;

  void SetUp() override {
    CtranCudaGraphTestBase::SetUp();
    algoStats_.enable();
  }
};

static constexpr size_t kSegmentSize = 20UL * 1024 * 1024;

static size_t segmentsNeeded(size_t bytes) {
  return (bytes + kSegmentSize - 1) / kSegmentSize;
}

TEST_P(CudaGraphSendRecvCtgraphExpandable, CaptureReplayVerify) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  const int rank = globalRank;
  const int nRanks = numRanks;
  const int sendPeer = (rank + 1) % nRanks;
  const int recvPeer = (rank - 1 + nRanks) % nRanks;

  if (nRanks < 2) {
    GTEST_SKIP() << "SendRecv requires at least 2 ranks";
  }
  if (comm->ctran_->mapper->getBackend(sendPeer) != CtranMapperBackend::IB ||
      comm->ctran_->mapper->getBackend(recvPeer) != CtranMapperBackend::IB) {
    GTEST_SKIP() << "SendRecv ctgraph requires IB backend for peers";
  }

  const size_t count = GetParam();
  const size_t dataBytes = count * sizeof(int32_t);

  std::mt19937 rng(rank);
  const size_t offsetElems = rng() % 4096 + 1;
  const size_t offsetBytes = offsetElems * sizeof(int32_t);

  const size_t numSeg = segmentsNeeded(dataBytes + offsetBytes);

  ctran::TestDeviceBuffer send(
      numSeg * kSegmentSize, kCuMemAllocDisjoint, numSeg);
  ctran::TestDeviceBuffer recv(
      numSeg * kSegmentSize, kCuMemAllocDisjoint, numSeg);

  auto* sendbuf = static_cast<char*>(send.get()) + offsetBytes;
  auto* recvbuf = static_cast<char*>(recv.get()) + offsetBytes;

  fillSendBuf(sendbuf, count, rank);
  CUDACHECK_TEST(cudaMemset(recvbuf, 0, dataBytes));

  meta::comms::CudaStream stream(cudaStreamNonBlocking);

  EnvRAII algoEnv(NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::ctgraph);

  // Capture
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  ASSERT_EQ(
      cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal),
      cudaSuccess);

  commGroupDepth++;
  ASSERT_EQ(
      ctranSend(sendbuf, count, commInt32, sendPeer, comm.get(), stream.get()),
      commSuccess);
  ASSERT_EQ(
      ctranRecv(recvbuf, count, commInt32, recvPeer, comm.get(), stream.get()),
      commSuccess);
  commGroupDepth--;
  ASSERT_EQ(
      ctranGroupEndHook(NCCL_SENDRECV_ALGO::ctgraph, std::nullopt),
      commSuccess);

  ASSERT_EQ(cudaStreamEndCapture(stream.get(), &graph), cudaSuccess);
  ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);

  // Replay
  ASSERT_EQ(cudaGraphLaunch(exec, stream.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);

  // Verify
  std::vector<int32_t> host(count);
  CUDACHECK_TEST(
      cudaMemcpy(host.data(), recvbuf, dataBytes, cudaMemcpyDefault));
  for (size_t i = 0; i < count; i++) {
    ASSERT_EQ(host[i], recvPeer) << "mismatch at elem " << i;
  }

  algoStats_.verify(comm.get(), "SendRecv", "Ctran");

  ASSERT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(
    CudaGraphSendRecvCtgraphExpandableTests,
    CudaGraphSendRecvCtgraphExpandable,
    ::testing::Values(2097152UL, 10485760UL));

// Verify ctranSendRecvSupport returns false for ctgraph when preconditions
// are not met: null stream, non-capturing stream, or NVL-only peer topology.
class CudaGraphSendRecvCtgraphSupport : public CtranCudaGraphTestBase {};

TEST_F(CudaGraphSendRecvCtgraphSupport, SupportCheckCtgraph) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }

  const int peer = (globalRank + 1) % numRanks;

  EnvRAII algoEnv(NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::ctgraph);

  const auto ctgraph = NCCL_SENDRECV_ALGO::ctgraph;

  EXPECT_FALSE(ctranSendRecvSupport(peer, comm.get(), ctgraph, nullptr))
      << "should return false with null stream";

  meta::comms::CudaStream stream(cudaStreamNonBlocking);
  EXPECT_FALSE(ctranSendRecvSupport(peer, comm.get(), ctgraph, stream.get()))
      << "should return false outside capture";

  const auto backend = comm->ctran_->mapper->getBackend(peer);
  if (backend != CtranMapperBackend::IB) {
    ASSERT_EQ(
        cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeRelaxed),
        cudaSuccess);
    EXPECT_FALSE(ctranSendRecvSupport(peer, comm.get(), ctgraph, stream.get()))
        << "should return false for non-IB peer during capture";
    cudaGraph_t graph;
    cudaStreamEndCapture(stream.get(), &graph);
    if (graph) {
      cudaGraphDestroy(graph);
    }
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranCudaGraphEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
