// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Tests for the ctgraph AllGather algorithm variants.
// ctgraph auto-selects based on topology; ctgraph_pipeline, ctgraph_rdpipeline,
// ctgraph_ring, ctgraph_rd allow explicit selection. All variants are only
// active during CUDA graph capture and fall back to baseline otherwise.

#include <random>

#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/tests/VerifyAlgoStatsUtil.h"
#include "comms/ctran/tests/cudagraph/CtranCudaGraphParamTest.h"

static AlgoDescriptor makeAllGatherCtgraph(
    enum NCCL_ALLGATHER_ALGO algo = NCCL_ALLGATHER_ALGO::ctgraph) {
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
  desc.name = allGatherAlgoName(algo);
  desc.isSupported = [algo](CtranComm* comm, size_t, int) {
    if (!ctran::allGatherPSupport(comm)) {
      return false;
    }
    const auto statex = comm->statex_.get();
    if ((algo == NCCL_ALLGATHER_ALGO::ctgraph_ring ||
         algo == NCCL_ALLGATHER_ALGO::ctgraph_rd) &&
        statex->nLocalRanks() > 1) {
      return false;
    }
    if (algo == NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline) {
      const auto nNodes = statex->nNodes();
      if (nNodes > 1 && (nNodes & (nNodes - 1)) != 0) {
        return false;
      }
    }
    return true;
  };
  desc.expectsHostNodes = [](CtranComm* comm, size_t) {
    auto statex = comm->statex_.get();
    return statex->nLocalRanks() < statex->nRanks();
  };
  desc.makeBuffers = [](size_t c, int rank, int nR) {
    return std::make_shared<B>(c, rank, nR);
  };
  desc.capture = [algo](
                     AlgoDescriptor::Buffers* base,
                     size_t count,
                     ctran::testing::CaptureContext& ctx) {
    auto* b = static_cast<B*>(base);
    // Use specified algo during capture, ctran for eager warmup
    cudaStreamCaptureStatus status;
    ASSERT_EQ(cudaStreamIsCapturing(ctx.stream, &status), cudaSuccess);
    const auto resolvedAlgo = (status == cudaStreamCaptureStatusActive)
        ? algo
        : NCCL_ALLGATHER_ALGO::ctran;
    ASSERT_EQ(
        ctranAllGather(
            b->send.get(),
            b->recv.get(),
            count,
            commInt32,
            ctx.comm,
            ctx.stream,
            resolvedAlgo),
        commSuccess);
  };
  return desc;
}

DEFINE_CUDAGRAPH_PARAM_TEST(
    CudaGraphAllGatherCtgraph,
    makeAllGatherCtgraph(),
    makeAllGatherCtgraph(NCCL_ALLGATHER_ALGO::ctgraph_pipeline),
    makeAllGatherCtgraph(NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline),
    makeAllGatherCtgraph(NCCL_ALLGATHER_ALGO::ctgraph_ring),
    makeAllGatherCtgraph(NCCL_ALLGATHER_ALGO::ctgraph_rd));

// Expandable segment test: verifies cudagraph-aware AllGather with
// kCuMemAllocDisjoint memory (multiple disjoint physical segments per buffer,
// 20MB each) and a random offset from the allocation base.
class CudaGraphAllGatherCtgraphExpandable
    : public CtranCudaGraphTestBase,
      public ::testing::WithParamInterface<
          std::tuple<enum NCCL_ALLGATHER_ALGO, size_t>> {};

static constexpr size_t kSegmentSize = 20UL * 1024 * 1024;

static size_t segmentsNeeded(size_t bytes) {
  return (bytes + kSegmentSize - 1) / kSegmentSize;
}

TEST_P(CudaGraphAllGatherCtgraphExpandable, CaptureReplayVerify) {
  const auto [algo, count] = GetParam();
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  if (!ctran::allGatherPSupport(comm.get())) {
    GTEST_SKIP() << "allGatherP not supported";
  }

  const auto statex = comm->statex_.get();
  if ((algo == NCCL_ALLGATHER_ALGO::ctgraph_ring ||
       algo == NCCL_ALLGATHER_ALGO::ctgraph_rd) &&
      statex->nLocalRanks() > 1) {
    GTEST_SKIP() << allGatherAlgoName(algo) << " requires nLocalRanks == 1";
  }
  if (algo == NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline) {
    const auto nNodes = statex->nNodes();
    if (nNodes > 1 && (nNodes & (nNodes - 1)) != 0) {
      GTEST_SKIP() << allGatherAlgoName(algo)
                   << " requires nNodes to be a power of 2";
    }
  }

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
          sendbuf, recvbuf, count, commInt32, comm.get(), stream.get(), algo),
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

std::string expandableTestName(
    const ::testing::TestParamInfo<
        std::tuple<enum NCCL_ALLGATHER_ALGO, size_t>>& info) {
  const auto& [algo, count] = info.param;
  return allGatherAlgoName(algo) + "_" + std::to_string(count);
}

INSTANTIATE_TEST_SUITE_P(
    CudaGraphAllGatherCtgraphExpandableTests,
    CudaGraphAllGatherCtgraphExpandable,
    ::testing::Combine(
        ::testing::Values(
            NCCL_ALLGATHER_ALGO::ctgraph,
            NCCL_ALLGATHER_ALGO::ctgraph_pipeline,
            NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline,
            NCCL_ALLGATHER_ALGO::ctgraph_ring,
            NCCL_ALLGATHER_ALGO::ctgraph_rd),
        ::testing::Values(2097152UL, 10485760UL)),
    expandableTestName);

// Verifies ctgraph auto-select picks the correct algorithm based on topology
// and message size. Parameterized by sendcount (element count).
// nLocalRanks > 1 → pipeline; nLocalRanks == 1 + small msg → rd;
// nLocalRanks == 1 + large msg (>= threshold) → ring.
struct AutoSelectParam {
  std::string name;
  size_t count;
  std::string expectedAlgoWhenLocal;
  std::string expectedAlgoWhenNolocal;
};

class CudaGraphAllGatherCtgraphAutoSelect
    : public CtranCudaGraphTestBase,
      public ::testing::WithParamInterface<AutoSelectParam> {
 protected:
  ctran::test::VerifyAlgoStatsHelper algoStats_;

  void SetUp() override {
    CtranCudaGraphTestBase::SetUp();
    algoStats_.enable();
  }
};

TEST_P(CudaGraphAllGatherCtgraphAutoSelect, CaptureReplayVerify) {
  const auto& param = GetParam();
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  if (!ctran::allGatherPSupport(comm.get())) {
    GTEST_SKIP() << "allGatherP not supported";
  }

  const size_t count = param.count;
  const int nRanks = numRanks;
  const auto statex = comm->statex_.get();

  // Large msg test only meaningful with nLocalRanks == 1
  if (count * sizeof(int32_t) >= NCCL_CTGRAPH_ALLGATHER_RING_THRESHOLD &&
      statex->nLocalRanks() > 1) {
    GTEST_SKIP() << "Large message ring test requires nLocalRanks == 1";
  }

  ctran::TestDeviceBuffer send(count * sizeof(int32_t));
  ctran::TestDeviceBuffer recv(count * nRanks * sizeof(int32_t));
  fillSendBuf(send.get(), count, globalRank);

  meta::comms::CudaStream stream(cudaStreamNonBlocking);

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

  ASSERT_EQ(cudaGraphLaunch(exec, stream.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);

  verifyAllGather(recv.get(), count, nRanks);

  const auto& expectedAlgo = (statex->nLocalRanks() > 1)
      ? param.expectedAlgoWhenLocal
      : param.expectedAlgoWhenNolocal;
  algoStats_.verify(comm.get(), "AllGather", expectedAlgo);

  ASSERT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
}

// 128MB / sizeof(int32_t) — matches NCCL_CTGRAPH_ALLGATHER_RING_THRESHOLD
// default. Hardcoded because INSTANTIATE runs at static init before cvars are
// initialized.
static constexpr size_t kRingThresholdCount = 134217728UL / sizeof(int32_t);

INSTANTIATE_TEST_SUITE_P(
    CudaGraphAllGatherCtgraphAutoSelectTests,
    CudaGraphAllGatherCtgraphAutoSelect,
    ::testing::Values(
        AutoSelectParam{"SmallMsg", 1024, "Pipeline", "Rd"},
        AutoSelectParam{"LargeMsg", kRingThresholdCount, "Pipeline", "Ring"}),
    [](const ::testing::TestParamInfo<AutoSelectParam>& info) {
      return info.param.name;
    });

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
