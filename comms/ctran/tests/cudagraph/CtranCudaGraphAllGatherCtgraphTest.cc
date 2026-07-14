// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Tests for the ctgraph AllGather algorithm variants.
// ctgraph auto-selects based on topology; ctgraph_pipeline, ctgraph_rdpipeline,
// ctgraph_ring, ctgraph_rd allow explicit selection. All variants are only
// active during CUDA graph capture and fall back to baseline otherwise.

#include <nccl.h>
#include <cstring>
#include <random>

#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/regcache/IpcRegCache.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/tests/VerifyAlgoStatsUtil.h"
#include "comms/ctran/tests/cudagraph/CtranCudaGraphParamTest.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/MathUtils.h"

#define NCCLCHECK_TEST_BENCH(cmd)                                         \
  do {                                                                    \
    ncclResult_t r = cmd;                                                 \
    ASSERT_EQ(r, ncclSuccess) << "NCCL error: " << ncclGetErrorString(r); \
  } while (0)

// Caches the recvbuf's segment in the regcache allocator cache for the
// lifetime of this object, mirroring the production CCA memory hook. The
// graph AGP path's acquireScopedRegister requires the recvbuf segment to be
// cached. Deregistration in the destructor MUST run after the capturing
// graph/request is destroyed (the scoped registration ref is held until
// then), which callers guarantee by declaring the guard so it outlives the
// graph.
class ScopedRecvBufReg {
 public:
  ScopedRecvBufReg(void* buf, size_t bytes) : buf_(buf), bytes_(bytes) {
    ctran::RegCache::getInstance()->globalRegister(
        buf_, bytes_, /*forceReg=*/true);
  }
  ~ScopedRecvBufReg() {
    if (buf_ != nullptr) {
      ctran::RegCache::getInstance()->globalDeregister(buf_, bytes_);
    }
  }
  ScopedRecvBufReg(ScopedRecvBufReg&& other) noexcept
      : buf_(other.buf_), bytes_(other.bytes_) {
    other.buf_ = nullptr;
  }
  ScopedRecvBufReg(const ScopedRecvBufReg&) = delete;
  ScopedRecvBufReg& operator=(const ScopedRecvBufReg&) = delete;
  ScopedRecvBufReg& operator=(ScopedRecvBufReg&&) = delete;

 private:
  void* buf_;
  size_t bytes_;
};

static AlgoDescriptor makeAllGatherCtgraph(
    enum NCCL_ALLGATHER_ALGO algo = NCCL_ALLGATHER_ALGO::ctgraph) {
  struct B : AlgoDescriptor::Buffers {
    ctran::TestDeviceBuffer send, recv;
    size_t bytes;
    ScopedRecvBufReg recvReg;
    B(size_t c, int rank, int nR)
        : send(c * sizeof(int32_t)),
          recv(c * nR * sizeof(int32_t)),
          bytes(c * nR * sizeof(int32_t)),
          recvReg(recv.get(), c * nR * sizeof(int32_t)) {
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
    if ((algo == NCCL_ALLGATHER_ALGO::ctgraph_pipeline ||
         algo == NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline) &&
        statex->nLocalRanks() <= 1) {
      return false;
    }
    if (algo == NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline &&
        !ctran::utils::isPowerOfTwo(statex->nNodes())) {
      return false;
    }
    if (algo == NCCL_ALLGATHER_ALGO::ctgraph_rd &&
        !ctran::utils::isPowerOfTwo(statex->nRanks())) {
      return false;
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
  if ((algo == NCCL_ALLGATHER_ALGO::ctgraph_pipeline ||
       algo == NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline) &&
      statex->nLocalRanks() <= 1) {
    GTEST_SKIP() << allGatherAlgoName(algo) << " requires nLocalRanks > 1";
  }
  if (algo == NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline &&
      !ctran::utils::isPowerOfTwo(statex->nNodes())) {
    GTEST_SKIP() << allGatherAlgoName(algo)
                 << " requires nNodes to be a power of 2";
  }
  if (algo == NCCL_ALLGATHER_ALGO::ctgraph_rd &&
      !ctran::utils::isPowerOfTwo(statex->nRanks())) {
    GTEST_SKIP() << allGatherAlgoName(algo)
                 << " requires nRanks to be a power of 2";
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

  ScopedRecvBufReg recvReg(recv.get(), recvNumSeg * kSegmentSize);

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
// nLocalRanks > 1 + small msg + power-of-2 nNodes → rdpipeline;
// nLocalRanks > 1 + large msg or non-power-of-2 nNodes → pipeline;
// nLocalRanks == 1 + small msg + power-of-2 nRanks → rd/ctsrd;
// nLocalRanks == 1 + large msg or non-power-of-2 nRanks → ring.
struct AutoSelectParam {
  std::string name;
  size_t count;
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

  ctran::TestDeviceBuffer send(count * sizeof(int32_t));
  ctran::TestDeviceBuffer recv(count * nRanks * sizeof(int32_t));
  fillSendBuf(send.get(), count, globalRank);

  ScopedRecvBufReg recvReg(recv.get(), count * nRanks * sizeof(int32_t));

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

  const bool largeMessage =
      count * sizeof(int32_t) >= NCCL_CTGRAPH_ALLGATHER_RING_THRESHOLD;
  const std::string expectedAlgo = (statex->nLocalRanks() > 1)
      ? ((!largeMessage && ctran::utils::isPowerOfTwo(statex->nNodes()))
             ? "StreamedRd"
             : "Pipeline")
      : ((!largeMessage && ctran::utils::isPowerOfTwo(statex->nRanks()))
             ? "StreamedRd"
             : "Ring");
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
        AutoSelectParam{"SmallMsg", 1024},
        AutoSelectParam{"LargeMsg", kRingThresholdCount}),
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

  ScopedRecvBufReg recvReg(recv.get(), count * nRanks * sizeof(int32_t));

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

// Multiple independent comms each capture and replay a ctgraph AllGather
// back-to-back without syncing between captures, so one comm's windowed
// registration is set up while another's is still in flight.
TEST_F(CtranCudaGraphTestBase, BackToBackMultiCommCtgraph) {
  constexpr int kComms = 2;
  constexpr int kReplays = 3;
  const size_t count = kDefaultCount;
  const int nRanks = numRanks;

  std::vector<std::unique_ptr<CtranComm>> comms;
  std::vector<ctran::TestDeviceBuffer> sends;
  std::vector<ctran::TestDeviceBuffer> recvs;
  std::vector<meta::comms::CudaStream> streams;
  std::vector<cudaGraph_t> graphs(kComms, nullptr);
  std::vector<cudaGraphExec_t> execs(kComms, nullptr);
  std::vector<ScopedRecvBufReg> recvRegs;
  recvRegs.reserve(kComms);

  for (int c = 0; c < kComms; ++c) {
    auto comm = makeCtranComm();
    ASSERT_NE(comm, nullptr);
    if (!ctran::allGatherPSupport(comm.get())) {
      GTEST_SKIP() << "allGatherP not supported";
    }
    if (comm->statex_->nLocalRanks() <= 1) {
      GTEST_SKIP() << "windowed pipeline AGP requires nLocalRanks > 1";
    }
    comms.push_back(std::move(comm));
    sends.emplace_back(count * sizeof(int32_t));
    recvs.emplace_back(count * nRanks * sizeof(int32_t));
    streams.emplace_back(cudaStreamNonBlocking);
    fillSendBuf(sends[c].get(), count, globalRank);
    recvRegs.emplace_back(recvs[c].get(), count * nRanks * sizeof(int32_t));
  }

  // Capture one graph per comm back-to-back, without syncing between captures,
  // so the per-comm windowed-AGP setups overlap.
  for (int c = 0; c < kComms; ++c) {
    ASSERT_EQ(
        cudaStreamBeginCapture(streams[c].get(), cudaStreamCaptureModeGlobal),
        cudaSuccess);
    ASSERT_EQ(
        ctranAllGather(
            sends[c].get(),
            recvs[c].get(),
            count,
            commInt32,
            comms[c].get(),
            streams[c].get(),
            NCCL_ALLGATHER_ALGO::ctgraph_pipeline),
        commSuccess);
    ASSERT_EQ(cudaStreamEndCapture(streams[c].get(), &graphs[c]), cudaSuccess);
    ASSERT_EQ(cudaGraphInstantiate(&execs[c], graphs[c], 0), cudaSuccess);
  }

  for (int iter = 0; iter < kReplays; ++iter) {
    for (int c = 0; c < kComms; ++c) {
      ASSERT_EQ(cudaGraphLaunch(execs[c], streams[c].get()), cudaSuccess);
    }
  }
  for (int c = 0; c < kComms; ++c) {
    ASSERT_EQ(cudaStreamSynchronize(streams[c].get()), cudaSuccess);
    verifyAllGather(recvs[c].get(), count, nRanks);
  }

  for (int c = 0; c < kComms; ++c) {
    ASSERT_EQ(cudaGraphExecDestroy(execs[c]), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graphs[c]), cudaSuccess);
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (int c = 0; c < kComms; ++c) {
    waitAndVerifyGpeClean(comms[c].get());
  }
}

class CudaGraphAllGatherCtgraphSharedRecvbuf
    : public CtranCudaGraphTestBase,
      public ::testing::WithParamInterface<enum NCCL_ALLGATHER_ALGO> {};

TEST_P(
    CudaGraphAllGatherCtgraphSharedRecvbuf,
    ReusedRecvbufSurvivesOneCommCleanup) {
  const auto algo = GetParam();
  auto comm1 = makeCtranComm();
  auto comm2 = makeCtranComm();
  ASSERT_NE(comm1, nullptr);
  ASSERT_NE(comm2, nullptr);

  if (!ctran::allGatherPSupport(comm1.get()) ||
      !ctran::allGatherPSupport(comm2.get())) {
    GTEST_SKIP() << "allGatherP not supported";
  }

  const auto statex = comm1->statex_.get();
  if (statex->nLocalRanks() < 2) {
    GTEST_SKIP() << "Need local NVL peers for pipeline ctgraph AGP";
  }
  if (algo == NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline &&
      !ctran::utils::isPowerOfTwo(statex->nNodes())) {
    GTEST_SKIP() << allGatherAlgoName(algo)
                 << " requires nNodes to be a power of 2";
  }

  const size_t count = kDefaultCount;
  const int nRanks = numRanks;
  ctran::TestDeviceBuffer send1(count * sizeof(int32_t));
  ctran::TestDeviceBuffer send2(count * sizeof(int32_t));
  ctran::TestDeviceBuffer recv(count * nRanks * sizeof(int32_t));
  fillSendBuf(send1.get(), count, globalRank);
  fillSendBuf(send2.get(), count, globalRank);

  ScopedRecvBufReg recvReg(recv.get(), count * nRanks * sizeof(int32_t));

  meta::comms::CudaStream stream1(cudaStreamNonBlocking);
  meta::comms::CudaStream stream2(cudaStreamNonBlocking);
  cudaGraph_t graph1 = nullptr;
  cudaGraph_t graph2 = nullptr;
  cudaGraphExec_t exec1 = nullptr;
  cudaGraphExec_t exec2 = nullptr;

  auto capture = [&](CtranComm* comm,
                     void* sendbuf,
                     cudaStream_t stream,
                     cudaGraph_t* graph,
                     cudaGraphExec_t* exec) {
    ASSERT_EQ(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
        cudaSuccess);
    ASSERT_EQ(
        ctranAllGather(
            sendbuf, recv.get(), count, commInt32, comm, stream, algo),
        commSuccess);
    ASSERT_EQ(cudaStreamEndCapture(stream, graph), cudaSuccess);
    ASSERT_EQ(cudaGraphInstantiate(exec, *graph, 0), cudaSuccess);
  };

  capture(comm1.get(), send1.get(), stream1.get(), &graph1, &exec1);
  capture(comm2.get(), send2.get(), stream2.get(), &graph2, &exec2);

  ASSERT_EQ(cudaGraphLaunch(exec1, stream1.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream1.get()), cudaSuccess);
  verifyAllGather(recv.get(), count, nRanks);

  ASSERT_EQ(cudaGraphExecDestroy(exec1), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph1), cudaSuccess);
  comm1->cudagraphDeferredCleanup.runAll();
  waitAndVerifyGpeClean(comm1.get());

  ASSERT_EQ(
      cudaMemsetAsync(
          recv.get(), 0, count * nRanks * sizeof(int32_t), stream2.get()),
      cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(exec2, stream2.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream2.get()), cudaSuccess);
  verifyAllGather(recv.get(), count, nRanks);

  ASSERT_EQ(cudaGraphExecDestroy(exec2), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph2), cudaSuccess);
  comm2->cudagraphDeferredCleanup.runAll();
  waitAndVerifyGpeClean(comm2.get());
}

INSTANTIATE_TEST_SUITE_P(
    CudaGraphAllGatherCtgraph,
    CudaGraphAllGatherCtgraphSharedRecvbuf,
    ::testing::Values(
        NCCL_ALLGATHER_ALGO::ctgraph_pipeline,
        NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline),
    [](const auto& info) { return allGatherAlgoName(info.param); });

// Multiple persistent AGP requests over the same recvbuf coexist in one
// captured graph: each ctranAllGather builds its own persistent request
// (setupPersistentRequest) whose NVL IPC imports refcount up on the shared
// recvbuf. Capturing three back-to-back allgathers on the same recvbuf in a
// single graph exercises this coexistence, and the CTRAN-AGP log shows the
// IpcImport refCount rising 1->2->3 across the three requests.
class CudaGraphAllGatherCtgraphMultiAgpSameRecvbuf
    : public CtranCudaGraphTestBase,
      public ::testing::WithParamInterface<enum NCCL_ALLGATHER_ALGO> {};

TEST_P(
    CudaGraphAllGatherCtgraphMultiAgpSameRecvbuf,
    MultipleAgpReuseRecvbufInOneGraph) {
  const auto algo = GetParam();
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  if (!ctran::allGatherPSupport(comm.get())) {
    GTEST_SKIP() << "allGatherP not supported";
  }

  const auto statex = comm->statex_.get();
  if (statex->nLocalRanks() < 2) {
    GTEST_SKIP() << "Need local NVL peers for pipeline ctgraph AGP";
  }
  if (algo == NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline &&
      !ctran::utils::isPowerOfTwo(statex->nNodes())) {
    GTEST_SKIP() << allGatherAlgoName(algo)
                 << " requires nNodes to be a power of 2";
  }

  const size_t count = kDefaultCount;
  const int nRanks = numRanks;
  ctran::TestDeviceBuffer send(count * sizeof(int32_t));
  fillSendBuf(send.get(), count, globalRank);

  ctran::TestDeviceBuffer recv(count * nRanks * sizeof(int32_t));
  ScopedRecvBufReg recvReg(recv.get(), count * nRanks * sizeof(int32_t));

  meta::comms::CudaStream stream(cudaStreamNonBlocking);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  constexpr int kNumAgp = 3;

  ASSERT_EQ(
      cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal),
      cudaSuccess);
  for (int i = 0; i < kNumAgp; ++i) {
    ASSERT_EQ(
        ctranAllGather(
            send.get(),
            recv.get(),
            count,
            commInt32,
            comm.get(),
            stream.get(),
            algo),
        commSuccess);
  }
  ASSERT_EQ(cudaStreamEndCapture(stream.get(), &graph), cudaSuccess);
  ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);

  // All kNumAgp AllGathers captured into this graph reuse the same recvbuf,
  // so their intra-node NVL peer imports dedup in IpcRegCache and the shared
  // refCount rises above 1. Imports live until cudaGraphDestroy below.
  const auto ipcRegCache = ctran::IpcRegCache::getInstance();
  ASSERT_NE(ipcRegCache, nullptr);
  EXPECT_GE(ipcRegCache->maxRemRegRefCount(), 2)
      << "expected reused NVL IPC import refCount > 1 across " << kNumAgp
      << " same-recvbuf AllGathers captured in one graph";

  ASSERT_EQ(cudaGraphLaunch(exec, stream.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  verifyAllGather(recv.get(), count, nRanks);

  ASSERT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  comm->cudagraphDeferredCleanup.runAll();
  waitAndVerifyGpeClean(comm.get());
}

INSTANTIATE_TEST_SUITE_P(
    CudaGraphAllGatherCtgraph,
    CudaGraphAllGatherCtgraphMultiAgpSameRecvbuf,
    ::testing::Values(
        NCCL_ALLGATHER_ALGO::ctgraph_pipeline,
        NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline),
    [](const auto& info) { return allGatherAlgoName(info.param); });

// Failure path: graph-capture ctgraph pipeline AGP registers the recvbuff via
// RegCache::acquireScopedRegister, which requires the recvbuff segment to be
// allocator-cached (CCA hook). A raw cudaMalloc buffer that was never cached
// must make capture fail with a clean error (commInvalidUsage surfaced through
// ctranAllGather) rather than crash or silently succeed. The
// createPersistentRequest seam is file-local to AllGatherCudagraphAware.cc and
// cannot be unit-tested in isolation, so this exercises the closest feasible
// graph/algo seam.
TEST_F(CtranCudaGraphTestBase, CtgraphPipelineUncachedRecvbufFailsCleanly) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  if (!ctran::allGatherPSupport(comm.get())) {
    GTEST_SKIP() << "allGatherP not supported";
  }
  const auto statex = comm->statex_.get();
  if (statex->nLocalRanks() < 2) {
    GTEST_SKIP() << "Need local NVL peers for pipeline ctgraph AGP";
  }

  const size_t count = kDefaultCount;
  const int nRanks = numRanks;
  ctran::TestDeviceBuffer send(count * sizeof(int32_t));
  fillSendBuf(send.get(), count, globalRank);

  // Raw cudaMalloc recvbuff: its segment is NOT registered in the regcache
  // allocator cache, so acquireScopedRegister must reject it.
  void* recvbuf = nullptr;
  ASSERT_EQ(
      cudaMalloc(&recvbuf, count * nRanks * sizeof(int32_t)), cudaSuccess);

  meta::comms::CudaStream stream(cudaStreamNonBlocking);
  ASSERT_EQ(
      cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal),
      cudaSuccess);
  const auto rc = ctranAllGather(
      send.get(),
      recvbuf,
      count,
      commInt32,
      comm.get(),
      stream.get(),
      NCCL_ALLGATHER_ALGO::ctgraph_pipeline);
  EXPECT_EQ(rc, commInvalidUsage)
      << "Uncached recvbuf must fail ctgraph pipeline capture with "
         "commInvalidUsage from acquireScopedRegister";

  // Tear the capture down regardless of whether an incomplete graph was left
  // on the stream, then confirm the device is still in a clean state (no crash,
  // no leaked capture).
  cudaGraph_t captured = nullptr;
  cudaStreamEndCapture(stream.get(), &captured);
  if (captured != nullptr) {
    cudaGraphDestroy(captured);
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  ASSERT_EQ(cudaFree(recvbuf), cudaSuccess);
}

// Reproduces the ctgraph_rdpipeline in-place corruption covered by
// nccl-tests-suite `-F full_replay_check`: many distinct-slot AllGather calls
// captured in one CUDA graph, then replayed several times on a 2-rank local
// setup.
TEST_F(
    CtranCudaGraphTestBase,
    DISABLED_CtgraphRdpipelineInPlaceMultiAgReplayPreservesAllBytes) {
  setenv("NCCL_CTRAN_ENABLE", "1", 0);
  setenv("NCCL_MNNVL_ENABLE", "1", 0);
  setenv("NCCL_NVLS_ENABLE", "0", 0);
  setenv("NCCL_P2P_DISABLE", "0", 0);
  setenv("NCCL_NET_GDR_C2C", "1", 0);
  setenv("NCCL_CTRAN_IB_DEVICES_PER_RANK", "1", 0);
  setenv("NCCL_IB_HCA", "mlx5_0:1,mlx5_1:1", 0);
  setenv("NCCL_IB_DISABLE", "0", 0);
  setenv("NCCL_IB_GID_INDEX", "3", 0);
  setenv("NCCL_SOCKET_IFNAME", "eth0", 0);
  setenv("NCCL_CUMEM_ENABLE", "1", 1);
  setenv("NCCL_ALLGATHER_ALGO", "ctgraph_rdpipeline", 1);

  auto ctranCommHolder = makeCtranComm();
  ASSERT_NE(ctranCommHolder, nullptr);

  if (!ctran::allGatherPSupport(ctranCommHolder.get())) {
    GTEST_SKIP() << "allGatherP not supported";
  }
  const auto statex = ctranCommHolder->statex_.get();
  if (statex->nLocalRanks() < 2) {
    GTEST_SKIP() << "Need at least 2 local ranks (NVL peers) to repro";
  }
  if (statex->nNodes() > 1 &&
      (statex->nNodes() & (statex->nNodes() - 1)) != 0) {
    GTEST_SKIP() << "ctgraph_rdpipeline requires nNodes to be a power of 2";
  }

  // Use the public NCCL API path and broadcast the ncclUniqueId through the
  // test bootstrap.
  ncclUniqueId ncclId;
  if (globalRank == 0) {
    NCCLCHECK_TEST_BENCH(ncclGetUniqueId(&ncclId));
  }
  std::vector<char> idBuf(numRanks * sizeof(ncclId));
  std::memcpy(
      idBuf.data() + globalRank * sizeof(ncclId), &ncclId, sizeof(ncclId));
  {
    auto rc =
        ctranCommHolder->bootstrap_
            ->allGather(idBuf.data(), sizeof(ncclId), globalRank, numRanks)
            .get();
    ASSERT_EQ(rc, 0) << "Bootstrap allGather for ncclUniqueId failed";
  }
  std::memcpy(&ncclId, idBuf.data(), sizeof(ncclId));
  CUDACHECK_TEST(cudaSetDevice(localRank));
  ncclComm_t ncclComm = nullptr;
  ncclConfig_t ncclCfg = NCCL_CONFIG_INITIALIZER;
  ncclCfg.commDesc = "nccl-tests-suite-benchmarking-comms";
  NCCLCHECK_TEST_BENCH(ncclGroupStart());
  NCCLCHECK_TEST_BENCH(ncclCommInitRankConfig(
      &ncclComm, numRanks, ncclId, globalRank, &ncclCfg));
  NCCLCHECK_TEST_BENCH(ncclGroupEnd());

  ctranCommHolder.reset();

  const int nRanks = numRanks;
  constexpr size_t kIters = 50;
  constexpr size_t kReplays = 5;
  constexpr size_t typeBytes = 2;
  constexpr size_t sendcount = 8UL * 1024 * 1024;
  constexpr size_t sendBytesPerSlot = sendcount * typeBytes;
  const size_t recvBytesPerSlot = sendcount * nRanks * typeBytes;

  auto fillBf16 = [](void* buf, size_t count, uint16_t val) {
    std::vector<uint16_t> h(count, val);
    CUDACHECK_TEST(
        cudaMemcpy(buf, h.data(), count * sizeof(uint16_t), cudaMemcpyDefault));
  };
  auto rankVal = [](int rank) -> uint16_t {
    return static_cast<uint16_t>(0x4000 + rank);
  };
  auto slotVal = [](size_t slot, int rank) -> uint16_t {
    return static_cast<uint16_t>((slot & 0xff) << 8) |
        static_cast<uint16_t>(0x40 + rank);
  };

  meta::comms::CudaStream stream(cudaStreamNonBlocking);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Mirror the production sequence: perf OOP, datacheck OOP, perf IP, then
  // datacheck IP. The datacheck phases use distinct slots and verify bytes.
  auto doCapturedRun = [&](bool oop,
                           bool sameBuffer,
                           size_t* outBadSlots = nullptr) {
    if (outBadSlots) {
      *outBadSlots = 0;
    }
    if (sameBuffer) {
      ctran::TestDeviceBuffer perfSend(sendBytesPerSlot);
      ctran::TestDeviceBuffer perfRecv(recvBytesPerSlot);
      char* sb = static_cast<char*>(perfSend.get());
      char* rb = static_cast<char*>(perfRecv.get());
      fillBf16(sb, sendcount, rankVal(globalRank));
      CUDACHECK_TEST(cudaMemset(rb, 0, recvBytesPerSlot));
      auto* myChunkRb = rb + globalRank * sendBytesPerSlot;
      fillBf16(myChunkRb, sendcount, rankVal(globalRank));
      CUDACHECK_TEST(cudaDeviceSynchronize());
      cudaGraph_t g = nullptr;
      cudaGraphExec_t ge = nullptr;
      ASSERT_EQ(
          cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal),
          cudaSuccess);
      for (size_t i = 0; i < kIters; ++i) {
        void* aghSrc =
            oop ? static_cast<void*>(sb) : static_cast<void*>(myChunkRb);
        NCCLCHECK_TEST_BENCH(ncclAllGather(
            aghSrc, rb, sendcount, ncclBfloat16, ncclComm, stream.get()));
      }
      ASSERT_EQ(cudaStreamEndCapture(stream.get(), &g), cudaSuccess);
      ASSERT_EQ(cudaGraphInstantiate(&ge, g, 0), cudaSuccess);
      for (int r = 0; r < kReplays; ++r) {
        ASSERT_EQ(cudaGraphLaunch(ge, stream.get()), cudaSuccess);
      }
      ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
      ASSERT_EQ(cudaGraphExecDestroy(ge), cudaSuccess);
      ASSERT_EQ(cudaGraphDestroy(g), cudaSuccess);
      CUDACHECK_TEST(cudaDeviceSynchronize());
      return;
    }
    {
      const size_t bigDcSendBytes = kIters * sendBytesPerSlot;
      const size_t bigDcRecvBytes = kIters * recvBytesPerSlot;
      ctran::TestDeviceBuffer bigDcSend(bigDcSendBytes);
      ctran::TestDeviceBuffer bigDcRecv(bigDcRecvBytes);
      CUDACHECK_TEST(cudaMemset(bigDcRecv.get(), 0, bigDcRecvBytes));
      for (size_t k = 0; k < kIters; ++k) {
        auto* slotSend =
            static_cast<char*>(bigDcSend.get()) + k * sendBytesPerSlot;
        auto* slotRecv =
            static_cast<char*>(bigDcRecv.get()) + k * recvBytesPerSlot;
        fillBf16(slotSend, sendcount, slotVal(k, globalRank));
        auto* slotMyChunk = slotRecv + globalRank * sendBytesPerSlot;
        fillBf16(slotMyChunk, sendcount, slotVal(k, globalRank));
      }
      CUDACHECK_TEST(cudaDeviceSynchronize());
      cudaGraph_t g = nullptr;
      cudaGraphExec_t ge = nullptr;
      ASSERT_EQ(
          cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal),
          cudaSuccess);
      for (size_t k = 0; k < kIters; ++k) {
        auto* slotSend =
            static_cast<char*>(bigDcSend.get()) + k * sendBytesPerSlot;
        auto* slotRecv =
            static_cast<char*>(bigDcRecv.get()) + k * recvBytesPerSlot;
        auto* slotMyChunk = slotRecv + globalRank * sendBytesPerSlot;
        void* aghSrc = oop ? static_cast<void*>(slotSend)
                           : static_cast<void*>(slotMyChunk);
        NCCLCHECK_TEST_BENCH(ncclAllGather(
            aghSrc, slotRecv, sendcount, ncclBfloat16, ncclComm, stream.get()));
      }
      ASSERT_EQ(cudaStreamEndCapture(stream.get(), &g), cudaSuccess);
      ASSERT_EQ(cudaGraphInstantiate(&ge, g, 0), cudaSuccess);
      for (int r = 0; r < kReplays; ++r) {
        ASSERT_EQ(cudaGraphLaunch(ge, stream.get()), cudaSuccess);
      }
      ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
      ASSERT_EQ(cudaGraphExecDestroy(ge), cudaSuccess);
      ASSERT_EQ(cudaGraphDestroy(g), cudaSuccess);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      size_t badSlots = 0;
      for (size_t k = 0; k < kIters; ++k) {
        std::vector<uint16_t> h(sendcount * nRanks);
        auto* slotRecv =
            static_cast<char*>(bigDcRecv.get()) + k * recvBytesPerSlot;
        CUDACHECK_TEST(cudaMemcpy(
            h.data(), slotRecv, recvBytesPerSlot, cudaMemcpyDefault));
        size_t mm = 0;
        for (int r = 0; r < nRanks; ++r) {
          uint16_t expected = slotVal(k, r);
          for (size_t i = 0; i < sendcount; ++i) {
            if (h[r * sendcount + i] != expected) {
              ++mm;
            }
          }
        }
        if (mm > 0) {
          ++badSlots;
        }
      }
      if (outBadSlots) {
        *outBadSlots = badSlots;
      }
    }
  };

  constexpr int kTotalChecks = 10;
  size_t mmDcOOP = 0;
  size_t mmDcIP = 0;
  doCapturedRun(/*oop=*/true, /*sameBuffer=*/true);
  for (int t = 0; t < kTotalChecks; ++t) {
    size_t mm = 0;
    doCapturedRun(/*oop=*/true, /*sameBuffer=*/false, &mm);
    mmDcOOP += mm;
  }
  doCapturedRun(/*oop=*/false, /*sameBuffer=*/true);
  for (int t = 0; t < kTotalChecks; ++t) {
    size_t mm = 0;
    doCapturedRun(/*oop=*/false, /*sameBuffer=*/false, &mm);
    mmDcIP += mm;
  }

  EXPECT_EQ(mmDcIP, 0u)
      << "ctgraph_rdpipeline in-place datacheck capture corrupted " << mmDcIP
      << " slots' recvbufs across " << kTotalChecks << " iters";
  EXPECT_EQ(mmDcOOP, 0u)
      << "ctgraph_rdpipeline out-of-place datacheck warmup corrupted "
      << mmDcOOP << " slots' recvbufs across " << kTotalChecks << " iters";

  ncclCommDestroy(ncclComm);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranCudaGraphEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
