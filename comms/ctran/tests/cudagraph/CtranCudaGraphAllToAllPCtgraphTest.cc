// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Tests for the ctgraph AllToAll algorithm (AllToAllP). The ctgraph algo is
// only active during CUDA graph capture: it transparently dispatches to the
// persistent window-based AllToAllP path. Outside capture it falls back to
// the baseline ctran algo (ctranAllToAllSupport returns false).

#include <cstdint>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllToAll/AllToAllImpl.h"
#include "comms/ctran/algos/AllToAll/AllToAllvImpl.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/tests/cudagraph/CtranCudaGraphParamTest.h"

// AllToAll transpose fill: sendbuf slot j on rank i encodes both the source
// rank i and the destination slot j, so a correct transpose is verifiable:
// after AllToAll, rank i's recvbuf slot j must equal what rank j sent to i =
// rank j's sendbuf slot i. Using i * kRankStride + j keeps every (rank, slot)
// pair distinct for the topologies under test.
static constexpr int32_t kRankStride = 1 << 16;

static int32_t alltoallSendValue(int rank, int slot) {
  return rank * kRankStride + slot;
}

// Fills sendbuf slot j (one chunk per peer) with alltoallSendValue(rank, j),
// encoding both the source rank and the destination slot so the transpose is
// verifiable.
static void fillAllToAllSendBuf(void* sendbuf, size_t count, int rank, int nR) {
  std::vector<int32_t> host(count * nR);
  for (int slot = 0; slot < nR; ++slot) {
    const int32_t val = alltoallSendValue(rank, slot);
    for (size_t i = 0; i < count; ++i) {
      host[slot * count + i] = val;
    }
  }
  CUDACHECK_TEST(cudaMemcpy(
      sendbuf, host.data(), count * nR * sizeof(int32_t), cudaMemcpyDefault));
}

// Caches the recvbuf's segment in the regcache allocator cache for the
// lifetime of this object, mirroring the production CCA memory hook. The
// graph AllToAllP path's acquireScopedRegister requires the recvbuf segment
// to be cached. Deregistration in the destructor MUST run after the capturing
// graph/request is destroyed (the scoped registration ref is held until then),
// which callers guarantee by declaring the guard so it outlives the graph.
class ScopedRecvBufReg {
 public:
  ScopedRecvBufReg(void* buf, size_t bytes) : buf_(buf), bytes_(bytes) {
    // Fail fast: an uncached recvbuf makes the graph AllToAllP path's
    // acquireScopedRegister fail later with a confusing error.
    EXPECT_EQ(
        ctran::RegCache::getInstance()->globalRegister(
            buf_, bytes_, /*forceReg=*/true),
        commSuccess)
        << "ScopedRecvBufReg: globalRegister failed for buf " << buf_;
  }
  ~ScopedRecvBufReg() {
    // Best-effort: cannot assert in a destructor (may run during unwinding), so
    // surface a deregistration failure via a warning instead of swallowing it.
    if (buf_ != nullptr &&
        ctran::RegCache::getInstance()->globalDeregister(buf_, bytes_) !=
            commSuccess) {
      GTEST_LOG_(WARNING)
          << "ScopedRecvBufReg: globalDeregister failed for buf " << buf_;
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

static AlgoDescriptor makeAllToAllPCtgraph(
    enum NCCL_ALLTOALL_ALGO algo = NCCL_ALLTOALL_ALGO::ctgraph) {
  struct B : AlgoDescriptor::Buffers {
    ctran::TestDeviceBuffer send, recv;
    size_t bytes;
    ScopedRecvBufReg recvReg;
    B(size_t c, int rank, int nR)
        : send(c * nR * sizeof(int32_t)),
          recv(c * nR * sizeof(int32_t)),
          bytes(c * nR * sizeof(int32_t)),
          recvReg(recv.get(), c * nR * sizeof(int32_t)) {
      fillAllToAllSendBuf(send.get(), c, rank, nR);
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
  desc.name = allToAllAlgoName(algo);
  desc.isSupported = [algo](CtranComm* comm, size_t count, int) {
    // ctgraph support is gated behind an active capture. Probe with a
    // capturing stream so ctranAllToAllSupport reports the true capability.
    meta::comms::CudaStream probeStream(cudaStreamNonBlocking);
    cudaGraph_t probeGraph = nullptr;
    if (cudaStreamBeginCapture(
            probeStream.get(), cudaStreamCaptureModeGlobal) != cudaSuccess) {
      return false;
    }
    const bool supported =
        ctranAllToAllSupport(count, commInt32, comm, algo, probeStream.get());
    cudaStreamEndCapture(probeStream.get(), &probeGraph);
    if (probeGraph != nullptr) {
      cudaGraphDestroy(probeGraph);
    }
    return supported;
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
    // Use the ctgraph algo during active capture, plain ctran for eager warmup.
    cudaStreamCaptureStatus status;
    ASSERT_EQ(cudaStreamIsCapturing(ctx.stream, &status), cudaSuccess);
    const auto resolvedAlgo = (status == cudaStreamCaptureStatusActive)
        ? algo
        : NCCL_ALLTOALL_ALGO::ctran;
    ASSERT_EQ(
        ctranAllToAll(
            b->sendbuf(),
            b->recvbuf(),
            count,
            commInt32,
            ctx.comm,
            ctx.stream,
            resolvedAlgo),
        commSuccess);
  };
  return desc;
}

DEFINE_CUDAGRAPH_PARAM_TEST(CudaGraphAllToAllPCtgraph, makeAllToAllPCtgraph());

// Transpose check for a given rank: recvbuf slot peer (elements
// [peer*count, (peer+1)*count)) must equal peer's sendbuf slot myRank =
// alltoallSendValue(peer, myRank).
static void verifyAllToAllTranspose(
    const void* recvbuf,
    size_t count,
    int myRank,
    int nRanks) {
  const size_t totalCount = count * nRanks;
  std::vector<int32_t> host(totalCount);
  CUDACHECK_TEST(cudaMemcpy(
      host.data(), recvbuf, totalCount * sizeof(int32_t), cudaMemcpyDefault));
  for (int peer = 0; peer < nRanks; ++peer) {
    const int32_t expected = alltoallSendValue(peer, myRank);
    for (size_t i = 0; i < count; ++i) {
      ASSERT_EQ(host[peer * count + i], expected)
          << "AllToAll transpose mismatch from peer " << peer << " index " << i;
    }
  }
}

// Verifies that graph destruction cleans up without CUDA API errors. The
// retainUserObject destructor callback defers cleanup to comm destruction
// since CUDA APIs are forbidden in the callback context.
class CudaGraphAllToAllPCtgraphDestroy : public CtranCudaGraphTestBase {};

TEST_F(CudaGraphAllToAllPCtgraphDestroy, DestroyGraphCleanly) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  const size_t count = kDefaultCount;
  const int nRanks = numRanks;

  meta::comms::CudaStream stream(cudaStreamNonBlocking);
  ctran::TestDeviceBuffer send(count * nRanks * sizeof(int32_t));
  ctran::TestDeviceBuffer recv(count * nRanks * sizeof(int32_t));
  fillAllToAllSendBuf(send.get(), count, globalRank, nRanks);

  ScopedRecvBufReg recvReg(recv.get(), count * nRanks * sizeof(int32_t));

  // Capture
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  ASSERT_EQ(
      cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal),
      cudaSuccess);
  if (!ctranAllToAllSupport(
          count,
          commInt32,
          comm.get(),
          NCCL_ALLTOALL_ALGO::ctgraph,
          stream.get())) {
    cudaGraph_t skipped = nullptr;
    cudaStreamEndCapture(stream.get(), &skipped);
    if (skipped != nullptr) {
      cudaGraphDestroy(skipped);
    }
    GTEST_SKIP() << "AllToAllP ctgraph not supported";
  }
  ASSERT_EQ(
      ctranAllToAll(
          send.get(),
          recv.get(),
          count,
          commInt32,
          comm.get(),
          stream.get(),
          NCCL_ALLTOALL_ALGO::ctgraph),
      commSuccess);
  ASSERT_EQ(cudaStreamEndCapture(stream.get(), &graph), cudaSuccess);
  ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);

  // Replay
  ASSERT_EQ(cudaGraphLaunch(exec, stream.get()), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);

  // Verify the transpose landed correctly before tearing down.
  verifyAllToAllTranspose(recv.get(), count, globalRank, nRanks);

  // Destroy — triggers retainUserObject destructor callback.
  ASSERT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

// Failure path: graph-capture ctgraph AllToAllP registers the recvbuff via
// RegCache::acquireScopedRegister, which requires the recvbuff segment to be
// allocator-cached (CCA hook). A raw cudaMalloc buffer that was never cached
// must make capture fail with a clean error (commInvalidUsage surfaced through
// ctranAllToAll) rather than crash or silently succeed.
TEST_F(CtranCudaGraphTestBase, CtgraphUncachedRecvbufFailsCleanly) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  const size_t count = kDefaultCount;
  const int nRanks = numRanks;
  ctran::TestDeviceBuffer send(count * nRanks * sizeof(int32_t));
  fillAllToAllSendBuf(send.get(), count, globalRank, nRanks);

  // Raw cudaMalloc recvbuff: its segment is NOT registered in the regcache
  // allocator cache, so acquireScopedRegister must reject it.
  void* recvbuf = nullptr;
  ASSERT_EQ(
      cudaMalloc(&recvbuf, count * nRanks * sizeof(int32_t)), cudaSuccess);

  meta::comms::CudaStream stream(cudaStreamNonBlocking);
  ASSERT_EQ(
      cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal),
      cudaSuccess);
  if (!ctranAllToAllSupport(
          count,
          commInt32,
          comm.get(),
          NCCL_ALLTOALL_ALGO::ctgraph,
          stream.get())) {
    cudaGraph_t skipped = nullptr;
    cudaStreamEndCapture(stream.get(), &skipped);
    if (skipped != nullptr) {
      cudaGraphDestroy(skipped);
    }
    ASSERT_EQ(cudaFree(recvbuf), cudaSuccess);
    GTEST_SKIP() << "AllToAllP ctgraph not supported";
  }
  const auto rc = ctranAllToAll(
      send.get(),
      recvbuf,
      count,
      commInt32,
      comm.get(),
      stream.get(),
      NCCL_ALLTOALL_ALGO::ctgraph);
  EXPECT_EQ(rc, commInvalidUsage)
      << "Uncached recvbuf must fail ctgraph AllToAllP capture with "
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

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranCudaGraphEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
