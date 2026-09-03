// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * HIP-graph capture coverage for the sharded-relay collectives.
 *
 * The relay had no graph-capture coverage at all, which made four separate
 * graph hazards unverifiable:
 *
 *   G1 ScratchBufferCache uses hipMallocAsync/hipFreeAsync keyed on
 *      (device, stream, key). Capturing turns the allocation into a graph
 *      allocation node whose address is only valid while that graph executes,
 *      and a later cache growth can hipFreeAsync a pointer a live graph still
 *      references.
 *   G2 The one-shot IPC handshake advances a host-side epoch that is passed to
 *      the kernel by value, so a replay would spin on a baked epoch that every
 *      flag already satisfies and reduce capture-time staging.
 *   G3 One-shot region creation does a synchronous hipMemset plus two
 *      bootstrapAllGather calls, none of which are capturable.
 *   G4 The reduce-scatter overlap side stream is forked from the caller stream,
 *      which has no defined relationship to its captured form.
 *
 * G2, G3 and G4 currently sit behind explicit capture bail-outs, so the cases
 * covering them pass as-is and are the regression guard their fixes get
 * measured against. The two ShardedRelayGraphCaptureScratchHazardTest cases are
 * the exception: they fail without a graph-aware scratch cache, and landed with
 * the fix for it.
 *
 * One ordering constraint worth knowing before adding a case:
 * ScratchBufferCache is a process-global keyed on (device, stream, key), so a
 * case that runs after a larger one finds an entry big enough to satisfy it and
 * never allocates under capture. Anything meant to exercise a cold cache has to
 * run first, on its own stream.
 *
 * Every multi-replay case varies its input per replay and skews odd ranks with
 * a host sleep before the launch. The skew is what makes a stale-epoch or
 * stale-staging bug deterministic rather than a race: the unskewed rank pushes,
 * flags, falls through the handshake and reduces data the skewed rank has not
 * written yet.
 *
 * Capture uses hipStreamCaptureModeRelaxed to match RCCL's own allocation path
 * (enqueue.cc), so these tests measure data-path correctness rather than
 * capture-mode policy.
 */

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "bootstrap.h"
#include "comm.h"
#include "comms/rcclx/develop/meta/testinfra/TestUtils.h"
#include "comms/rcclx/develop/meta/testinfra/TestsDistUtils.h"
#include "meta/relay/sharded_relay_lp.h"
#include "meta/relay/sharded_relay_lp_arena.h"
#include "meta/relay/sharded_relay_oneshot.h"
#include "nccl.h"

#define HIPCHECK_TEST(cmd)                                          \
  do {                                                              \
    hipError_t error = cmd;                                         \
    if (error != hipSuccess) {                                      \
      FAIL() << "HIP error: " << hipGetErrorString(error) << " at " \
             << __FILE__ << ":" << __LINE__;                        \
    }                                                               \
  } while (0)

#define HIPEXPECT_TEST(cmd)                                                \
  do {                                                                     \
    hipError_t error = cmd;                                                \
    if (error != hipSuccess) {                                             \
      ADD_FAILURE() << "HIP error: " << hipGetErrorString(error) << " at " \
                    << __FILE__ << ":" << __LINE__;                        \
    }                                                                      \
  } while (0)

#define NCCLCHECK_TEST(cmd)                                            \
  do {                                                                 \
    ncclResult_t result = cmd;                                         \
    if (result != ncclSuccess) {                                       \
      FAIL() << "NCCL error: " << ncclGetErrorString(result) << " at " \
             << __FILE__ << ":" << __LINE__;                           \
    }                                                                  \
  } while (0)

namespace {

constexpr int kActive = 4;
constexpr int kGroups = 1;
constexpr int kSkewMs = 20;
// A graph-compat regression shows up as a kernel that never retires. Cap the
// wait so the case reports a failure instead of burning the harness timeout
// and taking every later case down with it.
constexpr int kSyncTimeoutSec = 20;

// Element count whose single-group A=4 footprint stays inside
// kRelayOneShotMaxBytes (1 MiB): 4 * 16384 * 4 B = 256 KiB.
constexpr size_t kOneShotBandCount = 16 * 1024;
// Comfortably above the one-shot band, below the side-stream threshold.
constexpr size_t kMidCount = 1024 * 1024;
// 4 * (1 << 24) * 4 B = 256 MiB, exactly kRelayOverlapReduceMinBytes, so the
// reduce-scatter overlap side stream is eligible.
constexpr size_t kSideStreamCount = 1ULL << 24;

// Uncaptured sizes below the one-shot band take the one-shot path, so warming
// up at the size under test leaves the PureDirect scratch cache cold and the
// capture then allocates inside itself. Priming at this larger size fills that
// cache first, so the cases below measure the hazard they name rather than G1.
constexpr size_t kScratchPrimeCount = kMidCount;

// Value active rank `rank` contributes for block `block` on `replay`. Varying
// with the replay is what makes a graph that re-reduces capture-time staging
// detectable; varying with the block catches a mis-routed slice.
int32_t fillValue(int rank, int block, int replay) {
  return (rank + 1) * 100 + (block + 1) * 10 + (replay + 1);
}

int32_t expectedReduceScatter(int myActiveIndex, int replay) {
  int32_t sum = 0;
  for (int r = 0; r < kActive; r++) {
    sum += fillValue(r, myActiveIndex, replay);
  }
  return sum;
}

int32_t expectedAllReduce(int replay) {
  int32_t sum = 0;
  for (int r = 0; r < kActive; r++) {
    sum += fillValue(r, 0, replay);
  }
  return sum;
}

} // namespace

class ShardedRelayGraphCaptureTest : public ::testing::Test {
 protected:
  // The comm lives for the whole binary instead of being rebuilt per case. It
  // used to be created in SetUp and destroyed in TearDown, so every case freed
  // everything an 8-rank comm owns. On MI350 freeing VRAM makes amdgpu wipe it
  // (amdgpu_bo_release_notify -> amdgpu_fill_buffer) while holding mmap_lock
  // for write, so 8 ranks cycling multi-GB comms serialise into a stall that
  // takes the whole host down. Reusing also matches how comms are really used:
  // a handful, kept for the life of the process. Only the comm moves here; the
  // per-case stream and scratch stay in SetUp.
  //
  // Guarded so a derived fixture, which is a separate suite and so runs these
  // hooks again, rebuilds rather than leaks.
  static void SetUpTestSuite() {
    if (comm != nullptr) {
      return;
    }
    int localSize;
    std::tie(localRank, globalRank, numRanks, localSize) =
        getTcpStoreOrMpiInfo();
    const bool isServer = (globalRank == 0);
    if (checkTcpStoreEnv()) {
      server = createTcpStore(isServer);
    } else if (isServer) {
      server = createTcpStore(true);
    }
    comm = createNcclComm(
        globalRank, numRanks, localRank, false, nullptr, server.get());
  }

  static void TearDownTestSuite() {
    if (server && checkTcpStoreEnv()) {
      finalizeNcclComm(globalRank, server.get());
    }
    if (comm != nullptr) {
      NCCLCHECK_TEST(ncclCommDestroy(comm));
      comm = nullptr;
    }
    server.reset();
  }

  void SetUp() override {
    ASSERT_NE(this->comm, nullptr)
        << "suite-scoped comm was not created; SetUpTestSuite did not run";
    HIPCHECK_TEST(hipStreamCreate(&stream));
    this->isActive = this->globalRank < kActive;
    this->myActiveIndex = this->globalRank;
    this->activeRanks[0] = 0;
    this->activeRanks[1] = 1;
    this->activeRanks[2] = 2;
    this->activeRanks[3] = 3;
    this->allActiveRanks[0] = this->activeRanks;
  }

  void TearDown() override {
    HIPEXPECT_TEST(hipStreamDestroy(this->stream));
  }

  // Bounded replacement for hipStreamSynchronize.
  void syncStream(const char* what) {
    if (this->streamWedged) {
      return;
    }
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(kSyncTimeoutSec);
    for (;;) {
      const hipError_t status = hipStreamQuery(this->stream);
      if (status == hipSuccess) {
        return;
      }
      if (status != hipErrorNotReady) {
        ADD_FAILURE() << "R" << this->globalRank << " " << what << ": "
                      << hipGetErrorString(status);
        return;
      }
      if (std::chrono::steady_clock::now() > deadline) {
        ADD_FAILURE() << "R" << this->globalRank << " " << what
                      << ": stream did not drain within " << kSyncTimeoutSec
                      << "s";
        // Everything after this would just time out again on a stream that is
        // never going to drain, so record it and let the case bail out.
        this->streamWedged = true;
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }

  // Host-side barrier. Deliberately NOT an ncclAllReduce: issuing a
  // non-captured collective on this comm between graph replays interleaves with
  // the work descriptors the captured graph replays, and the correctness suites
  // do not exercise that combination. The stream is idle at every call site, so
  // a host barrier is also a stream barrier.
  void barrier() {
    syncStream("stream drain");
    NCCLCHECK_TEST(bootstrapBarrier(
        this->comm->bootstrap,
        this->comm->rank,
        this->comm->nRanks,
        this->barrierTag++));
  }

  // Capture `body` on the fixture stream and instantiate it. Returns nullptr on
  // failure; EndCapture always runs so a failure inside `body` cannot leave the
  // stream stuck in capture mode.
  template <typename Body>
  hipGraphExec_t captureGraph(Body&& body) {
    syncStream("stream drain");
    hipGraph_t graph = nullptr;
    HIPEXPECT_TEST(
        hipStreamBeginCapture(this->stream, hipStreamCaptureModeRelaxed));
    body();
    const hipError_t endErr = hipStreamEndCapture(this->stream, &graph);
    if (endErr != hipSuccess || graph == nullptr) {
      ADD_FAILURE() << "hipStreamEndCapture failed: "
                    << hipGetErrorString(endErr);
      return nullptr;
    }
    hipGraphExec_t exec = nullptr;
    const hipError_t instErr =
        hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    HIPEXPECT_TEST(hipGraphDestroy(graph));
    if (instErr != hipSuccess) {
      ADD_FAILURE() << "hipGraphInstantiate failed: "
                    << hipGetErrorString(instErr);
      return nullptr;
    }
    return exec;
  }

  // Launch a replay, skewing odd ranks so an unskewed peer reaches the
  // handshake before the skewed rank has written anything for this replay.
  void replay(hipGraphExec_t exec, bool skew) {
    if (this->streamWedged) {
      return;
    }
    if (skew && (this->globalRank % 2) == 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kSkewMs));
    }
    HIPEXPECT_TEST(hipGraphLaunch(exec, this->stream));
    syncStream("stream drain");
  }

  void fillUniform(int32_t* deviceBuf, size_t count, int32_t value) {
    const std::vector<int32_t> host(count, value);
    HIPEXPECT_TEST(hipMemcpy(
        deviceBuf,
        host.data(),
        count * sizeof(int32_t),
        hipMemcpyHostToDevice));
  }

  // Fill an A-block buffer so block j holds fillValue(myActiveIndex, j,
  // replay).
  void fillBlocks(int32_t* deviceBuf, size_t blockCount, int replay) {
    std::vector<int32_t> host(static_cast<size_t>(kActive) * blockCount);
    for (int j = 0; j < kActive; j++) {
      std::fill_n(
          host.data() + static_cast<size_t>(j) * blockCount,
          blockCount,
          fillValue(this->myActiveIndex, j, replay));
    }
    HIPEXPECT_TEST(hipMemcpy(
        deviceBuf,
        host.data(),
        host.size() * sizeof(int32_t),
        hipMemcpyHostToDevice));
  }

  void expectUniform(
      const int32_t* deviceBuf,
      size_t count,
      int32_t expected,
      const char* what) {
    std::vector<int32_t> host(count);
    HIPEXPECT_TEST(hipMemcpy(
        host.data(),
        deviceBuf,
        count * sizeof(int32_t),
        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < count; i++) {
      if (host[i] != expected) {
        ADD_FAILURE() << "R" << this->globalRank << " " << what << ": index "
                      << i << " expected " << expected << " got " << host[i];
        return;
      }
    }
  }

  static inline ncclComm_t comm{nullptr};
  hipStream_t stream{};
  static inline int localRank{0};
  static inline int globalRank{0};
  static inline int numRanks{0};
  bool isActive{false};
  int myActiveIndex{0};
  int activeRanks[kActive]{};
  const int* allActiveRanks[kGroups]{};
  // Bootstrap barrier tags must agree across ranks; every rank walks the same
  // call sequence, so a plain counter does.
  int barrierTag{0};
  bool streamWedged{false};
  static inline std::unique_ptr<c10d::TCPStore> server{nullptr};
};

namespace {

// Single-group A=4 reduce-scatter buffers. Helpers hand in an A-block scratch
// and alias their output onto it, matching the helper contract the
// correctness suites use.
struct ReduceScatterBuffers {
  int32_t* send{nullptr};
  int32_t* recv{nullptr};
};

} // namespace

// ---------------------------------------------------------------------------
// Reduce-scatter
// ---------------------------------------------------------------------------

class ShardedRelayGraphCaptureReduceScatterTest
    : public ShardedRelayGraphCaptureTest {
 protected:
  ReduceScatterBuffers allocate(size_t recvCount) {
    ReduceScatterBuffers b;
    const size_t sendBytes =
        static_cast<size_t>(kActive) * recvCount * sizeof(int32_t);
    HIPEXPECT_TEST(hipMalloc(&b.send, sendBytes));
    HIPEXPECT_TEST(hipMemset(b.send, 0, sendBytes));
    if (this->isActive) {
      HIPEXPECT_TEST(hipMalloc(&b.recv, recvCount * sizeof(int32_t)));
      HIPEXPECT_TEST(hipMemset(b.recv, 0, recvCount * sizeof(int32_t)));
    } else {
      b.recv = b.send;
    }
    return b;
  }

  void release(ReduceScatterBuffers& b) {
    if (this->isActive && b.recv != nullptr) {
      HIPEXPECT_TEST(hipFree(b.recv));
    }
    if (b.send != nullptr) {
      HIPEXPECT_TEST(hipFree(b.send));
    }
  }

  ncclResult_t enqueue(const ReduceScatterBuffers& b, size_t recvCount) {
    const void* sendPtrs[kGroups] = {b.send};
    void* recvPtrs[kGroups] = {b.recv};
    const size_t recvCounts[kGroups] = {recvCount};
    return ncclShardedRelayMultiGroupReduceScatter(
        sendPtrs,
        recvPtrs,
        recvCounts,
        ncclInt32,
        ncclSum,
        this->comm,
        this->stream,
        this->allActiveRanks,
        kActive,
        kGroups,
        /*lowPrecision=*/0);
  }

  // Run one uncaptured reduce-scatter and verify it, with a barrier either
  // side.
  void runUncaptured(
      ReduceScatterBuffers& b,
      size_t recvCount,
      int replayIndex,
      const char* what) {
    if (this->isActive) {
      fillBlocks(b.send, recvCount, replayIndex);
    }
    barrier();
    ASSERT_EQ(enqueue(b, recvCount), ncclSuccess);
    syncStream("stream drain");
    if (this->isActive) {
      expectUniform(
          b.recv,
          recvCount,
          expectedReduceScatter(this->myActiveIndex, replayIndex),
          what);
    }
    barrier();
  }

  // Capture one reduce-scatter at `recvCount`, then replay it `replays` times
  // with per-replay-varying input, verifying after each replay.
  void runCaptureReplay(size_t recvCount, int replays, bool warmUp, bool skew) {
    // Size the buffers for the priming call so the same allocation serves both.
    const size_t primeCount = std::max(recvCount, kScratchPrimeCount);
    ReduceScatterBuffers b = allocate(primeCount);
    barrier();

    if (warmUp) {
      if (primeCount != recvCount) {
        runUncaptured(b, primeCount, 0, "reduce-scatter scratch prime");
      }
      runUncaptured(b, recvCount, 0, "reduce-scatter warm-up");
    }

    ncclResult_t captured = ncclSuccess;
    hipGraphExec_t exec =
        captureGraph([&]() { captured = enqueue(b, recvCount); });
    ASSERT_EQ(captured, ncclSuccess);
    ASSERT_NE(exec, nullptr);

    for (int r = 0; r < replays; r++) {
      if (this->isActive) {
        fillBlocks(b.send, recvCount, r + 1);
      }
      barrier();
      replay(exec, skew);
      if (this->isActive) {
        expectUniform(
            b.recv,
            recvCount,
            expectedReduceScatter(this->myActiveIndex, r + 1),
            "reduce-scatter replay");
      }
    }

    HIPEXPECT_TEST(hipGraphExecDestroy(exec));
    release(b);
  }
};

// ---------------------------------------------------------------------------
// Scratch-cache hazards
//
// These are the cases that fail on a relay without a graph-aware scratch cache.
// ScratchBufferCache is a process-global keyed on (device, stream, key), so
// they must run FIRST: a single earlier case at a larger size leaves an entry
// big enough to satisfy them, the capture then allocates nothing, and they
// silently stop testing anything. gtest runs suites in the order their first
// TEST_F appears, which is why this block sits above the others.
// ---------------------------------------------------------------------------

class ShardedRelayGraphCaptureScratchHazardTest
    : public ShardedRelayGraphCaptureReduceScatterTest {
 protected:
  void SetUp() override {
    ShardedRelayGraphCaptureReduceScatterTest::SetUp();
    // Run on a second, still-live stream. Keying the cache on the stream means
    // a distinct live handle cannot collide with an entry left by the fixture
    // stream, which hipStreamCreate is free to hand back after a destroy.
    HIPCHECK_TEST(hipStreamCreate(&hazardStream));
    baseStream = this->stream;
    this->stream = hazardStream;
  }

  void TearDown() override {
    this->stream = baseStream;
    HIPEXPECT_TEST(hipStreamDestroy(hazardStream));
    ShardedRelayGraphCaptureReduceScatterTest::TearDown();
  }

  hipStream_t hazardStream{};
  hipStream_t baseStream{};
};

// G1 mode 1 and G3: the capture is the FIRST relay call on this comm, so the
// scratch cache is cold and the one-shot region has never been created. Any
// hipMallocAsync or region bootstrap the relay does lands inside the capture.
TEST_F(ShardedRelayGraphCaptureScratchHazardTest, CaptureAsFirstRelayCall) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, got " << this->numRanks;
  }
  runCaptureReplay(
      kOneShotBandCount, /*replays=*/2, /*warmUp=*/false, /*skew=*/true);
}

// G1 mode 3: growing the scratch cache AFTER a capture hipFreeAsyncs the
// pointer the graph baked, handing it back to the pool while a live graph still
// references it. The replay must still be correct.
TEST_F(
    ShardedRelayGraphCaptureScratchHazardTest,
    ScratchGrowthAfterCaptureThenReplay) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, got " << this->numRanks;
  }

  const size_t smallCount = 256 * 1024; // 1 MiB out, 4 MiB in
  const size_t largeCount = kSideStreamCount; // 64 MiB out, 256 MiB in

  ReduceScatterBuffers small = allocate(smallCount);
  barrier();

  if (this->isActive) {
    fillBlocks(small.send, smallCount, /*replay=*/0);
  }
  ASSERT_EQ(enqueue(small, smallCount), ncclSuccess);
  syncStream("stream drain");
  barrier();

  ncclResult_t captured = ncclSuccess;
  hipGraphExec_t exec =
      captureGraph([&]() { captured = enqueue(small, smallCount); });
  ASSERT_EQ(captured, ncclSuccess);
  ASSERT_NE(exec, nullptr);

  // Uncaptured call at a much larger size on the same stream. This is what
  // forces the cache to free the entry the graph above captured.
  ReduceScatterBuffers large = allocate(largeCount);
  barrier();
  if (this->isActive) {
    fillBlocks(large.send, largeCount, /*replay=*/0);
  }
  ASSERT_EQ(enqueue(large, largeCount), ncclSuccess);
  syncStream("stream drain");
  release(large);
  barrier();

  for (int r = 0; r < 2; r++) {
    if (this->isActive) {
      fillBlocks(small.send, smallCount, r + 1);
    }
    barrier();
    replay(exec, /*skew=*/true);
    if (this->isActive) {
      expectUniform(
          small.recv,
          smallCount,
          expectedReduceScatter(this->myActiveIndex, r + 1),
          "reduce-scatter replay after scratch growth");
    }
  }

  HIPEXPECT_TEST(hipGraphExecDestroy(exec));
  release(small);
}

// Whether the region exists before any relay call is the whole of G3, and it is
// directly observable, so assert it rather than inferring it from a timing or a
// pass. The eager binary (-DRCCLX_RELAY_TEST_EAGER_ONESHOT, which sets
// NCCL_SHARDED_RELAY_MODE_ENABLE=1 before the first comm is built) must have
// one already; the default binary must not, because that is still the lazy
// path.
TEST_F(ShardedRelayGraphCaptureReduceScatterTest, OneShotRegionReadyAtInit) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, got " << this->numRanks;
  }
#ifdef RCCLX_RELAY_TEST_EAGER_ONESHOT
  EXPECT_TRUE(rcclx::relay::oneShotReady(this->comm));
#else
  EXPECT_FALSE(rcclx::relay::oneShotReady(this->comm));
#endif
}

// Baseline sanity: capture and replay work at all on this build and stream.
TEST_F(ShardedRelayGraphCaptureReduceScatterTest, SmokeCaptureAndReplay) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, got " << this->numRanks;
  }
  runCaptureReplay(kMidCount, /*replays=*/1, /*warmUp=*/true, /*skew=*/false);
}

// Regression guard for G2: sizes inside the one-shot band, replayed with a skew
// that would surface a stale baked epoch.
TEST_F(
    ShardedRelayGraphCaptureReduceScatterTest,
    MultiReplayInOneShotBandWithSkew) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, got " << this->numRanks;
  }
  runCaptureReplay(
      kOneShotBandCount, /*replays=*/3, /*warmUp=*/true, /*skew=*/true);
}

TEST_F(
    ShardedRelayGraphCaptureReduceScatterTest,
    MultiReplayAboveOneShotBandWithSkew) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, got " << this->numRanks;
  }
  runCaptureReplay(kMidCount, /*replays=*/3, /*warmUp=*/true, /*skew=*/true);
}

// Regression guard for G4: at exactly kRelayOverlapReduceMinBytes the owner
// would fork the overlap side stream, which today is suppressed under capture.
TEST_F(
    ShardedRelayGraphCaptureReduceScatterTest,
    MultiReplayAtSideStreamThreshold) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, got " << this->numRanks;
  }
  runCaptureReplay(
      kSideStreamCount, /*replays=*/2, /*warmUp=*/true, /*skew=*/false);
}

// ---------------------------------------------------------------------------
// All-reduce, all-gather, all-to-all
// ---------------------------------------------------------------------------

class ShardedRelayGraphCaptureOtherCollectivesTest
    : public ShardedRelayGraphCaptureTest {};

// All-reduce is in-place; helpers hand in an A-block scratch.
TEST_F(ShardedRelayGraphCaptureOtherCollectivesTest, AllReduceMultiReplay) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, got " << this->numRanks;
  }
  const size_t count = kMidCount;
  const size_t bytes = count * sizeof(int32_t);

  int32_t* buff = nullptr;
  HIPCHECK_TEST(hipMalloc(
      &buff, this->isActive ? bytes : static_cast<size_t>(kActive) * bytes));
  barrier();
  if (!this->isActive) {
    HIPCHECK_TEST(hipMemset(buff, 0, static_cast<size_t>(kActive) * bytes));
  }

  const void* sendPtrs[kGroups] = {buff};
  void* recvPtrs[kGroups] = {buff};
  const size_t counts[kGroups] = {count};
  auto enqueue = [&]() {
    return ncclShardedRelayMultiGroupAllReduce(
        sendPtrs,
        recvPtrs,
        counts,
        ncclInt32,
        ncclSum,
        this->comm,
        this->stream,
        this->allActiveRanks,
        kActive,
        kGroups,
        /*lowPrecision=*/0);
  };

  if (this->isActive) {
    fillUniform(buff, count, fillValue(this->myActiveIndex, 0, 0));
  }
  ASSERT_EQ(enqueue(), ncclSuccess);
  syncStream("stream drain");
  barrier();

  ncclResult_t captured = ncclSuccess;
  hipGraphExec_t exec = captureGraph([&]() { captured = enqueue(); });
  ASSERT_EQ(captured, ncclSuccess);
  ASSERT_NE(exec, nullptr);

  for (int r = 0; r < 3; r++) {
    if (this->isActive) {
      fillUniform(buff, count, fillValue(this->myActiveIndex, 0, r + 1));
    }
    barrier();
    replay(exec, /*skew=*/true);
    if (this->isActive) {
      expectUniform(buff, count, expectedAllReduce(r + 1), "all-reduce replay");
    }
  }

  HIPEXPECT_TEST(hipGraphExecDestroy(exec));
  HIPEXPECT_TEST(hipFree(buff));
}

// All-gather out-of-place: every rank owns an A-slot output; helpers pass the
// output as their placeholder input.
TEST_F(ShardedRelayGraphCaptureOtherCollectivesTest, AllGatherMultiReplay) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, got " << this->numRanks;
  }
  const size_t sendCount = kMidCount;
  const size_t outCount = static_cast<size_t>(kActive) * sendCount;

  int32_t* recvBuff = nullptr;
  int32_t* sendBuff = nullptr;
  HIPCHECK_TEST(hipMalloc(&recvBuff, outCount * sizeof(int32_t)));
  if (this->isActive) {
    HIPCHECK_TEST(hipMalloc(&sendBuff, sendCount * sizeof(int32_t)));
  }
  barrier();
  HIPCHECK_TEST(hipMemset(recvBuff, 0, outCount * sizeof(int32_t)));

  const void* sendPtrs[kGroups] = {this->isActive ? sendBuff : recvBuff};
  void* recvPtrs[kGroups] = {recvBuff};
  const size_t sendCounts[kGroups] = {sendCount};
  auto enqueue = [&]() {
    return ncclShardedRelayMultiGroupAllGather(
        sendPtrs,
        recvPtrs,
        sendCounts,
        ncclInt32,
        this->comm,
        this->stream,
        this->allActiveRanks,
        kActive,
        kGroups,
        /*lowPrecision=*/0);
  };

  if (this->isActive) {
    fillUniform(sendBuff, sendCount, fillValue(this->myActiveIndex, 0, 0));
  }
  ASSERT_EQ(enqueue(), ncclSuccess);
  syncStream("stream drain");
  barrier();

  ncclResult_t captured = ncclSuccess;
  hipGraphExec_t exec = captureGraph([&]() { captured = enqueue(); });
  ASSERT_EQ(captured, ncclSuccess);
  ASSERT_NE(exec, nullptr);

  for (int r = 0; r < 3; r++) {
    if (this->isActive) {
      fillUniform(
          sendBuff, sendCount, fillValue(this->myActiveIndex, 0, r + 1));
    }
    barrier();
    replay(exec, /*skew=*/true);
    if (this->isActive) {
      for (int slot = 0; slot < kActive; slot++) {
        expectUniform(
            recvBuff + static_cast<size_t>(slot) * sendCount,
            sendCount,
            fillValue(slot, 0, r + 1),
            "all-gather replay");
      }
    }
  }

  HIPEXPECT_TEST(hipGraphExecDestroy(exec));
  HIPEXPECT_TEST(hipFree(recvBuff));
  if (sendBuff != nullptr) {
    HIPEXPECT_TEST(hipFree(sendBuff));
  }
}

// All-to-all: in-place is unsupported, so send and recv are distinct A-segment
// buffers. Helpers get a generously sized A-segment scratch.
TEST_F(ShardedRelayGraphCaptureOtherCollectivesTest, AllToAllMultiReplay) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, got " << this->numRanks;
  }
  const size_t segmentCount = kMidCount;
  const size_t bufCount = static_cast<size_t>(kActive) * segmentCount;

  int32_t* sendBuff = nullptr;
  int32_t* recvBuff = nullptr;
  HIPCHECK_TEST(hipMalloc(&sendBuff, (bufCount + 1) * sizeof(int32_t)));
  if (this->isActive) {
    HIPCHECK_TEST(hipMalloc(&recvBuff, bufCount * sizeof(int32_t)));
  } else {
    recvBuff = sendBuff;
  }
  barrier();
  HIPCHECK_TEST(hipMemset(sendBuff, 0, (bufCount + 1) * sizeof(int32_t)));

  const void* sendPtrs[kGroups] = {sendBuff};
  void* recvPtrs[kGroups] = {recvBuff};
  const size_t segmentCounts[kGroups] = {segmentCount};
  auto enqueue = [&]() {
    return ncclShardedRelayMultiGroupAllToAll(
        sendPtrs,
        recvPtrs,
        segmentCounts,
        ncclInt32,
        this->comm,
        this->stream,
        this->allActiveRanks,
        kActive,
        kGroups,
        /*lowPrecision=*/0);
  };

  if (this->isActive) {
    fillBlocks(sendBuff, segmentCount, /*replay=*/0);
  }
  ASSERT_EQ(enqueue(), ncclSuccess);
  syncStream("stream drain");
  barrier();

  ncclResult_t captured = ncclSuccess;
  hipGraphExec_t exec = captureGraph([&]() { captured = enqueue(); });
  ASSERT_EQ(captured, ncclSuccess);
  ASSERT_NE(exec, nullptr);

  for (int r = 0; r < 3; r++) {
    if (this->isActive) {
      fillBlocks(sendBuff, segmentCount, r + 1);
    }
    barrier();
    replay(exec, /*skew=*/true);
    if (this->isActive) {
      // recvSeg[i] holds the segment source i addressed to me.
      for (int src = 0; src < kActive; src++) {
        expectUniform(
            recvBuff + static_cast<size_t>(src) * segmentCount,
            segmentCount,
            fillValue(src, this->myActiveIndex, r + 1),
            "all-to-all replay");
      }
    }
  }

  HIPEXPECT_TEST(hipGraphExecDestroy(exec));
  if (this->isActive) {
    HIPEXPECT_TEST(hipFree(recvBuff));
  }
  HIPEXPECT_TEST(hipFree(sendBuff));
}

// ===========================================================================
// LOW PRECISION UNDER GRAPH CAPTURE
// ===========================================================================
//
// ONE case, because it has to run in a defined order and this binary shares a
// single communicator across every suite: the arena is per-communicator and
// created once, so "the arena is not up yet" is a state that exists exactly
// once per process. Splitting this into two cases would make the second depend
// on gtest's ordering.
//
// The whole low-precision graph-capture contract in sequence:
//
//   1. arena cold + capture  -> LP DECLINES (GraphCapture), because bringing
//   the
//      arena up runs a bootstrap all-gather and that must never land inside a
//      capture. The captured graph is still valid and still correct -- the
//      decline is a clean fall back to full precision, not a broken graph.
//   2. eager call             -> LP ENGAGES and the arena comes up.
//   3. arena warm + capture   -> LP ENGAGES INSIDE THE CAPTURE.
//   4. replay x3              -> exact every time.
//
// Step 4 is what the arena exists for. Its partition is carved from a fixed
// base at offsets derived only from the counts, so every replay reads and
// writes the same addresses the capture recorded. A ScratchBufferCache-style
// allocation would fail here in three separate documented ways; see
// sharded_relay_lp_arena.h.
class ShardedRelayGraphCaptureLowPrecisionTest
    : public ShardedRelayGraphCaptureTest {
 protected:
  // A multiple of kActive * 128, which is what the flat allreduce needs: it
  // splits its direct region into A per-owner shards, and a shard is only a
  // whole number of wire blocks when the count is a multiple of A * 128. 8 MiB
  // in fp32, comfortably above the low-precision size threshold.
  static constexpr size_t kLpCount = 2ULL * 1024 * 1024;

  static float shardValue(int activeIndex) {
    return static_cast<float>(activeIndex + 1);
  }

  // 1 + 2 + 3 + 4, exactly, in every element.
  static float expectedSum() {
    float sum = 0.0f;
    for (int r = 0; r < kActive; r++) {
      sum += shardValue(r);
    }
    return sum;
  }

  void fillMine(float* buff, size_t count) {
    if (!this->isActive) {
      return;
    }
    const std::vector<float> host(count, shardValue(this->myActiveIndex));
    HIPCHECK_TEST(hipMemcpy(
        buff, host.data(), count * sizeof(float), hipMemcpyHostToDevice));
  }

  void expectSum(float* buff, size_t count, const char* what) {
    if (!this->isActive) {
      return;
    }
    const float want = expectedSum();
    std::vector<float> got(count);
    HIPCHECK_TEST(hipMemcpy(
        got.data(), buff, count * sizeof(float), hipMemcpyDeviceToHost));
    size_t reported = 0;
    for (size_t i = 0; i < count && reported < 8; i++) {
      if (got[i] != want) {
        reported++;
        ADD_FAILURE() << "R" << this->globalRank << ": " << what << " element "
                      << i << ": got " << got[i] << ", want " << want;
      }
    }
  }
};

TEST_F(ShardedRelayGraphCaptureLowPrecisionTest, DeclinesColdThenEngagesWarm) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, got " << this->numRanks;
  }
  const size_t count = kLpCount;
  const size_t bytes = count * sizeof(float);

  float* buff = nullptr;
  HIPCHECK_TEST(hipMalloc(
      &buff, this->isActive ? bytes : static_cast<size_t>(kActive) * bytes));
  barrier();
  if (!this->isActive) {
    HIPCHECK_TEST(hipMemset(buff, 0, static_cast<size_t>(kActive) * bytes));
  }

  const void* sendPtrs[kGroups] = {buff};
  void* recvPtrs[kGroups] = {buff};
  const size_t counts[kGroups] = {count};
  auto enqueueLp = [&]() {
    return ncclShardedRelayMultiGroupAllReduce(
        sendPtrs,
        recvPtrs,
        counts,
        ncclFloat32,
        ncclSum,
        this->comm,
        this->stream,
        this->allActiveRanks,
        kActive,
        kGroups,
        /*lowPrecision=*/1);
  };

  // This case owns the cold-arena state for the whole process. If a future low
  // precision case in this binary runs first, this fires rather than silently
  // testing the warm path twice.
  ASSERT_FALSE(rcclx::relay::lpArenaReady(this->comm))
      << "another low-precision case already brought the arena up; this case "
         "must be the first, because the cold-arena state exists once per "
         "communicator";

  // ---- 1. Cold arena inside a capture: LP must decline, graph must be sound.
  fillMine(buff, count);
  barrier();
  rcclx::relay::lpResetCounters();
  ncclResult_t captured = ncclSuccess;
  hipGraphExec_t coldExec = captureGraph([&]() { captured = enqueueLp(); });
  ASSERT_EQ(captured, ncclSuccess);
  ASSERT_NE(coldExec, nullptr);
  EXPECT_EQ(rcclx::relay::lpEngageCount(), 0u)
      << "low precision must not bootstrap its arena inside a capture";
  EXPECT_GT(
      rcclx::relay::lpDeclineCount(rcclx::relay::LpDecline::GraphCapture), 0u)
      << "the decline must be recorded against GraphCapture, not silent";

  // The declined capture is a full-precision graph and has to work like one.
  for (int r = 0; r < 2; r++) {
    fillMine(buff, count);
    barrier();
    replay(coldExec, /*skew=*/true);
    expectSum(buff, count, "cold-arena (full precision) replay");
  }
  HIPEXPECT_TEST(hipGraphExecDestroy(coldExec));

  // ---- 2. Eager call: LP engages and the arena comes up.
  fillMine(buff, count);
  barrier();
  rcclx::relay::lpResetCounters();
  ASSERT_EQ(enqueueLp(), ncclSuccess);
  syncStream("eager low-precision drain");
  ASSERT_GT(rcclx::relay::lpEngageCount(), 0u)
      << "the eager call must engage, or the arena never came up and the rest "
         "of this case proves nothing";
  expectSum(buff, count, "eager low precision");
  ASSERT_TRUE(rcclx::relay::lpArenaReady(this->comm));
  barrier();

  // ---- 3. Warm arena inside a capture: LP must engage this time.
  fillMine(buff, count);
  barrier();
  rcclx::relay::lpResetCounters();
  hipGraphExec_t warmExec = captureGraph([&]() { captured = enqueueLp(); });
  ASSERT_EQ(captured, ncclSuccess);
  ASSERT_NE(warmExec, nullptr);
  EXPECT_GT(rcclx::relay::lpEngageCount(), 0u)
      << "low precision must engage inside a capture once the arena is up";
  EXPECT_EQ(
      rcclx::relay::lpDeclineCount(rcclx::relay::LpDecline::GraphCapture), 0u);

  // ---- 4. Replay: the arena's fixed partition must give the same addresses
  // every time, so every replay is exact.
  for (int r = 0; r < 3; r++) {
    fillMine(buff, count);
    barrier();
    replay(warmExec, /*skew=*/true);
    expectSum(buff, count, "warm-arena low-precision replay");
  }

  HIPEXPECT_TEST(hipGraphExecDestroy(warmExec));
  HIPEXPECT_TEST(hipFree(buff));
}

int main(int argc, char* argv[]) {
#ifdef RCCLX_RELAY_TEST_EAGER_ONESHOT
  // Has to happen before the first communicator is built, and NCCL caches the
  // parameter on first read, so covering both settings needs two binaries
  // rather than two cases.
  setenv("NCCL_SHARDED_RELAY_MODE_ENABLE", "1", /*overwrite=*/1);
#else
  // This binary is the lazy half of the pair and asserts the region is NOT up
  // at init, so it must not inherit the variable from whoever invoked it.
  unsetenv("NCCL_SHARDED_RELAY_MODE_ENABLE");
#endif
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
