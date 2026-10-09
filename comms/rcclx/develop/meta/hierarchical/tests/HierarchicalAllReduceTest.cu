// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * Hierarchical allreduce on one 8-GPU box emulating several hosts.
 *
 * NCCL_HOSTID gives each group of local ranks its own host hash, so RCCL sees
 * HIER_TEST_NODES nodes and carries inter-"node" traffic over the Socket
 * transport on loopback, while intra-node traffic stays on xGMI. The host
 * hash is computed once per process, so each node layout is its own binary.
 *
 * Inputs are small integers wherever results are checked exactly, so any
 * summation order yields the same value and a mis-routed piece shows up as a
 * wrong number rather than as rounding noise.
 */

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bootstrap.h"
#include "comm.h"
#include "comms/rcclx/develop/meta/testinfra/TestUtils.h"
#include "comms/rcclx/develop/meta/testinfra/TestsDistUtils.h"
#include "meta/hierarchical/hierarchical_allreduce.h"
#include "nccl.h"

#ifndef HIER_TEST_NODES
#define HIER_TEST_NODES 2
#endif

#define HIPCHECK_TEST(cmd)                                          \
  do {                                                              \
    hipError_t error = cmd;                                         \
    if (error != hipSuccess) {                                      \
      FAIL() << "HIP error: " << hipGetErrorString(error) << " at " \
             << __FILE__ << ":" << __LINE__;                        \
    }                                                               \
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

constexpr int kNodes = HIER_TEST_NODES;

// Set in main(): 1 MiB tiles so modest sizes pipeline, and a 4 KiB floor.
constexpr size_t kTileBytes = 1024 * 1024;
constexpr size_t kMinBytes = 4096;

// One aligned unit per sub-shard at 8 ranks: a single tile.
constexpr size_t kSmallCount = 8 * 512;
// Not a multiple of anything the geometry aligns to; still one tile.
constexpr size_t kTailCount = 200'000 + 13;
// About 12.6 MB of floats: 13 pipelined tiles.
constexpr size_t kPipelinedCount = 3 * 1024 * 1024 + 4097;
// Below NCCL_HIER_ALLREDUCE_MIN_BYTES.
constexpr size_t kTinyCount = 512;

uint16_t floatToBf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

float bf16ToFloat(uint16_t h) {
  const uint32_t bits = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

uint64_t fnv1a(const void* data, size_t bytes) {
  uint64_t h = 1469598103934665603ULL;
  const auto* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < bytes; i++) {
    h = (h ^ p[i]) * 1099511628211ULL;
  }
  return h;
}

} // namespace

class HierarchicalAllReduceTest : public ::testing::Test {
 protected:
  // Suite-scoped comm, as in the relay suites: cycling multi-GB comms per case
  // stalls MI350 hosts on VRAM release.
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
      ncclCommDestroy(comm);
      comm = nullptr;
    }
    server.reset();
  }

  void SetUp() override {
    ASSERT_NE(comm, nullptr);
    ASSERT_EQ(comm->nNodes, kNodes) << "NCCL_HOSTID emulation did not apply";
    HIPCHECK_TEST(hipStreamCreate(&stream));
  }

  void TearDown() override {
    for (void* p : allocations) {
      hipFree(p);
    }
    allocations.clear();
    hipStreamDestroy(stream);
  }

  template <typename T>
  T* deviceAlloc(size_t count) {
    void* p = nullptr;
    EXPECT_EQ(hipMalloc(&p, count * sizeof(T)), hipSuccess);
    allocations.push_back(p);
    return static_cast<T*>(p);
  }

  template <typename T>
  void toDevice(T* dst, const std::vector<T>& src) {
    ASSERT_EQ(
        hipMemcpy(
            dst, src.data(), src.size() * sizeof(T), hipMemcpyHostToDevice),
        hipSuccess);
  }

  template <typename T>
  std::vector<T> fromDevice(const T* src, size_t count) {
    std::vector<T> out(count);
    EXPECT_EQ(
        hipMemcpy(out.data(), src, count * sizeof(T), hipMemcpyDeviceToHost),
        hipSuccess);
    return out;
  }

  // input[i] = (rank + 1) + (i % 13) + 100 * replay
  std::vector<float> floatInput(size_t count, int replay = 0) const {
    std::vector<float> v(count);
    for (size_t i = 0; i < count; i++) {
      v[i] = static_cast<float>(globalRank + 1 + (i % 13) + 100 * replay);
    }
    return v;
  }

  float floatExpectedSum(size_t i, int replay = 0) const {
    const int p = numRanks;
    return static_cast<float>(
        p * (p + 1) / 2 + p * static_cast<int>(i % 13) + p * 100 * replay);
  }

  void runFloatSum(size_t count, bool inPlace) {
    float* send = deviceAlloc<float>(count);
    float* recv = inPlace ? send : deviceAlloc<float>(count);
    toDevice(send, floatInput(count));
    NCCLCHECK_TEST(
        ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, stream));
    HIPCHECK_TEST(hipStreamSynchronize(stream));
    const std::vector<float> got = fromDevice(recv, count);
    for (size_t i = 0; i < count; i++) {
      ASSERT_EQ(got[i], floatExpectedSum(i))
          << "rank " << globalRank << " i " << i;
    }
  }

  std::vector<uint64_t> allGatherHash(uint64_t mine) {
    std::vector<uint64_t> all(numRanks);
    all[globalRank] = mine;
    EXPECT_EQ(
        bootstrapAllGather(comm->bootstrap, all.data(), sizeof(uint64_t)),
        ncclSuccess);
    return all;
  }

  template <typename Body>
  hipGraphExec_t captureGraph(Body&& body) {
    hipGraph_t graph = nullptr;
    EXPECT_EQ(
        hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed), hipSuccess);
    body();
    EXPECT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
    hipGraphExec_t exec = nullptr;
    EXPECT_EQ(
        hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);
    hipGraphDestroy(graph);
    return exec;
  }

  void graphReplays(size_t count, bool inPlace) {
    float* send = deviceAlloc<float>(count);
    float* recv = inPlace ? send : deviceAlloc<float>(count);
    toDevice(send, floatInput(count));
    const uint64_t before = rcclx::hier::hierAllReduceEngageCount();
    hipGraphExec_t exec = captureGraph([&] {
      EXPECT_EQ(
          ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, stream),
          ncclSuccess);
    });
    ASSERT_NE(exec, nullptr);
    EXPECT_EQ(rcclx::hier::hierAllReduceEngageCount(), before + 1);

    for (int replay = 1; replay <= 3; replay++) {
      toDevice(send, floatInput(count, replay));
      HIPCHECK_TEST(hipGraphLaunch(exec, stream));
      HIPCHECK_TEST(hipStreamSynchronize(stream));
      const std::vector<float> got = fromDevice(recv, count);
      for (size_t i = 0; i < count; i++) {
        ASSERT_EQ(got[i], floatExpectedSum(i, replay))
            << "rank " << globalRank << " replay " << replay << " i " << i;
      }
    }
    hipGraphExecDestroy(exec);
  }

  static inline ncclComm_t comm{nullptr};
  static inline int localRank{0};
  static inline int globalRank{0};
  static inline int numRanks{0};
  static inline std::unique_ptr<c10d::TCPStore> server;

  hipStream_t stream{nullptr};
  std::vector<void*> allocations;
};

#ifdef HIER_TEST_DEFAULT_MIN_RANKS

// Default NCCL_HIER_ALLREDUCE_MIN_RANKS: a comm of 8 ranks is at most one
// node's worth of GPUs, so it stays on the standard allreduce even though it
// spans hosts over Socket.
TEST_F(HierarchicalAllReduceTest, EightRankCommStaysOnStandardPath) {
  const uint64_t before = rcclx::hier::hierAllReduceEngageCount();
  runFloatSum(kPipelinedCount, /*inPlace=*/false);
  EXPECT_EQ(rcclx::hier::hierAllReduceEngageCount(), before);
}

#elif defined(HIER_TEST_DEFAULT_OFF)

// NCCL_HIER_ALLREDUCE unset: always the standard allreduce.
TEST_F(HierarchicalAllReduceTest, UnsetStaysOnStandardPath) {
  const uint64_t before = rcclx::hier::hierAllReduceEngageCount();
  runFloatSum(kPipelinedCount, /*inPlace=*/false);
  EXPECT_EQ(rcclx::hier::hierAllReduceEngageCount(), before);
}

#elif defined(HIER_TEST_DEFAULT_MAX_NODES)

// Default NCCL_HIER_ALLREDUCE_MAX_NODES (2): a comm spanning more nodes stays
// on the standard allreduce even with NCCL_HIER_ALLREDUCE=1.
TEST_F(HierarchicalAllReduceTest, MoreThanMaxNodesStaysOnStandardPath) {
  ASSERT_GT(kNodes, 2);
  const uint64_t before = rcclx::hier::hierAllReduceEngageCount();
  runFloatSum(kPipelinedCount, /*inPlace=*/false);
  EXPECT_EQ(rcclx::hier::hierAllReduceEngageCount(), before);
}

#else

// First on purpose: a fresh comm with no connections and no scratch, so the
// capture itself has to set up every peer connection it uses.
TEST_F(HierarchicalAllReduceTest, GraphCaptureColdOutOfPlace) {
  graphReplays(kPipelinedCount, /*inPlace=*/false);
}

TEST_F(HierarchicalAllReduceTest, GraphCaptureInPlace) {
  graphReplays(kPipelinedCount, /*inPlace=*/true);
}

TEST_F(HierarchicalAllReduceTest, EngagesOnSocketMultiNode) {
  const uint64_t before = rcclx::hier::hierAllReduceEngageCount();
  runFloatSum(kSmallCount, /*inPlace=*/false);
  EXPECT_EQ(rcclx::hier::hierAllReduceEngageCount(), before + 1);
}

// Integer Avg must truncate the global sum once; dividing per node first would
// truncate each node's partial and can land one lower.
TEST_F(HierarchicalAllReduceTest, Int32AvgTruncatesOnce) {
  const size_t count = kTailCount;
  int32_t* buf = deviceAlloc<int32_t>(count);
  std::vector<int32_t> in(count);
  for (size_t i = 0; i < count; i++) {
    in[i] = (globalRank % 3) + static_cast<int32_t>(i % 7);
  }
  toDevice(buf, in);
  NCCLCHECK_TEST(
      ncclAllReduce(buf, buf, count, ncclInt32, ncclAvg, comm, stream));
  HIPCHECK_TEST(hipStreamSynchronize(stream));
  int32_t rankSum = 0;
  for (int r = 0; r < numRanks; r++) {
    rankSum += r % 3;
  }
  const std::vector<int32_t> got = fromDevice(buf, count);
  for (size_t i = 0; i < count; i++) {
    ASSERT_EQ(
        got[i], (rankSum + numRanks * static_cast<int32_t>(i % 7)) / numRanks)
        << "i " << i;
  }
}

TEST_F(HierarchicalAllReduceTest, FloatSumSingleTile) {
  runFloatSum(kSmallCount, /*inPlace=*/false);
  runFloatSum(kSmallCount, /*inPlace=*/true);
}

TEST_F(HierarchicalAllReduceTest, FloatSumUnalignedTail) {
  runFloatSum(kTailCount, /*inPlace=*/false);
  runFloatSum(kTailCount, /*inPlace=*/true);
}

TEST_F(HierarchicalAllReduceTest, FloatSumPipelined) {
  runFloatSum(kPipelinedCount, /*inPlace=*/false);
  runFloatSum(kPipelinedCount, /*inPlace=*/true);
}

TEST_F(HierarchicalAllReduceTest, FloatAvg) {
  const size_t count = kPipelinedCount;
  float* send = deviceAlloc<float>(count);
  float* recv = deviceAlloc<float>(count);
  std::vector<float> in(count, 2.0f * (globalRank + 1));
  toDevice(send, in);
  NCCLCHECK_TEST(
      ncclAllReduce(send, recv, count, ncclFloat, ncclAvg, comm, stream));
  HIPCHECK_TEST(hipStreamSynchronize(stream));
  const std::vector<float> got = fromDevice(recv, count);
  const std::vector<float> expected(count, static_cast<float>(numRanks + 1));
  EXPECT_EQ(got, expected);
}

TEST_F(HierarchicalAllReduceTest, Bf16SumInPlace) {
  const size_t count = kTailCount;
  uint16_t* buf = deviceAlloc<uint16_t>(count);
  std::vector<uint16_t> in(count);
  for (size_t i = 0; i < count; i++) {
    in[i] = floatToBf16(static_cast<float>(globalRank % 3 + i % 5));
  }
  toDevice(buf, in);
  NCCLCHECK_TEST(
      ncclAllReduce(buf, buf, count, ncclBfloat16, ncclSum, comm, stream));
  HIPCHECK_TEST(hipStreamSynchronize(stream));
  int rankSum = 0;
  for (int r = 0; r < numRanks; r++) {
    rankSum += r % 3;
  }
  const std::vector<uint16_t> got = fromDevice(buf, count);
  for (size_t i = 0; i < count; i++) {
    ASSERT_EQ(
        bf16ToFloat(got[i]),
        static_cast<float>(rankSum + numRanks * static_cast<int>(i % 5)))
        << "i " << i;
  }
}

TEST_F(HierarchicalAllReduceTest, Int32SumInPlace) {
  const size_t count = kPipelinedCount;
  int32_t* buf = deviceAlloc<int32_t>(count);
  std::vector<int32_t> in(count);
  for (size_t i = 0; i < count; i++) {
    in[i] = globalRank * 1000 + static_cast<int32_t>(i % 97);
  }
  toDevice(buf, in);
  NCCLCHECK_TEST(
      ncclAllReduce(buf, buf, count, ncclInt32, ncclSum, comm, stream));
  HIPCHECK_TEST(hipStreamSynchronize(stream));
  const int32_t rankSum = 1000 * numRanks * (numRanks - 1) / 2;
  const std::vector<int32_t> got = fromDevice(buf, count);
  for (size_t i = 0; i < count; i++) {
    ASSERT_EQ(got[i], rankSum + numRanks * static_cast<int32_t>(i % 97))
        << "i " << i;
  }
}

// Non-integer inputs, where summation order matters: every rank must still
// hold the same bits.
TEST_F(HierarchicalAllReduceTest, ResultIsBitIdenticalAcrossRanks) {
  const size_t count = kPipelinedCount;
  float* buf = deviceAlloc<float>(count);
  std::vector<float> in(count);
  for (size_t i = 0; i < count; i++) {
    in[i] = std::sin(static_cast<float>(globalRank * 7919 + i)) * 1e3f;
  }
  toDevice(buf, in);
  NCCLCHECK_TEST(
      ncclAllReduce(buf, buf, count, ncclFloat, ncclSum, comm, stream));
  HIPCHECK_TEST(hipStreamSynchronize(stream));
  const std::vector<float> got = fromDevice(buf, count);
  const std::vector<uint64_t> hashes =
      allGatherHash(fnv1a(got.data(), got.size() * sizeof(float)));
  for (int r = 1; r < numRanks; r++) {
    EXPECT_EQ(hashes[r], hashes[0]) << "rank " << r;
  }
}

TEST_F(HierarchicalAllReduceTest, FallsBackInsideUserGroup) {
  const uint64_t before = rcclx::hier::hierAllReduceEngageCount();
  const size_t count = kSmallCount;
  float* send = deviceAlloc<float>(count);
  float* recv = deviceAlloc<float>(count);
  toDevice(send, floatInput(count));
  NCCLCHECK_TEST(ncclGroupStart());
  NCCLCHECK_TEST(
      ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, stream));
  NCCLCHECK_TEST(ncclGroupEnd());
  HIPCHECK_TEST(hipStreamSynchronize(stream));
  EXPECT_EQ(rcclx::hier::hierAllReduceEngageCount(), before);
  const std::vector<float> got = fromDevice(recv, count);
  for (size_t i = 0; i < count; i++) {
    ASSERT_EQ(got[i], floatExpectedSum(i)) << "i " << i;
  }
}

TEST_F(HierarchicalAllReduceTest, FallsBackForUnsupportedOp) {
#ifdef RCCL_FAST_BUILD
  GTEST_SKIP() << "rccl_fast_build compiles only Sum reduction kernels";
#endif
  const uint64_t before = rcclx::hier::hierAllReduceEngageCount();
  const size_t count = kSmallCount;
  float* buf = deviceAlloc<float>(count);
  toDevice(buf, floatInput(count));
  NCCLCHECK_TEST(
      ncclAllReduce(buf, buf, count, ncclFloat, ncclMax, comm, stream));
  HIPCHECK_TEST(hipStreamSynchronize(stream));
  EXPECT_EQ(rcclx::hier::hierAllReduceEngageCount(), before);
  const std::vector<float> got = fromDevice(buf, count);
  for (size_t i = 0; i < count; i++) {
    ASSERT_EQ(got[i], static_cast<float>(numRanks + i % 13)) << "i " << i;
  }
}

TEST_F(HierarchicalAllReduceTest, FallsBackBelowMinBytes) {
  const uint64_t before = rcclx::hier::hierAllReduceEngageCount();
  runFloatSum(kTinyCount, /*inPlace=*/false);
  EXPECT_EQ(rcclx::hier::hierAllReduceEngageCount(), before);
}

#endif // HIER_TEST_DEFAULT_*

int main(int argc, char* argv[]) {
  const char* lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (lr == nullptr) {
    lr = getenv("LOCAL_RANK");
  }
  const char* ls = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  if (ls == nullptr) {
    ls = getenv("LOCAL_SIZE");
  }
  const int localRank = lr != nullptr ? atoi(lr) : 0;
  const int localSize = ls != nullptr ? atoi(ls) : 8;
  const int perNode = std::max(1, localSize / kNodes);
  const std::string hostId =
      "hier-allreduce-test-node-" + std::to_string(localRank / perNode);
  setenv("NCCL_HOSTID", hostId.c_str(), 1);
  setenv("NCCL_NET", "Socket", 1);
  setenv("NCCL_IB_DISABLE", "1", 1);
  setenv("NCCL_SOCKET_IFNAME", "lo", 1);
#ifndef HIER_TEST_DEFAULT_OFF
  setenv("NCCL_HIER_ALLREDUCE", "1", 1);
#endif
  setenv(
      "NCCL_HIER_ALLREDUCE_TILE_BYTES", std::to_string(kTileBytes).c_str(), 1);
  setenv("NCCL_HIER_ALLREDUCE_MIN_BYTES", std::to_string(kMinBytes).c_str(), 1);
#ifndef HIER_TEST_DEFAULT_MIN_RANKS
  // One 8-GPU box emulates the hosts, so the comm has only 8 ranks.
  setenv("NCCL_HIER_ALLREDUCE_MIN_RANKS", "1", 1);
#endif
#ifndef HIER_TEST_DEFAULT_MAX_NODES
  // Cover the 4- and 8-node schedules, which the default limit skips.
  setenv("NCCL_HIER_ALLREDUCE_MAX_NODES", "8", 1);
#endif

  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
