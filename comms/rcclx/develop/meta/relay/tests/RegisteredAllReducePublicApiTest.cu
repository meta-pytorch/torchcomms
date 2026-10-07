// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <tuple>
#include <vector>

#include "bootstrap.h"
#include "comm.h"
#include "comms/rcclx/develop/meta/testinfra/TestUtils.h"
#include "comms/rcclx/develop/meta/testinfra/TestsDistUtils.h"
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

#define NCCLEXPECT_TEST(cmd)                                                  \
  do {                                                                        \
    ncclResult_t result = cmd;                                                \
    if (result != ncclSuccess) {                                              \
      ADD_FAILURE() << "NCCL error: " << ncclGetErrorString(result) << " at " \
                    << __FILE__ << ":" << __LINE__;                           \
    }                                                                         \
  } while (0)

namespace {

constexpr int kMaxRanks = 4;
constexpr int kSyncTimeoutSec = 20;
constexpr int kStressReplays = 10000;
constexpr size_t kHalfMiB = 512 * 1024;
constexpr size_t kOneMiB = 1024 * 1024;

struct GraphRun {
  hipGraph_t graph{nullptr};
  hipGraphExec_t exec{nullptr};
};

uint16_t floatToBfloat16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t roundToNearestEven = 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + roundToNearestEven) >> 16);
}

constexpr uint16_t kAdversarialInputs[][kMaxRanks] = {
    {0x3f4c, 0xc336, 0x3d04, 0x4329},
    {0xc1e2, 0x4218, 0xbf03, 0xc0dd},
    {0x4417, 0xc3cd, 0xc0fa, 0xc12b},
    {0x42c4, 0xbdef, 0xbe42, 0x4089},
    {0x426d, 0xbdf1, 0xc170, 0xbc00},
    {0xc37d, 0xbc4e, 0xbfb1, 0x439d},
    {0x4193, 0xbea2, 0x3ccb, 0xc065},
    {0x444a, 0x3d39, 0x405b, 0x401b},
};
constexpr size_t kAdversarialInputCount =
    sizeof(kAdversarialInputs) / sizeof(kAdversarialInputs[0]);

float bfloat16ToFloat(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint16_t expectedInputBits(int rank, size_t element, int generation) {
  // Standard RCCL does not promise the registered kernel's stepwise BF16 order.
  if (generation == 14) {
    return floatToBfloat16(static_cast<float>(rank + 1 + (element % 4)));
  }
  const size_t pattern =
      (element + static_cast<size_t>(generation) * 3) % kAdversarialInputCount;
  return kAdversarialInputs[pattern][rank];
}

uint16_t expectedOutputBits(int nRanks, size_t element, int generation) {
  uint16_t accumulator = expectedInputBits(0, element, generation);
  for (int rank = 1; rank < nRanks; ++rank) {
    accumulator = floatToBfloat16(
        bfloat16ToFloat(accumulator) +
        bfloat16ToFloat(expectedInputBits(rank, element, generation)));
  }
  return accumulator;
}

} // namespace

class RegisteredAllReduceTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (comm != nullptr) {
      return;
    }
    int localSize = 0;
    std::tie(localRank, globalRank, numRanks, localSize) =
        getTcpStoreOrMpiInfo();
    const bool isServer = globalRank == 0;
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
      NCCLEXPECT_TEST(ncclCommDestroy(comm));
      comm = nullptr;
    }
    server.reset();
  }

  void SetUp() override {
    ASSERT_NE(comm, nullptr)
        << "suite-scoped comm was not created; SetUpTestSuite did not run";
    HIPCHECK_TEST(hipStreamCreate(&stream));
    HIPCHECK_TEST(hipMalloc(&input, kOneMiB));
    HIPCHECK_TEST(hipMalloc(&output, kOneMiB));
    HIPCHECK_TEST(hipMalloc(&snapshotA, kOneMiB));
    HIPCHECK_TEST(hipMalloc(&snapshotB, kOneMiB));
  }

  void TearDown() override {
    if (request != nullptr) {
      NCCLEXPECT_TEST(ncclRegisteredAllReduceFinalize(request, stream));
      request = nullptr;
    }
    HIPEXPECT_TEST(hipFree(snapshotB));
    HIPEXPECT_TEST(hipFree(snapshotA));
    HIPEXPECT_TEST(hipFree(output));
    HIPEXPECT_TEST(hipFree(input));
    HIPEXPECT_TEST(hipStreamDestroy(stream));
  }

  bool isSupportedRankCount() const {
    return numRanks == 2 || numRanks == kMaxRanks;
  }

  bool isSupportedTopology() const {
    return isSupportedRankCount() && comm->nNodes == 1 &&
        comm->localRanks == numRanks && comm->archName != nullptr &&
        std::strncmp(comm->archName, "gfx950", 6) == 0 && comm->isAllDirectP2p;
  }

  bool allVote(bool localOk) {
    std::vector<uint8_t> votes(numRanks, 0);
    votes[globalRank] = localOk ? 1 : 0;
    const ncclResult_t result =
        bootstrapAllGather(comm->bootstrap, votes.data(), sizeof(uint8_t));
    if (result != ncclSuccess) {
      ADD_FAILURE() << "NCCL error: " << ncclGetErrorString(result) << " at "
                    << __FILE__ << ":" << __LINE__;
      return false;
    }
    return std::all_of(
        votes.begin(), votes.end(), [](uint8_t vote) { return vote != 0; });
  }

  void syncStream(const char* what) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(kSyncTimeoutSec);
    for (;;) {
      const hipError_t status = hipStreamQuery(stream);
      if (status == hipSuccess) {
        return;
      }
      if (status != hipErrorNotReady) {
        FAIL() << "R" << globalRank << " " << what << ": "
               << hipGetErrorString(status);
      }
      if (std::chrono::steady_clock::now() > deadline) {
        std::cerr << "R" << globalRank << " " << what
                  << ": stream did not drain within " << kSyncTimeoutSec
                  << "s; exiting without tearing down live IPC mappings\n";
        std::_Exit(EXIT_FAILURE);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }

  void createRequest() {
    ASSERT_EQ(
        ncclRegisteredAllReduceInit(input, output, kOneMiB, comm, &request),
        ncclSuccess);
    ASSERT_NE(request, nullptr);
  }

  void finalizeRequest() {
    ASSERT_NE(request, nullptr);
    ASSERT_EQ(ncclRegisteredAllReduceFinalize(request, stream), ncclSuccess);
    request = nullptr;
  }

  size_t countForBytes(size_t bytes) const {
    return bytes / sizeof(uint16_t);
  }

  void
  fillInputAsync(size_t bytes, int generation, std::vector<uint16_t>& host) {
    const size_t count = countForBytes(bytes);
    host.resize(count);
    for (size_t i = 0; i < count; ++i) {
      host[i] = expectedInputBits(globalRank, i, generation);
    }
    HIPCHECK_TEST(hipMemcpyAsync(
        input, host.data(), bytes, hipMemcpyHostToDevice, stream));
  }

  void clearOutputAsync(size_t bytes) {
    HIPCHECK_TEST(hipMemsetAsync(output, 0, bytes, stream));
  }

  void execute(size_t bytes) {
    ASSERT_EQ(
        ncclRegisteredAllReduceExec(
            input,
            output,
            countForBytes(bytes),
            ncclBfloat16,
            ncclSum,
            stream,
            request),
        ncclSuccess);
  }

  void snapshotAsync(uint16_t* snapshot, size_t bytes) {
    HIPCHECK_TEST(hipMemcpyAsync(
        snapshot, output, bytes, hipMemcpyDeviceToDevice, stream));
  }

  bool validateDevice(
      const uint16_t* device,
      size_t bytes,
      int generation,
      bool inputBuffer,
      const char* what) {
    const size_t count = countForBytes(bytes);
    std::vector<uint16_t> host(count);
    if (hipMemcpy(host.data(), device, bytes, hipMemcpyDeviceToHost) !=
        hipSuccess) {
      ADD_FAILURE() << "R" << globalRank << " " << what
                    << ": device-to-host copy failed";
      return false;
    }
    for (size_t i = 0; i < count; ++i) {
      const uint16_t expected = inputBuffer
          ? expectedInputBits(globalRank, i, generation)
          : expectedOutputBits(numRanks, i, generation);
      if (host[i] != expected) {
        ADD_FAILURE() << "R" << globalRank << " " << what << " index " << i
                      << " expected BF16 bits " << expected << " got "
                      << host[i];
        return false;
      }
    }
    return true;
  }

  void expectOutput(size_t bytes, int generation, const char* what) {
    EXPECT_TRUE(
        allVote(validateDevice(output, bytes, generation, false, what)));
  }

  void expectInput(size_t bytes, int generation, const char* what) {
    EXPECT_TRUE(allVote(validateDevice(input, bytes, generation, true, what)));
  }

  void expectSnapshot(
      const uint16_t* snapshot,
      size_t bytes,
      int generation,
      const char* what) {
    EXPECT_TRUE(
        allVote(validateDevice(snapshot, bytes, generation, false, what)));
  }

  GraphRun captureExecute(size_t bytes) {
    GraphRun run;
    syncStream("pre-capture stream drain");

    bool localOk = true;
    const hipError_t beginResult =
        hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed);
    if (beginResult != hipSuccess) {
      ADD_FAILURE() << "hipStreamBeginCapture failed: "
                    << hipGetErrorString(beginResult);
      localOk = false;
    } else {
      const ncclResult_t captured = ncclRegisteredAllReduceExec(
          input,
          output,
          countForBytes(bytes),
          ncclBfloat16,
          ncclSum,
          stream,
          request);
      if (captured != ncclSuccess) {
        ADD_FAILURE() << "registeredAllReduceExecute failed during capture: "
                      << ncclGetErrorString(captured);
        localOk = false;
      }
      const hipError_t endResult = hipStreamEndCapture(stream, &run.graph);
      if (endResult != hipSuccess || run.graph == nullptr) {
        ADD_FAILURE() << "hipStreamEndCapture failed: "
                      << hipGetErrorString(endResult);
        localOk = false;
      }
    }
    if (!allVote(localOk)) {
      destroyGraph(run);
      return {};
    }

    size_t nodeCount = 0;
    hipError_t hipResult = hipGraphGetNodes(run.graph, nullptr, &nodeCount);
    if (hipResult != hipSuccess) {
      ADD_FAILURE() << "hipGraphGetNodes(count) failed: "
                    << hipGetErrorString(hipResult);
      localOk = false;
    }
    std::vector<hipGraphNode_t> nodes(nodeCount);
    if (localOk) {
      hipResult = hipGraphGetNodes(run.graph, nodes.data(), &nodeCount);
      if (hipResult != hipSuccess) {
        ADD_FAILURE() << "hipGraphGetNodes(nodes) failed: "
                      << hipGetErrorString(hipResult);
        localOk = false;
      }
    }
    size_t kernelNodes = 0;
    if (localOk) {
      for (hipGraphNode_t node : nodes) {
        hipGraphNodeType type{};
        hipResult = hipGraphNodeGetType(node, &type);
        if (hipResult != hipSuccess) {
          ADD_FAILURE() << "hipGraphNodeGetType failed: "
                        << hipGetErrorString(hipResult);
          localOk = false;
          break;
        }
        if (type == hipGraphNodeTypeKernel) {
          ++kernelNodes;
        }
      }
    }
    if (localOk && (nodeCount != 1 || kernelNodes != 1)) {
      ADD_FAILURE() << "captured graph has " << nodeCount << " nodes and "
                    << kernelNodes << " kernel nodes; expected one of each";
      localOk = false;
    }
    if (!allVote(localOk)) {
      destroyGraph(run);
      return {};
    }

    hipResult = hipGraphInstantiate(&run.exec, run.graph, nullptr, nullptr, 0);
    if (hipResult != hipSuccess) {
      ADD_FAILURE() << "hipGraphInstantiate failed: "
                    << hipGetErrorString(hipResult);
      localOk = false;
    }
    if (!allVote(localOk)) {
      destroyGraph(run);
      return {};
    }
    return run;
  }

  void destroyGraph(GraphRun& run) {
    if (run.exec != nullptr) {
      HIPEXPECT_TEST(hipGraphExecDestroy(run.exec));
      run.exec = nullptr;
    }
    if (run.graph != nullptr) {
      HIPEXPECT_TEST(hipGraphDestroy(run.graph));
      run.graph = nullptr;
    }
  }

  static inline ncclComm_t comm{nullptr};
  static inline int localRank{0};
  static inline int globalRank{0};
  static inline int numRanks{0};
  static inline std::unique_ptr<c10d::TCPStore> server{nullptr};

  hipStream_t stream{nullptr};
  uint16_t* input{nullptr};
  uint16_t* output{nullptr};
  uint16_t* snapshotA{nullptr};
  uint16_t* snapshotB{nullptr};
  void* request{nullptr};
};

TEST_F(RegisteredAllReduceTest, SymbolVersion) {
  EXPECT_EQ(NCCL_REGISTERED_ALL_REDUCE_ABI_VERSION, 1);
  EXPECT_EQ(ncclRegisteredAllReduceAbiVersion(), 1);
}

TEST_F(RegisteredAllReduceTest, LiveRequestBlocksCommunicatorDestroy) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  createRequest();

  EXPECT_EQ(ncclCommDestroy(comm), ncclInvalidUsage);

  std::vector<uint16_t> host;
  fillInputAsync(kHalfMiB, 1, host);
  clearOutputAsync(kHalfMiB);
  execute(kHalfMiB);
  syncStream("all-reduce after rejected communicator destroy");
  expectOutput(kHalfMiB, 1, "request remains usable after rejected destroy");
  finalizeRequest();
}

TEST_F(RegisteredAllReduceTest, InvalidTopologyRejected) {
  if (isSupportedTopology()) {
    GTEST_SKIP() << "Test requires an unsupported registered topology";
  }
  request = reinterpret_cast<void*>(1);
  EXPECT_EQ(
      ncclRegisteredAllReduceInit(input, output, kOneMiB, comm, &request),
      ncclInvalidArgument);
  EXPECT_EQ(request, nullptr);
}

TEST_F(RegisteredAllReduceTest, InvalidAliasRejected) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  EXPECT_EQ(
      ncclRegisteredAllReduceInit(input, input, kOneMiB, comm, &request),
      ncclInvalidArgument);
  EXPECT_EQ(request, nullptr);
}

TEST_F(RegisteredAllReduceTest, InvalidExecArgumentsRejected) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  createRequest();

  EXPECT_EQ(
      ncclRegisteredAllReduceExec(
          input,
          output,
          countForBytes(kHalfMiB),
          ncclFloat32,
          ncclSum,
          stream,
          request),
      ncclInvalidArgument);
  EXPECT_EQ(
      ncclRegisteredAllReduceExec(
          input,
          output,
          countForBytes(kHalfMiB),
          ncclBfloat16,
          ncclProd,
          stream,
          request),
      ncclInvalidArgument);
  EXPECT_EQ(
      ncclRegisteredAllReduceExec(
          input,
          output,
          countForBytes(kHalfMiB) - 1,
          ncclBfloat16,
          ncclSum,
          stream,
          request),
      ncclInvalidArgument);
  EXPECT_EQ(
      ncclRegisteredAllReduceExec(
          snapshotA,
          output,
          countForBytes(kHalfMiB),
          ncclBfloat16,
          ncclSum,
          stream,
          request),
      ncclInvalidArgument);
  EXPECT_EQ(
      ncclRegisteredAllReduceExec(
          input,
          snapshotB,
          countForBytes(kHalfMiB),
          ncclBfloat16,
          ncclSum,
          stream,
          request),
      ncclInvalidArgument);
  EXPECT_EQ(
      ncclRegisteredAllReduceExec(
          input,
          input,
          countForBytes(kHalfMiB),
          ncclBfloat16,
          ncclSum,
          stream,
          request),
      ncclInvalidArgument);
  EXPECT_EQ(
      ncclRegisteredAllReduceExec(
          input,
          output,
          countForBytes(kHalfMiB),
          ncclBfloat16,
          ncclSum,
          stream,
          nullptr),
      ncclInvalidArgument);

  finalizeRequest();
}

TEST_F(RegisteredAllReduceTest, EagerBothSizesPreserveInput) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  createRequest();

  std::vector<uint16_t> host;
  int generation = 1;
  for (size_t bytes : {kHalfMiB, kOneMiB}) {
    fillInputAsync(bytes, generation, host);
    clearOutputAsync(bytes);
    execute(bytes);
    syncStream("eager all-reduce");
    expectOutput(bytes, generation, "eager output");
    expectInput(bytes, generation, "eager input preservation");
    ++generation;
  }

  finalizeRequest();
}

TEST_F(RegisteredAllReduceTest, BackToBackChangedInputsSnapshotOnStream) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  createRequest();

  std::vector<uint16_t> first;
  std::vector<uint16_t> second;
  fillInputAsync(kOneMiB, 3, first);
  clearOutputAsync(kOneMiB);
  execute(kOneMiB);
  snapshotAsync(snapshotA, kOneMiB);

  fillInputAsync(kOneMiB, 4, second);
  clearOutputAsync(kOneMiB);
  execute(kOneMiB);
  snapshotAsync(snapshotB, kOneMiB);

  syncStream("back-to-back all-reduce");
  expectSnapshot(snapshotA, kOneMiB, 3, "first on-stream snapshot");
  expectSnapshot(snapshotB, kOneMiB, 4, "second on-stream snapshot");
  expectInput(kOneMiB, 4, "back-to-back input preservation");

  finalizeRequest();
}

TEST_F(RegisteredAllReduceTest, AlternatesHalfAndOneMiB) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  createRequest();

  std::vector<uint16_t> h0;
  std::vector<uint16_t> h1;
  std::vector<uint16_t> h2;
  std::vector<uint16_t> h3;

  fillInputAsync(kHalfMiB, 5, h0);
  clearOutputAsync(kHalfMiB);
  execute(kHalfMiB);
  snapshotAsync(snapshotA, kHalfMiB);

  fillInputAsync(kOneMiB, 6, h1);
  clearOutputAsync(kOneMiB);
  execute(kOneMiB);
  snapshotAsync(snapshotB, kOneMiB);
  syncStream("half-to-one sequence");
  expectSnapshot(snapshotA, kHalfMiB, 5, "half before one snapshot");
  expectSnapshot(snapshotB, kOneMiB, 6, "one after half snapshot");

  fillInputAsync(kOneMiB, 7, h2);
  clearOutputAsync(kOneMiB);
  execute(kOneMiB);
  snapshotAsync(snapshotB, kOneMiB);

  fillInputAsync(kHalfMiB, 8, h3);
  clearOutputAsync(kHalfMiB);
  execute(kHalfMiB);
  snapshotAsync(snapshotA, kHalfMiB);
  syncStream("one-to-half sequence");
  expectSnapshot(snapshotB, kOneMiB, 7, "one before half snapshot");
  expectSnapshot(snapshotA, kHalfMiB, 8, "half after one snapshot");

  finalizeRequest();
}

TEST_F(RegisteredAllReduceTest, GraphCaptureOneKernelNodeChangedInput) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  createRequest();

  std::vector<uint16_t> capturedInput;
  fillInputAsync(kHalfMiB, 9, capturedInput);
  GraphRun run = captureExecute(kHalfMiB);
  ASSERT_NE(run.exec, nullptr);

  std::vector<uint16_t> replayInput;
  fillInputAsync(kHalfMiB, 10, replayInput);
  clearOutputAsync(kHalfMiB);
  HIPCHECK_TEST(hipGraphLaunch(run.exec, stream));
  syncStream("graph replay with changed input");
  expectOutput(kHalfMiB, 10, "changed-input graph replay");

  destroyGraph(run);
  finalizeRequest();
}

TEST_F(RegisteredAllReduceTest, TenThousandQueuedGraphReplaysFinalizeDrains) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  createRequest();

  std::vector<uint16_t> capturedInput;
  fillInputAsync(kOneMiB, 11, capturedInput);
  GraphRun run = captureExecute(kOneMiB);
  ASSERT_NE(run.exec, nullptr);

  std::vector<uint16_t> replayInput;
  fillInputAsync(kOneMiB, 12, replayInput);
  clearOutputAsync(kOneMiB);
  for (int replay = 0; replay < kStressReplays; ++replay) {
    HIPCHECK_TEST(hipGraphLaunch(run.exec, stream));
  }
  destroyGraph(run);
  finalizeRequest();
  expectOutput(kOneMiB, 12, "queued graph replays drained by finalize");
}

TEST_F(RegisteredAllReduceTest, FinalizeImmediatelyAfterLastEagerDrains) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  createRequest();

  std::vector<uint16_t> host;
  fillInputAsync(kHalfMiB, 13, host);
  clearOutputAsync(kHalfMiB);
  execute(kHalfMiB);
  finalizeRequest();
  expectOutput(kHalfMiB, 13, "eager call drained by finalize");
}

TEST_F(RegisteredAllReduceTest, CaptureTimeFinalizeRejected) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
#if ROCM_VERSION < 60100
  GTEST_SKIP() << "hipStreamGetCaptureInfo_v2 is unavailable before ROCm 6.1";
#else
  createRequest();
  syncStream("pre-finalize-capture stream drain");

  hipGraph_t graph = nullptr;
  HIPCHECK_TEST(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
  EXPECT_EQ(ncclRegisteredAllReduceFinalize(request, stream), ncclInvalidUsage);
  HIPCHECK_TEST(hipStreamEndCapture(stream, &graph));
  if (graph != nullptr) {
    HIPEXPECT_TEST(hipGraphDestroy(graph));
  }

  finalizeRequest();
#endif
}

TEST_F(RegisteredAllReduceTest, NormalAllReduceStillWorks) {
  if (!isSupportedRankCount()) {
    GTEST_SKIP() << "Test requires 2 or 4 ranks, got " << numRanks;
  }
  constexpr size_t kBytes = 4096;
  constexpr size_t kCount = kBytes / sizeof(uint16_t);
  std::vector<uint16_t> host;
  fillInputAsync(kBytes, 14, host);
  clearOutputAsync(kBytes);
  ASSERT_EQ(
      ncclAllReduce(input, output, kCount, ncclBfloat16, ncclSum, comm, stream),
      ncclSuccess);
  syncStream("normal ncclAllReduce");
  expectOutput(kBytes, 14, "normal ncclAllReduce output");
}

TEST_F(RegisteredAllReduceTest, CollectiveInvalidCapacityRollsBack) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  EXPECT_EQ(
      ncclRegisteredAllReduceInit(
          input, output, kHalfMiB - sizeof(uint16_t), comm, &request),
      ncclInvalidArgument);
  EXPECT_EQ(request, nullptr);
}

TEST_F(RegisteredAllReduceTest, InitFinalizeWithoutExecute) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  createRequest();
  finalizeRequest();
}

TEST_F(RegisteredAllReduceTest, RepeatedCreateUseFinalizeCleansResources) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }

  for (int generation = 20; generation < 23; ++generation) {
    createRequest();

    std::vector<uint16_t> host;
    fillInputAsync(kHalfMiB, generation, host);
    clearOutputAsync(kHalfMiB);
    execute(kHalfMiB);
    finalizeRequest();
    expectOutput(kHalfMiB, generation, "repeated create/use/finalize");
  }
}

// Decode-sized payloads (hidden 4608 at batch 1, 4, 8, 32, 64), the smallest
// payload, and sizes on either side of the dedicated four-rank kernels.
constexpr size_t kAlignedSizes[] = {
    16,
    9216,
    36864,
    73728,
    294912,
    589824,
    kHalfMiB - 16,
    kHalfMiB,
    kHalfMiB + 16,
    kOneMiB - 16,
    kOneMiB,
};

TEST_F(RegisteredAllReduceTest, ArbitraryAlignedSizesMatchCanonicalOrder) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  createRequest();

  std::vector<uint16_t> host;
  int generation = 40;
  for (size_t bytes : kAlignedSizes) {
    fillInputAsync(bytes, generation, host);
    clearOutputAsync(kOneMiB);
    execute(bytes);
    syncStream("aligned-size all-reduce");
    expectOutput(bytes, generation, "aligned-size output");
    expectInput(bytes, generation, "aligned-size input preservation");
    ++generation;
  }

  finalizeRequest();
}

TEST_F(RegisteredAllReduceTest, MixedSizesBackToBackWithoutHostSync) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  createRequest();

  std::vector<uint16_t> first;
  std::vector<uint16_t> second;
  std::vector<uint16_t> third;
  constexpr size_t kDecode = 73728;
  fillInputAsync(kDecode, 60, first);
  execute(kDecode);
  snapshotAsync(snapshotA, kDecode);
  fillInputAsync(kHalfMiB, 61, second);
  execute(kHalfMiB);
  fillInputAsync(kDecode, 62, third);
  execute(kDecode);
  snapshotAsync(snapshotB, kDecode);
  syncStream("mixed-size back-to-back all-reduce");
  expectSnapshot(snapshotA, kDecode, 60, "first decode-size snapshot");
  expectSnapshot(snapshotB, kDecode, 62, "decode size after a dedicated size");

  finalizeRequest();
}

TEST_F(RegisteredAllReduceTest, AlignedSizeGraphReplayChangedInput) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  createRequest();
  constexpr size_t kDecode = 73728;

  std::vector<uint16_t> capturedInput;
  fillInputAsync(kDecode, 63, capturedInput);
  GraphRun run = captureExecute(kDecode);
  ASSERT_NE(run.exec, nullptr);

  for (int generation = 64; generation < 67; ++generation) {
    std::vector<uint16_t> replayInput;
    fillInputAsync(kDecode, generation, replayInput);
    clearOutputAsync(kDecode);
    HIPCHECK_TEST(hipGraphLaunch(run.exec, stream));
    syncStream("aligned-size graph replay");
    expectOutput(kDecode, generation, "aligned-size graph replay");
  }

  destroyGraph(run);
  finalizeRequest();
}

TEST_F(RegisteredAllReduceTest, SmallCapacityBoundsPayloads) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  constexpr size_t kCapacity = 4096;
  ASSERT_EQ(
      ncclRegisteredAllReduceInit(input, output, kCapacity, comm, &request),
      ncclSuccess);

  std::vector<uint16_t> host;
  fillInputAsync(kCapacity, 67, host);
  execute(kCapacity);
  syncStream("small-capacity all-reduce");
  expectOutput(kCapacity, 67, "small-capacity output");
  EXPECT_EQ(
      ncclRegisteredAllReduceExec(
          input,
          output,
          countForBytes(2 * kCapacity),
          ncclBfloat16,
          ncclSum,
          stream,
          request),
      ncclInvalidArgument);

  finalizeRequest();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
