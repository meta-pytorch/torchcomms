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
#include "meta/relay/registered_allreduce.h"
#include "meta/relay/registered_allreduce_kernels.h"
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

constexpr int kRanks = rcclx::relay::kRegisteredAllReduceRanks;
constexpr int kSyncTimeoutSec = 20;
constexpr int kStressReplays = 10000;
constexpr size_t kHalfMiB = rcclx::relay::kRegisteredAllReduceHalfMiBBytes;
constexpr size_t kOneMiB = rcclx::relay::kRegisteredAllReduceOneMiBBytes;

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

constexpr uint16_t kAdversarialInputs[][kRanks] = {
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
  const size_t pattern =
      (element + static_cast<size_t>(generation) * 3) % kAdversarialInputCount;
  return kAdversarialInputs[pattern][rank];
}

uint16_t expectedOutputBits(size_t element, int generation) {
  uint16_t accumulator = expectedInputBits(0, element, generation);
  for (int rank = 1; rank < kRanks; ++rank) {
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
    rcclx::relay::registeredAllReduceSetIpcOpenFailureRankForTest(-1);
    if (request != nullptr) {
      NCCLEXPECT_TEST(
          rcclx::relay::registeredAllReduceFinalize(request, stream, true));
      request = nullptr;
    }
    if (offsetAllocation != nullptr) {
      input = fixtureInput;
      HIPEXPECT_TEST(hipFree(offsetAllocation));
      offsetAllocation = nullptr;
    }
    HIPEXPECT_TEST(hipFree(snapshotB));
    HIPEXPECT_TEST(hipFree(snapshotA));
    HIPEXPECT_TEST(hipFree(output));
    HIPEXPECT_TEST(hipFree(input));
    HIPEXPECT_TEST(hipStreamDestroy(stream));
  }

  bool isSupportedTopology() const {
    return numRanks == kRanks && comm->nNodes == 1 &&
        comm->localRanks == kRanks && comm->archName != nullptr &&
        std::strncmp(comm->archName, "gfx950", 6) == 0 && comm->isAllDirectP2p;
  }

  void barrier() {
    NCCLCHECK_TEST(
        bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0));
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

  // Points `input` at offsetBytes inside a fresh allocationBytes allocation so
  // registration must export and apply the offset for peers to see it.
  void useInputAtAllocationOffset(size_t allocationBytes, size_t offsetBytes) {
    ASSERT_EQ(offsetAllocation, nullptr);
    HIPCHECK_TEST(hipMalloc(&offsetAllocation, allocationBytes));
    fixtureInput = input;
    input = reinterpret_cast<uint16_t*>(offsetAllocation + offsetBytes);
  }

  void createRequest() {
    ASSERT_EQ(
        rcclx::relay::registeredAllReducePrepare(comm, &request), ncclSuccess);
    ASSERT_NE(request, nullptr);
    ASSERT_EQ(
        rcclx::relay::registeredAllReduceInit(request, input, output, kOneMiB),
        ncclSuccess);
  }

  void finalizeRequest() {
    ASSERT_NE(request, nullptr);
    ASSERT_EQ(
        rcclx::relay::registeredAllReduceFinalize(request, stream, true),
        ncclSuccess);
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
        rcclx::relay::registeredAllReduceExecute(
            request,
            input,
            output,
            countForBytes(bytes),
            ncclBfloat16,
            ncclSum,
            stream),
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
          : expectedOutputBits(i, generation);
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
      const ncclResult_t captured = rcclx::relay::registeredAllReduceExecute(
          request,
          input,
          output,
          countForBytes(bytes),
          ncclBfloat16,
          ncclSum,
          stream);
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
  uint8_t* offsetAllocation{nullptr};
  uint16_t* fixtureInput{nullptr};
  rcclx::relay::RegisteredAllReduce* request{nullptr};
};

TEST_F(RegisteredAllReduceTest, UnsupportedKernelPayloadRejectedBeforeLaunch) {
  rcclx::relay::RegisteredAllReduceInputTable inputs{};
  rcclx::relay::RegisteredAllReduceStateTable states{};
  EXPECT_EQ(
      rcclx::relay::launchRegisteredAllReduceKernel(
          output, inputs, states, globalRank, 1, stream),
      hipErrorInvalidValue);
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

TEST_F(RegisteredAllReduceTest, CollectiveInvalidCapacityRollsBack) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  ASSERT_EQ(
      rcclx::relay::registeredAllReducePrepare(comm, &request), ncclSuccess);
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(
      rcclx::relay::registeredAllReduceInit(
          request, input, output, kHalfMiB - sizeof(uint16_t)),
      ncclInvalidArgument);
  EXPECT_EQ(rcclx::relay::registeredAllReduceLivePeerMappingsForTest(), 0u);
  EXPECT_EQ(rcclx::relay::registeredAllReduceLiveStateAllocationsForTest(), 0u);
  finalizeRequest();
  EXPECT_EQ(rcclx::relay::registeredAllReduceLiveRequestsForTest(), 0u);
}

TEST_F(
    RegisteredAllReduceTest,
    InputAtNonZeroAllocationOffsetReducesCorrectly) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  // Distinct per-rank offsets: a peer that maps the allocation base instead of
  // the registered pointer reads another rank's bytes and fails bit-exactness.
  useInputAtAllocationOffset(
      2 * kOneMiB, static_cast<size_t>(globalRank + 1) * (kOneMiB / 4));
  createRequest();

  std::vector<uint16_t> host;
  int generation = 30;
  for (size_t bytes : {kHalfMiB, kOneMiB}) {
    fillInputAsync(bytes, generation, host);
    clearOutputAsync(bytes);
    execute(bytes);
    syncStream("offset-input all-reduce");
    expectOutput(bytes, generation, "offset-input output");
    ++generation;
  }

  finalizeRequest();
}

TEST_F(RegisteredAllReduceTest, CapacityPastAllocationEndRejected) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  useInputAtAllocationOffset(2 * kOneMiB, 2 * kOneMiB - kHalfMiB);
  ASSERT_EQ(
      rcclx::relay::registeredAllReducePrepare(comm, &request), ncclSuccess);
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(
      rcclx::relay::registeredAllReduceInit(request, input, output, kOneMiB),
      ncclInvalidArgument);
  EXPECT_EQ(rcclx::relay::registeredAllReduceLivePeerMappingsForTest(), 0u);
  EXPECT_EQ(rcclx::relay::registeredAllReduceLiveStateAllocationsForTest(), 0u);
  finalizeRequest();
  EXPECT_EQ(rcclx::relay::registeredAllReduceLiveRequestsForTest(), 0u);
}

TEST_F(RegisteredAllReduceTest, OneRankIpcOpenFailureRollsBackCollectively) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  rcclx::relay::registeredAllReduceSetIpcOpenFailureRankForTest(1);
  ASSERT_EQ(
      rcclx::relay::registeredAllReducePrepare(comm, &request), ncclSuccess);
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(
      rcclx::relay::registeredAllReduceInit(request, input, output, kOneMiB),
      ncclInternalError);
  rcclx::relay::registeredAllReduceSetIpcOpenFailureRankForTest(-1);

  EXPECT_EQ(rcclx::relay::registeredAllReduceLivePeerMappingsForTest(), 0u);
  EXPECT_EQ(rcclx::relay::registeredAllReduceLiveStateAllocationsForTest(), 0u);
  finalizeRequest();
  EXPECT_EQ(rcclx::relay::registeredAllReduceLiveRequestsForTest(), 0u);
}

TEST_F(RegisteredAllReduceTest, InitFinalizeWithoutExecute) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  createRequest();
  finalizeRequest();
  EXPECT_EQ(rcclx::relay::registeredAllReduceLiveRequestsForTest(), 0u);
  EXPECT_EQ(rcclx::relay::registeredAllReduceLivePeerMappingsForTest(), 0u);
}

// The input-mapping check, driven deterministically: rank 2's check sees stale
// data in rank 0's input from byte 2 MiB on (the symptom observed with
// re-registered buffers). Registration must fail with ncclSystemError on every
// rank and leave no peer mappings or state, and a normal registration on the
// same buffers afterwards must work. Registration must also leave the input's
// contents unchanged, though it writes probe patterns into it.
TEST_F(RegisteredAllReduceTest, StaleInputMappingFailsRegistrationOnEveryRank) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  constexpr size_t kLarge = 5 * kOneMiB;
  uint16_t* largeIn = nullptr;
  uint16_t* largeOut = nullptr;
  HIPCHECK_TEST(hipMalloc(&largeIn, kLarge));
  HIPCHECK_TEST(hipMalloc(&largeOut, kLarge));
  uint16_t* const fixtureIn = input;
  uint16_t* const fixtureOut = output;
  input = largeIn;
  output = largeOut;

  rcclx::relay::registeredAllReduceSetStaleMappingForTest(2, 0);
  ASSERT_EQ(
      rcclx::relay::registeredAllReducePrepare(comm, &request), ncclSuccess);
  const ncclResult_t rejected =
      rcclx::relay::registeredAllReduceInit(request, input, output, kLarge);
  rcclx::relay::registeredAllReduceSetStaleMappingForTest(-1, -1);
  EXPECT_EQ(rejected, ncclSystemError);
  EXPECT_TRUE(allVote(rejected == ncclSystemError));
  EXPECT_EQ(rcclx::relay::registeredAllReduceLivePeerMappingsForTest(), 0u);
  EXPECT_EQ(rcclx::relay::registeredAllReduceLiveStateAllocationsForTest(), 0u);
  finalizeRequest();

  std::vector<uint16_t> host;
  fillInputAsync(kLarge, 31, host);
  syncStream("input before registration");
  ASSERT_EQ(
      rcclx::relay::registeredAllReducePrepare(comm, &request), ncclSuccess);
  ASSERT_EQ(
      rcclx::relay::registeredAllReduceInit(request, input, output, kLarge),
      ncclSuccess);
  expectInput(kLarge, 31, "input after registration");
  clearOutputAsync(kLarge);
  execute(kLarge);
  syncStream("after rejected registration");
  expectOutput(kLarge, 31, "after rejected registration");
  finalizeRequest();

  input = fixtureIn;
  output = fixtureOut;
  HIPCHECK_TEST(hipFree(largeOut));
  HIPCHECK_TEST(hipFree(largeIn));
}

TEST_F(RegisteredAllReduceTest, RepeatedCreateUseFinalizeCleansResources) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }

  for (int generation = 20; generation < 23; ++generation) {
    createRequest();
    EXPECT_EQ(rcclx::relay::registeredAllReduceLiveRequestsForTest(), 1u);
    // Per-request imports are the peers' inputs; state regions are pooled.
    EXPECT_EQ(
        rcclx::relay::registeredAllReduceLivePeerMappingsForTest(),
        static_cast<size_t>(kRanks - 1));

    std::vector<uint16_t> host;
    fillInputAsync(kHalfMiB, generation, host);
    clearOutputAsync(kHalfMiB);
    execute(kHalfMiB);
    finalizeRequest();
    expectOutput(kHalfMiB, generation, "repeated create/use/finalize");
    EXPECT_EQ(rcclx::relay::registeredAllReduceLiveRequestsForTest(), 0u);
    EXPECT_EQ(rcclx::relay::registeredAllReduceLivePeerMappingsForTest(), 0u);
  }
}

// Regression for state regions freed at Finalize: a later allocation that
// reused a freed region's address lost writes and returned wrong data in one
// 4 KiB page (the region's row-flag page), corrupting inputs, outputs and
// epilogue results in later requests. Registers, uses and finalizes requests
// on freshly allocated buffers over and over; every exchange and every fresh
// buffer written after a Finalize must read back exactly.
TEST_F(RegisteredAllReduceTest, ReRegistrationNeverCorruptsFreshBuffers) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  uint16_t* const fixtureIn = input;
  uint16_t* const fixtureOut = output;
  constexpr size_t kCanaryBytes = size_t{2} << 20;
  constexpr uint32_t kCanary = 0x5a5a5a5au;
  for (int iteration = 0; iteration < 24; ++iteration) {
    uint16_t* freshIn = nullptr;
    uint16_t* freshOut = nullptr;
    HIPCHECK_TEST(hipMalloc(&freshIn, kOneMiB));
    HIPCHECK_TEST(hipMalloc(&freshOut, kOneMiB));
    input = freshIn;
    output = freshOut;
    createRequest();
    std::vector<uint16_t> host;
    const size_t bytes = iteration % 2 == 0 ? kOneMiB : kHalfMiB;
    fillInputAsync(bytes, 60 + iteration, host);
    clearOutputAsync(bytes);
    execute(bytes);
    syncStream("re-registration exchange");
    expectOutput(bytes, 60 + iteration, "re-registration exchange");
    finalizeRequest();

    void* canary = nullptr;
    HIPCHECK_TEST(hipMalloc(&canary, kCanaryBytes));
    HIPCHECK_TEST(hipMemsetD32(
        static_cast<hipDeviceptr_t>(canary), kCanary, kCanaryBytes / 4));
    HIPCHECK_TEST(hipDeviceSynchronize());
    std::vector<uint32_t> words(kCanaryBytes / 4);
    HIPCHECK_TEST(
        hipMemcpy(words.data(), canary, kCanaryBytes, hipMemcpyDeviceToHost));
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < words.size(); ++i) {
      if (words[i] != kCanary && bad++ == 0) {
        first = i * 4;
      }
    }
    EXPECT_EQ(bad, 0u) << "R" << globalRank << " iteration " << iteration
                       << ": fresh buffer lost " << bad
                       << " writes, first at byte " << first;
    HIPCHECK_TEST(hipFree(canary));
    HIPCHECK_TEST(hipFree(freshOut));
    HIPCHECK_TEST(hipFree(freshIn));
    input = fixtureIn;
    output = fixtureOut;
  }
  EXPECT_EQ(rcclx::relay::registeredAllReduceLiveStateAllocationsForTest(), 0u);
  EXPECT_EQ(rcclx::relay::registeredAllReduceLivePeerMappingsForTest(), 0u);
}

// State regions are pooled per communicator: sequential requests reuse one
// region (no new peer imports), and a second live request gets its own.
TEST_F(RegisteredAllReduceTest, StateRegionsPooledPerCommunicator) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  createRequest();
  finalizeRequest();
  const size_t imports =
      rcclx::relay::registeredAllReducePooledStateImportsForTest();
  for (int k = 0; k < 3; ++k) {
    createRequest();
    EXPECT_EQ(
        rcclx::relay::registeredAllReducePooledStateImportsForTest(), imports);
    finalizeRequest();
  }
  createRequest();
  rcclx::relay::RegisteredAllReduce* second = nullptr;
  ASSERT_EQ(
      rcclx::relay::registeredAllReducePrepare(comm, &second), ncclSuccess);
  uint16_t* secondIn = nullptr;
  uint16_t* secondOut = nullptr;
  HIPCHECK_TEST(hipMalloc(&secondIn, kOneMiB));
  HIPCHECK_TEST(hipMalloc(&secondOut, kOneMiB));
  ASSERT_EQ(
      rcclx::relay::registeredAllReduceInit(
          second, secondIn, secondOut, kOneMiB),
      ncclSuccess);
  EXPECT_EQ(rcclx::relay::registeredAllReduceLiveStateAllocationsForTest(), 2u);
  EXPECT_GE(
      rcclx::relay::registeredAllReducePooledStateImportsForTest(), imports);
  // Alternate executions between the two live requests on one stream; each
  // owns its state region, so neither disturbs the other.
  rcclx::relay::RegisteredAllReduce* const firstRequest = request;
  uint16_t* const firstIn = input;
  uint16_t* const firstOut = output;
  std::vector<uint16_t> host;
  for (int step = 0; step < 8; ++step) {
    const bool useSecond = step % 2 == 1;
    request = useSecond ? second : firstRequest;
    input = useSecond ? secondIn : firstIn;
    output = useSecond ? secondOut : firstOut;
    const int generation = 90 + step;
    fillInputAsync(kOneMiB, generation, host);
    clearOutputAsync(kOneMiB);
    execute(kOneMiB);
    syncStream("two live requests");
    expectOutput(kOneMiB, generation, "two live requests");
  }
  request = firstRequest;
  input = firstIn;
  output = firstOut;
  ASSERT_EQ(
      rcclx::relay::registeredAllReduceFinalize(second, stream, true),
      ncclSuccess);
  HIPCHECK_TEST(hipFree(secondOut));
  HIPCHECK_TEST(hipFree(secondIn));
  finalizeRequest();
  EXPECT_EQ(rcclx::relay::registeredAllReduceLiveStateAllocationsForTest(), 0u);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
