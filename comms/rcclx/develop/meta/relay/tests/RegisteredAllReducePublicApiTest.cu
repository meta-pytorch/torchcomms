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

// Wide-row epilogue reference (4608-element rows): the specified order emulated
// sequentially, one thread per row. Lane l of wave w owns elements [8i, 8i + 8)
// with i = 64w + l, and lanes 0-63 of the block also own [4096 + 8i, +8).
constexpr int kWideHidden = 4608;
constexpr int kWideThreads = 512;
constexpr int kWideWaves = kWideThreads / 64;
constexpr int kWideHead = kWideThreads * 8;
constexpr int kWideTailThreads = (kWideHidden - kWideHead) / 8;
constexpr float kPostEpsilon = 1e-8f;
constexpr float kPreEpsilon = 1e-5f;

__device__ float wideRefRound(float value) {
  return static_cast<float>(static_cast<__bf16>(value));
}

__device__ float wideRefThreadSum(const float* row, int thread) {
#pragma clang fp contract(off)
  const float* head = row + thread * 8;
  float sum = head[1] * head[1];
  sum = __builtin_fmaf(head[0], head[0], sum);
  for (int element = 2; element < 8; ++element) {
    sum = __builtin_fmaf(head[element], head[element], sum);
  }
  for (int element = 0; element < 8; ++element) {
    const float value = thread < kWideTailThreads
        ? row[kWideHead + thread * 8 + element]
        : 0.0f;
    sum = sum + (thread < kWideTailThreads ? value * value : 0.0f);
  }
  return sum;
}

__device__ float wideRefRowSum(const float* row) {
#pragma clang fp contract(off)
  float waves[kWideWaves];
  for (int wave = 0; wave < kWideWaves; ++wave) {
    float lanes[64];
    float next[64];
    for (int lane = 0; lane < 64; ++lane) {
      lanes[lane] = wideRefThreadSum(row, wave * 64 + lane);
    }
    for (int shift = 8; shift >= 1; shift /= 2) {
      for (int lane = 0; lane < 64; ++lane) {
        next[lane] =
            lanes[lane] + (lane % 16 >= shift ? lanes[lane - shift] : 0.0f);
      }
      for (int lane = 0; lane < 64; ++lane) {
        lanes[lane] = next[lane];
      }
    }
    for (int lane = 0; lane < 64; ++lane) {
      const int group = lane / 16;
      next[lane] =
          ((group == 1 || group == 3) ? lanes[group * 16 - 1] : lanes[lane]) +
          lanes[lane];
    }
    waves[wave] = next[63] + next[31];
  }
  for (int step = kWideWaves / 2; step >= 1; step /= 2) {
    for (int wave = 0; wave < step; ++wave) {
      waves[wave] = waves[wave] + waves[wave + step];
    }
  }
  return waves[0];
}

__device__ float wideRefScale(float sum, float epsilon) {
#pragma clang fp contract(off)
  return __builtin_amdgcn_rsqf(
      __fdiv_rn(sum, static_cast<float>(kWideHidden)) + epsilon);
}

__global__ void wideReferenceKernel(
    const uint16_t* reduced,
    const float* residualIn,
    const uint16_t* postNormWeight,
    const uint16_t* preNormWeight,
    const float* gateAlpha,
    const float* gateBeta,
    float* scratch,
    uint16_t* expectedOutput,
    float* expectedResidual,
    float* expectedRouter) {
#pragma clang fp contract(off)
  const size_t rowOffset = static_cast<size_t>(blockIdx.x) * kWideHidden;
  float* values = scratch + rowOffset;
  auto widen = [](uint16_t bits) {
    return __builtin_bit_cast(float, static_cast<uint32_t>(bits) << 16);
  };
  for (int channel = 0; channel < kWideHidden; ++channel) {
    values[channel] = widen(reduced[rowOffset + channel]);
  }
  const float postScale = wideRefScale(wideRefRowSum(values), kPostEpsilon);
  for (int channel = 0; channel < kWideHidden; ++channel) {
    float scaled = values[channel] * postScale;
    if (postNormWeight != nullptr) {
      scaled = scaled * widen(postNormWeight[channel]);
    }
    const float residual = __builtin_fmaf(
        gateBeta[channel],
        wideRefRound(scaled),
        gateAlpha[channel] * residualIn[rowOffset + channel]);
    expectedResidual[rowOffset + channel] = residual;
    values[channel] = wideRefRound(residual);
  }
  const float preScale = wideRefScale(wideRefRowSum(values), kPreEpsilon);
  for (int channel = 0; channel < kWideHidden; ++channel) {
    const float normed = wideRefRound(
        values[channel] * preScale * widen(preNormWeight[channel]));
    expectedOutput[rowOffset + channel] =
        static_cast<uint16_t>(__builtin_bit_cast(uint32_t, normed) >> 16);
    expectedRouter[rowOffset + channel] = normed;
  }
}

// Deterministic, well-mixed pseudo-random value in [-1, 1).
float wideMixed(uint32_t seed, size_t index) {
  uint64_t x = (static_cast<uint64_t>(seed) << 32) ^ index;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ull;
  x ^= x >> 33;
  return static_cast<float>(x % 2000001ull) / 1000000.0f - 1.0f;
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
            nullptr,
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
          nullptr,
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

  // Bit-exact wide-row epilogue check: fills inputs, runs the fused call, and
  // compares output, residual, and router copy with the device reference
  // applied to the canonical reduced rows.
  struct WideCase {
    size_t rows;
    bool postWeight;
    bool router;
    uint32_t seed;
  };

  bool runWideEpilogue(const WideCase& wide, bool captureGraph = false) {
    const size_t count = wide.rows * kWideHidden;
    const size_t bytes = count * sizeof(uint16_t);
    std::vector<uint16_t> contribution(count);
    std::vector<uint16_t> reduced(count);
    for (size_t i = 0; i < count; ++i) {
      uint16_t accumulator = 0;
      for (int rank = 0; rank < numRanks; ++rank) {
        const uint16_t bits =
            floatToBfloat16(wideMixed(wide.seed + rank, i) * (1.0f + rank));
        if (rank == globalRank) {
          contribution[i] = bits;
        }
        accumulator = rank == 0
            ? bits
            : floatToBfloat16(
                  bfloat16ToFloat(accumulator) + bfloat16ToFloat(bits));
      }
      reduced[i] = accumulator;
    }
    std::vector<float> residual(count);
    for (size_t i = 0; i < count; ++i) {
      residual[i] = wideMixed(wide.seed + 97, i) * 4.0f;
    }
    std::vector<uint16_t> weights(2 * kWideHidden);
    std::vector<float> gates(2 * kWideHidden);
    for (int c = 0; c < kWideHidden; ++c) {
      weights[c] = floatToBfloat16(1.0f + 0.25f * wideMixed(wide.seed + 7, c));
      weights[kWideHidden + c] =
          floatToBfloat16(1.0f + 0.25f * wideMixed(wide.seed + 8, c));
      gates[c] = 0.75f + 0.5f * wideMixed(wide.seed + 9, c);
      gates[kWideHidden + c] = 0.5f + 0.25f * wideMixed(wide.seed + 10, c);
    }

    float* device[5] = {};
    uint16_t* deviceBf16[4] = {};
    const size_t floatBytes = count * sizeof(float);
    for (float*& buffer : device) {
      HIPEXPECT_TEST(hipMalloc(&buffer, floatBytes));
    }
    for (uint16_t*& buffer : deviceBf16) {
      HIPEXPECT_TEST(hipMalloc(
          &buffer, std::max(bytes, 2 * kWideHidden * sizeof(uint16_t))));
    }
    float* residualIn = device[0];
    float* residualOut = device[1];
    float* router = device[2];
    float* expectedResidual = device[3];
    float* scratch = device[4];
    float* expectedRouter = nullptr;
    float* gatesDevice = nullptr;
    HIPEXPECT_TEST(hipMalloc(&expectedRouter, floatBytes));
    HIPEXPECT_TEST(hipMalloc(&gatesDevice, 2 * kWideHidden * sizeof(float)));
    uint16_t* reducedDevice = deviceBf16[0];
    uint16_t* weightsDevice = deviceBf16[1];
    uint16_t* expectedOutput = deviceBf16[2];
    HIPEXPECT_TEST(
        hipMemcpy(input, contribution.data(), bytes, hipMemcpyHostToDevice));
    HIPEXPECT_TEST(
        hipMemcpy(reducedDevice, reduced.data(), bytes, hipMemcpyHostToDevice));
    HIPEXPECT_TEST(hipMemcpy(
        residualIn, residual.data(), floatBytes, hipMemcpyHostToDevice));
    HIPEXPECT_TEST(hipMemcpy(
        weightsDevice,
        weights.data(),
        weights.size() * sizeof(uint16_t),
        hipMemcpyHostToDevice));
    HIPEXPECT_TEST(hipMemcpy(
        gatesDevice,
        gates.data(),
        gates.size() * sizeof(float),
        hipMemcpyHostToDevice));
    HIPEXPECT_TEST(hipMemset(output, 0, bytes));
    HIPEXPECT_TEST(hipMemset(router, 0, floatBytes));

    ncclRegisteredAllReduceGatedResidualNorm norm{};
    norm.residualIn = residualIn;
    norm.residualOut = residualOut;
    norm.routerOut = wide.router ? router : nullptr;
    norm.postNormWeight = wide.postWeight ? weightsDevice : nullptr;
    norm.preNormWeight = weightsDevice + kWideHidden;
    norm.gateAlpha = gatesDevice;
    norm.gateBeta = gatesDevice + kWideHidden;
    norm.hiddenSize = kWideHidden;
    norm.postNormEpsilon = kPostEpsilon;
    norm.preNormEpsilon = kPreEpsilon;

    bool localOk = true;
    if (captureGraph) {
      hipGraph_t graph = nullptr;
      hipGraphExec_t exec = nullptr;
      localOk = hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed) ==
              hipSuccess &&
          ncclRegisteredAllReduceExec(
              input,
              output,
              count,
              ncclBfloat16,
              ncclSum,
              &norm,
              stream,
              request) == ncclSuccess;
      localOk = hipStreamEndCapture(stream, &graph) == hipSuccess && localOk &&
          hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == hipSuccess;
      if (localOk) {
        localOk = hipGraphLaunch(exec, stream) == hipSuccess;
      }
      syncStream("wide epilogue graph replay");
      if (exec != nullptr) {
        HIPEXPECT_TEST(hipGraphExecDestroy(exec));
      }
      if (graph != nullptr) {
        HIPEXPECT_TEST(hipGraphDestroy(graph));
      }
    } else {
      localOk = ncclRegisteredAllReduceExec(
                    input,
                    output,
                    count,
                    ncclBfloat16,
                    ncclSum,
                    &norm,
                    stream,
                    request) == ncclSuccess;
      syncStream("wide epilogue execution");
    }

    hipLaunchKernelGGL(
        wideReferenceKernel,
        dim3(static_cast<unsigned>(wide.rows)),
        dim3(1),
        0,
        nullptr,
        reducedDevice,
        residualIn,
        wide.postWeight ? weightsDevice : nullptr,
        weightsDevice + kWideHidden,
        gatesDevice,
        gatesDevice + kWideHidden,
        scratch,
        expectedOutput,
        expectedResidual,
        expectedRouter);
    HIPEXPECT_TEST(hipDeviceSynchronize());

    std::vector<uint16_t> gotOutput(count), wantOutput(count);
    std::vector<float> gotResidual(count), wantResidual(count);
    std::vector<float> gotRouter(count), wantRouter(count);
    HIPEXPECT_TEST(
        hipMemcpy(gotOutput.data(), output, bytes, hipMemcpyDeviceToHost));
    HIPEXPECT_TEST(hipMemcpy(
        wantOutput.data(), expectedOutput, bytes, hipMemcpyDeviceToHost));
    HIPEXPECT_TEST(hipMemcpy(
        gotResidual.data(), residualOut, floatBytes, hipMemcpyDeviceToHost));
    HIPEXPECT_TEST(hipMemcpy(
        wantResidual.data(),
        expectedResidual,
        floatBytes,
        hipMemcpyDeviceToHost));
    HIPEXPECT_TEST(
        hipMemcpy(gotRouter.data(), router, floatBytes, hipMemcpyDeviceToHost));
    HIPEXPECT_TEST(hipMemcpy(
        wantRouter.data(), expectedRouter, floatBytes, hipMemcpyDeviceToHost));
    const bool outputOk = gotOutput == wantOutput;
    const bool residualOk =
        std::memcmp(gotResidual.data(), wantResidual.data(), floatBytes) == 0;
    const bool routerOk = wide.router
        ? std::memcmp(gotRouter.data(), wantRouter.data(), floatBytes) == 0
        : std::all_of(gotRouter.begin(), gotRouter.end(), [](float v) {
            return v == 0.0f;
          });
    if (!(outputOk && residualOk && routerOk)) {
      ADD_FAILURE() << "R" << globalRank << " wide epilogue rows=" << wide.rows
                    << " postWeight=" << wide.postWeight
                    << " router=" << wide.router << ": output " << outputOk
                    << " residual " << residualOk << " router " << routerOk;
    }

    for (float* buffer : device) {
      HIPEXPECT_TEST(hipFree(buffer));
    }
    for (uint16_t* buffer : deviceBf16) {
      HIPEXPECT_TEST(hipFree(buffer));
    }
    HIPEXPECT_TEST(hipFree(expectedRouter));
    HIPEXPECT_TEST(hipFree(gatesDevice));
    return localOk && outputOk && residualOk && routerOk;
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
          nullptr,
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
          nullptr,
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
          nullptr,
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
          nullptr,
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
          nullptr,
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
          nullptr,
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
          nullptr,
          stream,
          nullptr),
      ncclInvalidArgument);

  finalizeRequest();
}

TEST_F(RegisteredAllReduceTest, PublicExecForwardsGatedResidualNorm) {
  if (!isSupportedTopology() || numRanks != kMaxRanks) {
    GTEST_SKIP() << "Test requires the four-rank registered topology";
  }
  createRequest();
  constexpr size_t kRows = 64;
  constexpr size_t kHidden = 8192;
  constexpr size_t kCount = kRows * kHidden;
  float* residualIn = nullptr;
  float* residualOut = nullptr;
  uint16_t* weights = nullptr;
  float* gates = nullptr;
  HIPCHECK_TEST(hipMalloc(&residualIn, kCount * sizeof(float)));
  HIPCHECK_TEST(hipMalloc(&residualOut, kCount * sizeof(float)));
  HIPCHECK_TEST(hipMalloc(&weights, 2 * kHidden * sizeof(uint16_t)));
  HIPCHECK_TEST(hipMalloc(&gates, 2 * kHidden * sizeof(float)));
  HIPCHECK_TEST(hipMemsetAsync(residualIn, 0, kCount * sizeof(float), stream));
  const std::vector<uint16_t> oneBf16(2 * kHidden, 0x3f80);
  const std::vector<float> oneFloat(2 * kHidden, 1.0f);
  HIPCHECK_TEST(hipMemcpyAsync(
      weights,
      oneBf16.data(),
      2 * kHidden * sizeof(uint16_t),
      hipMemcpyHostToDevice,
      stream));
  HIPCHECK_TEST(hipMemcpyAsync(
      gates,
      oneFloat.data(),
      2 * kHidden * sizeof(float),
      hipMemcpyHostToDevice,
      stream));

  ncclRegisteredAllReduceGatedResidualNorm norm{};
  norm.residualIn = residualIn;
  norm.residualOut = residualOut;
  norm.postNormWeight = weights;
  norm.preNormWeight = weights + kHidden;
  norm.gateAlpha = gates;
  norm.gateBeta = gates + kHidden;
  norm.hiddenSize = kHidden;
  norm.postNormEpsilon = 0.0f;
  norm.preNormEpsilon = 0.0f;

  auto invalid = norm;
  invalid.hiddenSize = kHidden / 2;
  EXPECT_EQ(
      ncclRegisteredAllReduceExec(
          input,
          output,
          kCount,
          ncclBfloat16,
          ncclSum,
          &invalid,
          stream,
          request),
      ncclInvalidArgument);

  // Every rank contributes 1.0: the reduced rows are 4.0 everywhere, so with
  // unit weights and gates both norms map every element to 1.0.
  const std::vector<uint16_t> ones(kCount, 0x3f80);
  HIPCHECK_TEST(hipMemcpyAsync(
      input,
      ones.data(),
      kCount * sizeof(uint16_t),
      hipMemcpyHostToDevice,
      stream));
  EXPECT_EQ(
      ncclRegisteredAllReduceExec(
          input, output, kCount, ncclBfloat16, ncclSum, &norm, stream, request),
      ncclSuccess);
  syncStream("public epilogue execution");
  std::vector<uint16_t> normed(kCount);
  std::vector<float> residual(kCount);
  HIPCHECK_TEST(hipMemcpy(
      normed.data(), output, kCount * sizeof(uint16_t), hipMemcpyDeviceToHost));
  HIPCHECK_TEST(hipMemcpy(
      residual.data(),
      residualOut,
      kCount * sizeof(float),
      hipMemcpyDeviceToHost));
  EXPECT_TRUE(
      allVote(normed == ones && residual == std::vector<float>(kCount, 1.0f)));

  HIPEXPECT_TEST(hipFree(gates));
  HIPEXPECT_TEST(hipFree(weights));
  HIPEXPECT_TEST(hipFree(residualOut));
  HIPEXPECT_TEST(hipFree(residualIn));
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

TEST_F(RegisteredAllReduceTest, GatedResidualNormRejectedOnTwoRanks) {
  if (!isSupportedTopology() || numRanks != 2) {
    GTEST_SKIP() << "Test requires the two-rank registered topology";
  }
  createRequest();
  ncclRegisteredAllReduceGatedResidualNorm norm{};
  norm.residualIn = reinterpret_cast<const float*>(snapshotA);
  norm.residualOut = reinterpret_cast<float*>(snapshotB);
  norm.postNormWeight = snapshotA;
  norm.preNormWeight = snapshotA;
  norm.gateAlpha = reinterpret_cast<const float*>(snapshotA);
  norm.gateBeta = reinterpret_cast<const float*>(snapshotA);
  norm.hiddenSize = 8192;
  EXPECT_EQ(
      ncclRegisteredAllReduceExec(
          input,
          output,
          countForBytes(kOneMiB),
          ncclBfloat16,
          ncclSum,
          &norm,
          stream,
          request),
      ncclInvalidArgument);
  finalizeRequest();
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
          nullptr,
          stream,
          request),
      ncclInvalidArgument);

  finalizeRequest();
}

TEST_F(RegisteredAllReduceTest, WideGatedResidualNormMatchesReference) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  createRequest();
  const WideCase cases[] = {
      {1, false, true, 100},
      {4, false, true, 200},
      {8, false, false, 300},
      {64, false, true, 400},
      {8, true, true, 500},
      {64, true, false, 600},
  };
  for (const WideCase& wide : cases) {
    EXPECT_TRUE(allVote(runWideEpilogue(wide)));
  }
  finalizeRequest();
}

TEST_F(
    RegisteredAllReduceTest,
    WideGatedResidualNormInterleavesWithPlainAndGraph) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  createRequest();
  std::vector<uint16_t> host;
  constexpr size_t kDecode = 73728;
  fillInputAsync(kDecode, 70, host);
  execute(kDecode);
  syncStream("plain before wide epilogue");
  expectOutput(kDecode, 70, "plain before wide epilogue");
  EXPECT_TRUE(allVote(runWideEpilogue({8, false, true, 700})));
  fillInputAsync(kHalfMiB, 71, host);
  execute(kHalfMiB);
  syncStream("plain after wide epilogue");
  expectOutput(kHalfMiB, 71, "plain after wide epilogue");
  EXPECT_TRUE(allVote(runWideEpilogue({4, false, true, 800}, true)));
  finalizeRequest();
}

TEST_F(RegisteredAllReduceTest, WideGatedResidualNormRejectsInvalidShapes) {
  if (!isSupportedTopology()) {
    GTEST_SKIP() << "Test requires the supported registered topology";
  }
  createRequest();
  ncclRegisteredAllReduceGatedResidualNorm norm{};
  norm.residualIn = reinterpret_cast<const float*>(snapshotA);
  norm.residualOut = reinterpret_cast<float*>(snapshotB);
  norm.preNormWeight = snapshotA;
  norm.gateAlpha = reinterpret_cast<const float*>(snapshotA);
  norm.gateBeta = reinterpret_cast<const float*>(snapshotA);
  norm.hiddenSize = kWideHidden;
  auto result = [&](size_t count,
                    const ncclRegisteredAllReduceGatedResidualNorm& n) {
    return ncclRegisteredAllReduceExec(
        input, output, count, ncclBfloat16, ncclSum, &n, stream, request);
  };
  EXPECT_EQ(result(65 * kWideHidden, norm), ncclInvalidArgument);
  EXPECT_EQ(result(kWideHidden + 8, norm), ncclInvalidArgument);
  auto noPre = norm;
  noPre.preNormWeight = nullptr;
  EXPECT_EQ(result(8 * kWideHidden, noPre), ncclInvalidArgument);
  auto other = norm;
  other.hiddenSize = 4096;
  EXPECT_EQ(result(8 * 4096, other), ncclInvalidArgument);
  auto noPostWide8192 = norm;
  noPostWide8192.hiddenSize = 8192;
  EXPECT_EQ(result(64 * 8192, noPostWide8192), ncclInvalidArgument);
  finalizeRequest();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
