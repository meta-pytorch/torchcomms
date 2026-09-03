// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * Unit tests for Fused Multi-Group Sharded Relay AllReduce
 *
 * This tests the phase-synchronized execution of multiple sharded relay
 * allreduces with passthrough-at-helper design.  All groups execute in
 * lockstep phases; helpers forward data without local reduction.
 *
 * Algorithm Design (Phase-Synchronized, Passthrough Helpers):
 * ===========================================================
 * Phase 1 (active→helpers): Both active ranks send chunks to helpers.
 *         Helpers receive into two slots (slot a = data from active rank a).
 *
 * Phase 2 (helpers→active, pipelined): For each helper h, helper forwards:
 *         slot 0 (a0's data) → a1, slot 1 (a1's data) → a0.
 *         Active ranks receive into relay scratch, then add to recvBuff.
 *
 * Phase 3 (scale): Active ranks divide relay region by nActiveRanks for AVG.
 *
 * Phase 4 (active→active): Direct exchange between active ranks.
 *
 * Phase 5 (active reduce): Final reduction on the direct-exchange chunk.
 *
 * Buffer Requirements:
 * ====================
 * ACTIVE ranks: sendBuff == recvBuff is OK (in-place), scratch allocated
 * internally for relay and direct exchange.
 * HELPER ranks: sendBuff == recvBuff is OK (same buffer, uses offset-based
 * access). Buffer must be large enough for nActiveRanks * chunkSize elements.
 * Each helper group MUST have its own buffer (no aliasing across groups).
 *
 * 2D Sparse Parallelism Configuration (8 GPUs, 4 groups):
 *   Group 0: activeRanks = {0, 1}, helpers = {2,3,4,5,6,7}
 *   Group 1: activeRanks = {2, 3}, helpers = {0,1,4,5,6,7}
 *   Group 2: activeRanks = {4, 5}, helpers = {0,1,2,3,6,7}
 *   Group 3: activeRanks = {6, 7}, helpers = {0,1,2,3,4,5}
 *
 * Each rank is ACTIVE for exactly ONE group, HELPER for the other 3.
 */

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "comm.h"
#include "comms/rcclx/develop/meta/testinfra/TestUtils.h"
#include "comms/rcclx/develop/meta/testinfra/TestsDistUtils.h"
#include "meta/relay/sharded_relay_lp.h"
#include "meta/relay/sharded_relay_route.h"
#include "nccl.h"

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

struct ShardedBandwidthResult {
  double algoBW_GBps;
  double busBW_GBps;
  double latency_us;
};

ShardedBandwidthResult calculateShardedBandwidth(
    size_t dataBytes,
    double elapsedMs,
    int numActiveRanks) {
  ShardedBandwidthResult result;
  double elapsedSec = elapsedMs / 1000.0;
  double dataSizeGB =
      static_cast<double>(dataBytes) / (1024.0 * 1024.0 * 1024.0);
  result.algoBW_GBps = dataSizeGB / elapsedSec;
  result.busBW_GBps =
      2.0 * (numActiveRanks - 1.0) / numActiveRanks * dataSizeGB / elapsedSec;
  result.latency_us = elapsedMs * 1000.0;
  return result;
}

// Calculate aggregate bandwidth for multi-group allreduce
// This is what we actually measure: total data across all groups / time
ShardedBandwidthResult calculateMultiGroupAggregateBandwidth(
    size_t dataBytesPerGroup,
    double elapsedMs,
    int numActiveRanks,
    int numGroups) {
  ShardedBandwidthResult result;
  double elapsedSec = elapsedMs / 1000.0;
  // Total data = per-group data × number of groups
  double totalDataSizeGB = static_cast<double>(dataBytesPerGroup) * numGroups /
      (1024.0 * 1024.0 * 1024.0);
  result.algoBW_GBps = totalDataSizeGB / elapsedSec;
  // Bus BW uses the 2-rank allreduce formula since each group is a 2-rank
  // allreduce For 2 ranks: 2*(2-1)/2 = 1.0, so busBW = algoBW for 2-rank groups
  result.busBW_GBps = 2.0 * (numActiveRanks - 1.0) / numActiveRanks *
      totalDataSizeGB / elapsedSec;
  result.latency_us = elapsedMs * 1000.0;
  return result;
}

void printMultiGroupBandwidthResults(
    const std::string& testName,
    size_t dataBytesPerGroup,
    int numRanks,
    int numGroups,
    int activeRanksPerGroup,
    const ShardedBandwidthResult& aggregateResult,
    bool isInPlace) {
  double dataSizePerGroupGB =
      static_cast<double>(dataBytesPerGroup) / (1024.0 * 1024.0 * 1024.0);

  // Total data = per-group data × number of groups
  double totalDataSizeGB = dataSizePerGroupGB * numGroups;

  // Per-group bandwidth (derived from aggregate)
  double perGroupAlgoBW = aggregateResult.algoBW_GBps / numGroups;
  double perGroupBusBW = aggregateResult.busBW_GBps / numGroups;

  std::cout << "\n";
  std::cout << "====================================================\n";
  std::cout << "Multi-Group Sharded Relay AllReduce: " << testName << "\n";
  std::cout << "====================================================\n";
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "  Total Ranks (np):      " << numRanks << "\n";
  std::cout << "  Number of Groups:      " << numGroups << "\n";
  std::cout << "  Active Ranks/Group:    " << activeRanksPerGroup << "\n";
  std::cout << "  Helper Ranks/Group:    " << (numRanks - activeRanksPerGroup)
            << "\n";
  std::cout << "  In-Place:              " << (isInPlace ? "YES" : "NO")
            << "\n";
  std::cout << "  Data Type:             int32\n";
  std::cout << "  Data Size per Group:   " << dataSizePerGroupGB << " GB\n";
  std::cout << "  Total Data (all groups): " << totalDataSizeGB << " GB\n";
  std::cout << "  Element Count/Group:   "
            << (dataBytesPerGroup / sizeof(int32_t)) << "\n";
  std::cout << "----------------------------------------------------\n";
  std::cout << "  Latency:               " << std::setprecision(3)
            << aggregateResult.latency_us << " us\n";
  std::cout << "----------------------------------------------------\n";
  std::cout << "  AGGREGATE BANDWIDTH (all " << numGroups << " groups):\n";
  std::cout << "    Algorithm BW:        " << std::setprecision(2)
            << aggregateResult.algoBW_GBps << " GB/s\n";
  std::cout << "    Bus BW:              " << aggregateResult.busBW_GBps
            << " GB/s\n";
  std::cout << "----------------------------------------------------\n";
  std::cout << "  PER-GROUP BANDWIDTH (derived):\n";
  std::cout << "    Algorithm BW:        " << perGroupAlgoBW << " GB/s\n";
  std::cout << "    Bus BW:              " << perGroupBusBW << " GB/s\n";
  std::cout << "====================================================\n\n";
}

// Test helper: forwards one contiguous send/recv buffer per group to the
// sharded-relay allreduce entry point. Allreduce may be in-place
// (sendPtrs[g] == recvPtrs[g]) or out-of-place (distinct buffers).
static ncclResult_t callAllReduceCompat(
    const void* const* sendPtrs,
    void* const* recvPtrs,
    const size_t* counts,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    hipStream_t stream,
    const int* const* allActiveRanks,
    int nActiveRanksPerGroup,
    int nGroups,
    int lowPrecision = 0) {
  return ncclShardedRelayMultiGroupAllReduce(
      sendPtrs,
      recvPtrs,
      counts,
      datatype,
      op,
      comm,
      stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups,
      lowPrecision);
}

class ShardedRelayMultiGroupAllReduceTest : public ::testing::Test {
 public:
  ShardedRelayMultiGroupAllReduceTest() = default;

  // Comms live for the whole binary instead of being rebuilt per case. They
  // used to be created in SetUp and destroyed in TearDown, so every case freed
  // everything an 8-rank comm owns. On MI350 freeing VRAM makes amdgpu wipe it
  // (amdgpu_bo_release_notify -> amdgpu_fill_buffer) while holding mmap_lock
  // for write, so 8 ranks cycling multi-GB comms serialise into a stall that
  // takes the whole host down. Reusing also matches how comms are really used:
  // a handful, kept for the life of the process.
  //
  // One comm per active-rank shape, because relay state is per-comm and a comm
  // cannot be shared between the 2- and 4-active-rank configurations, plus a
  // dedicated one for the rank barrier. A single store serves all three, with
  // incrTestCount() between them: the unique-ID rendezvous key is derived from
  // that counter, so without bumping it the second comm would read the first
  // one's stale ID. Built eagerly in a fixed order so every rank consumes the
  // same keys in the same sequence.
  static void SetUpTestSuite() {
    int localSize;
    std::tie(localRank, globalRank, numRanks, localSize) =
        getTcpStoreOrMpiInfo();
    const bool isServer = (globalRank == 0);
    if (checkTcpStoreEnv()) {
      server = createTcpStore(isServer);
    } else if (isServer) {
      server = createTcpStore(true);
    }
    barrierComm = makeComm();
    incrTestCount();
    commA2 = makeComm();
    incrTestCount();
    commA4 = makeComm();
  }

  static ncclComm_t makeComm() {
    return createNcclComm(
        globalRank, numRanks, localRank, false, nullptr, server.get());
  }

  // The comm for a given active-rank shape. Every collective call site has
  // nActiveRanksPerGroup in scope, which is how a test reaches its comm.
  static ncclComm_t commFor(int nActiveRanksPerGroup) {
    switch (nActiveRanksPerGroup) {
      case 2:
        return commA2;
      case 4:
        return commA4;
      default:
        ADD_FAILURE() << "no comm cached for nActiveRanksPerGroup="
                      << nActiveRanksPerGroup;
        return nullptr;
    }
  }

  static void TearDownTestSuite() {
    if (server && checkTcpStoreEnv()) {
      finalizeNcclComm(globalRank, server.get());
    }
    for (ncclComm_t* c : {&commA4, &commA2, &barrierComm}) {
      if (*c != nullptr) {
        NCCLCHECK_TEST(ncclCommDestroy(*c));
        *c = nullptr;
      }
    }
    server.reset();
  }

  void SetUp() override {
    ASSERT_NE(this->commA2, nullptr)
        << "suite-scoped comms were not created; SetUpTestSuite did not run";
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
  }

  // Standard 8-rank, 4-group, 2-active-per-group sparse parallelism layout:
  //   Group 0: activeRanks = {0, 1}
  //   Group 1: activeRanks = {2, 3}
  //   Group 2: activeRanks = {4, 5}
  //   Group 3: activeRanks = {6, 7}
  // Returned as raw int* pointers suitable for the multi-group API. The
  // backing storage lives in `storage` and must outlive any use of the
  // returned pointers.
  struct Standard4GroupActiveRanks {
    int storage[4][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}};
    const int* allActiveRanks[4] = {
        storage[0],
        storage[1],
        storage[2],
        storage[3]};
  };

  // 8-rank, 2-group, 4-active-per-group layout for the 4-active path:
  //   Group 0: activeRanks = {0, 1, 2, 3}, helpers = {4, 5, 6, 7}
  //   Group 1: activeRanks = {4, 5, 6, 7}, helpers = {0, 1, 2, 3}
  struct TwoGroupFourActiveRanks {
    int storage[2][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}};
    const int* allActiveRanks[2] = {storage[0], storage[1]};
  };

  struct PermutedTwoGroupFourActiveRanks {
    int storage[2][4] = {{0, 2, 5, 7}, {1, 3, 4, 6}};
    const int* allActiveRanks[2] = {storage[0], storage[1]};
  };

  struct Bfloat16AllReduceCase {
    size_t count;
    ncclRedOp_t op;
    const char* description;
    bool useShardPattern{false};
  };

  // Run a 1-element ncclAllReduce on `scratchBuffer` to act as a cross-rank
  // barrier (ensuring all ranks reach this point before proceeding with
  // benchmark / correctness work).
  void barrierSyncOn(int32_t* /*unused*/) {
    // Run the barrier all-reduce on a dedicated scratch allocation rather than
    // a caller buffer. Several tests pass a buffer that they immediately
    // (re)initialize and feed to the collective under test; running the barrier
    // all-reduce on that same buffer is a write-after-write hazard on element 0
    // (the all-reduce's device write is not guaranteed to be retired before the
    // subsequent host-side init), which can leave element 0 holding the barrier
    // value (0) instead of the init value and produces flaky "expected X but
    // got 0" mismatches.
    int32_t* barrierScratch = nullptr;
    HIPCHECK_TEST(hipMalloc(&barrierScratch, sizeof(int32_t)));
    HIPCHECK_TEST(hipMemset(barrierScratch, 0, sizeof(int32_t)));
    NCCLCHECK_TEST(ncclAllReduce(
        barrierScratch,
        barrierScratch,
        1,
        ncclInt32,
        ncclSum,
        this->barrierComm,
        this->stream));
    HIPCHECK_TEST(hipStreamSynchronize(this->stream));
    HIPCHECK_TEST(hipFree(barrierScratch));
  }

  // Copy `count` int32_t elements from `deviceBuffer` to host and verify
  // they all equal `expectedValue`. Reports up to 10 mismatches via stdout
  // and records failures via gtest. Returns the number of mismatches found
  // (0 on success). Uses ADD_FAILURE/EXPECT_EQ rather than FAIL/ASSERT so
  // it can be called from non-void functions.
  int verifyDeviceBufferEquals(
      const int32_t* deviceBuffer,
      size_t count,
      int32_t expectedValue,
      int groupIndex,
      const char* failureMessage) {
    std::vector<int32_t> hostOutput(count);
    hipError_t hipErr = hipMemcpy(
        hostOutput.data(),
        deviceBuffer,
        count * sizeof(int32_t),
        hipMemcpyDeviceToHost);
    if (hipErr != hipSuccess) {
      ADD_FAILURE() << "HIP error in verifyDeviceBufferEquals: "
                    << hipGetErrorString(hipErr) << " at " << __FILE__ << ":"
                    << __LINE__;
      return -1;
    }

    int errorCount = 0;
    for (size_t i = 0; i < count && errorCount < 10; ++i) {
      if (hostOutput[i] != expectedValue) {
        std::cout << "R" << this->globalRank << ": Group " << groupIndex
                  << " Mismatch at index " << i << ": expected "
                  << expectedValue << " but got " << hostOutput[i] << std::endl;
        errorCount++;
      }
    }
    EXPECT_EQ(errorCount, 0) << failureMessage;
    return errorCount;
  }

  static uint16_t floatToBfloat16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
  }

  static float bfloat16ShardPatternValue(
      int activeIndex,
      size_t elementIndex,
      size_t count,
      int nActiveRanks) {
    const size_t shardSize = count / static_cast<size_t>(nActiveRanks);
    const size_t shardIndex = elementIndex / shardSize;
    const size_t offsetLane = (elementIndex % shardSize) % 11;
    return static_cast<float>(
        1 + (activeIndex + 1) * (shardIndex + 1) + 4 * offsetLane);
  }

  static float expectedBfloat16ShardPatternValue(
      size_t elementIndex,
      size_t count,
      int nActiveRanks,
      ncclRedOp_t op) {
    const size_t shardSize = count / static_cast<size_t>(nActiveRanks);
    const size_t shardIndex = elementIndex / shardSize;
    const size_t offsetLane = (elementIndex % shardSize) % 11;
    const int activeIndexSum = nActiveRanks * (nActiveRanks + 1) / 2;
    const float expectedSum = static_cast<float>(
        nActiveRanks * (1 + 4 * offsetLane) +
        (shardIndex + 1) * activeIndexSum);
    return op == ncclAvg ? expectedSum / nActiveRanks : expectedSum;
  }

  void verifyBfloat16DeviceBufferEquals(
      const uint16_t* deviceBuffer,
      size_t count,
      float expectedValue,
      int nActiveRanks,
      int groupIndex,
      const Bfloat16AllReduceCase& testCase) {
    std::vector<uint16_t> hostOutput(count);
    HIPCHECK_TEST(hipMemcpy(
        hostOutput.data(),
        deviceBuffer,
        count * sizeof(uint16_t),
        hipMemcpyDeviceToHost));

    int errorCount = 0;
    for (size_t i = 0; i < count && errorCount < 10; ++i) {
      const float expected = testCase.useShardPattern
          ? expectedBfloat16ShardPatternValue(
                i, count, nActiveRanks, testCase.op)
          : expectedValue;
      const uint16_t expectedBits = floatToBfloat16(expected);
      if (hostOutput[i] != expectedBits) {
        std::cout << "R" << this->globalRank << ": Group " << groupIndex
                  << " BF16 mismatch for " << testCase.description
                  << " at index " << i << ": expected bits " << expectedBits
                  << " but got " << hostOutput[i] << std::endl;
        errorCount++;
      }
    }
    EXPECT_EQ(errorCount, 0) << testCase.description;
  }

  static const char* allReduceRouteName(rcclx::relay::AllReduceRoute route) {
    switch (route) {
      case rcclx::relay::AllReduceRoute::PureDirect:
        return "PureDirect";
      case rcclx::relay::AllReduceRoute::A2Relay:
        return "A2Relay";
      case rcclx::relay::AllReduceRoute::FlatOffload:
        return "FlatOffload";
    }
    return "unknown";
  }

  // Assert the collective's internal size -> route mapping resolves this
  // geometry to `expected`. Which route runs is owned by the collective and
  // derived only from the message size; this asks the very selector the
  // implementation dispatches on, so a test can neither drive the route nor
  // drift from the thresholds it means to pin.
  void expectAllReduceRoute(
      int nActiveRanksPerGroup,
      int nGroups,
      const size_t* counts,
      size_t elementSize,
      rcclx::relay::AllReduceRoute expected) {
    const rcclx::relay::AllReduceRoute actual =
        rcclx::relay::selectAllReduceRoute(
            nActiveRanksPerGroup, nGroups, counts, elementSize);
    EXPECT_EQ(actual, expected)
        << "internal route selection resolved to " << allReduceRouteName(actual)
        << " but this case is written for " << allReduceRouteName(expected)
        << " (A=" << nActiveRanksPerGroup << ", nGroups=" << nGroups
        << ", max count=" << rcclx::relay::relayMaxCount(counts, nGroups)
        << ", elementSize=" << elementSize << ")";
  }

  void runBfloat16AllReduceCases(
      const int* const* allActiveRanks,
      int nActiveRanksPerGroup,
      int nGroups,
      const std::vector<Bfloat16AllReduceCase>& cases) {
    int myActiveGroup = -1;
    int myActiveIndex = -1;
    for (int g = 0; g < nGroups; ++g) {
      for (int a = 0; a < nActiveRanksPerGroup; ++a) {
        if (allActiveRanks[g][a] == this->globalRank) {
          myActiveGroup = g;
          myActiveIndex = a;
        }
      }
    }

    for (const auto& testCase : cases) {
      const size_t dataBytes = testCase.count * sizeof(uint16_t);
      uint16_t* buffers[4] = {};
      const void* sendPtrs[4] = {};
      void* recvPtrs[4] = {};
      size_t counts[4] = {};

      for (int g = 0; g < nGroups; ++g) {
        const bool isActive = g == myActiveGroup;
        const size_t allocationBytes =
            isActive ? dataBytes : std::max(sizeof(uint16_t), 2 * dataBytes);
        HIPCHECK_TEST(hipMalloc(&buffers[g], allocationBytes));
        sendPtrs[g] = buffers[g];
        recvPtrs[g] = buffers[g];
        counts[g] = testCase.count;
      }

      barrierSyncOn(nullptr);

      for (int g = 0; g < nGroups; ++g) {
        if (g == myActiveGroup) {
          std::vector<uint16_t> hostInput(testCase.count);
          if (testCase.useShardPattern) {
            // The configured active index and logical output offset jointly
            // fingerprint each contribution. This exposes implementations that
            // infer the index from global rank or place a shard at a wrong
            // offset.
            for (size_t i = 0; i < testCase.count; ++i) {
              hostInput[i] = floatToBfloat16(bfloat16ShardPatternValue(
                  myActiveIndex, i, testCase.count, nActiveRanksPerGroup));
            }
          } else {
            // Active index a contributes 2a+1, so SUM=A^2 and AVG=A exactly.
            const float inputValue = static_cast<float>(2 * myActiveIndex + 1);
            std::fill(
                hostInput.begin(),
                hostInput.end(),
                floatToBfloat16(inputValue));
          }
          HIPCHECK_TEST(hipMemcpy(
              buffers[g], hostInput.data(), dataBytes, hipMemcpyHostToDevice));
        } else {
          HIPCHECK_TEST(hipMemset(
              buffers[g], 0, std::max(sizeof(uint16_t), 2 * dataBytes)));
        }
      }

      const ncclResult_t result = callAllReduceCompat(
          sendPtrs,
          recvPtrs,
          counts,
          ncclBfloat16,
          testCase.op,
          this->commFor(nActiveRanksPerGroup),
          this->stream,
          allActiveRanks,
          nActiveRanksPerGroup,
          nGroups);
      ASSERT_EQ(result, ncclSuccess) << testCase.description;
      HIPCHECK_TEST(hipStreamSynchronize(this->stream));

      if (myActiveGroup >= 0) {
        const float expectedValue = testCase.op == ncclAvg
            ? static_cast<float>(nActiveRanksPerGroup)
            : static_cast<float>(nActiveRanksPerGroup * nActiveRanksPerGroup);
        verifyBfloat16DeviceBufferEquals(
            buffers[myActiveGroup],
            testCase.count,
            expectedValue,
            nActiveRanksPerGroup,
            myActiveGroup,
            testCase);
      }

      for (int g = 0; g < nGroups; ++g) {
        HIPCHECK_TEST(hipFree(buffers[g]));
      }
    }
  }

  // Single-group (nGroups=1) A=4 allreduce (in-place). Ranks {0,1,2,3} are
  // active and {4,5,6,7} are passthrough helpers, exercising the A=4 kernel
  // path WITHOUT the multi-group fusion the 4Active_2Groups tests cover.
  // Active rank myActiveIndex fills (myActiveIndex+1)*scale so the reduction
  // detects a missing partner or mis-routed contribution.
  void runAllReduceA4SingleGroup(ncclRedOp_t op) {
    constexpr int nGroups = 1;
    constexpr int nActiveRanksPerGroup = 4;
    const size_t dataBytes = 64ULL * 1024 * 1024;
    const size_t count = dataBytes / sizeof(int32_t);

    const int activeRanks[] = {0, 1, 2, 3};
    const int* allActiveRanks[] = {activeRanks};

    const bool isActive = this->globalRank < nActiveRanksPerGroup;
    const int myActiveIndex = this->globalRank; // 0..3 for active ranks
    const size_t helperBytes =
        static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;

    // SUM: fill (m+1) -> 1+2+3+4 = 10. AVG: fill (m+1)*2 -> 20/4 = 5 (exact in
    // int32). A wrong AVG divisor (e.g. 2) would give 10, so this also checks
    // that AVG uses nActiveRanks rather than a hardcoded 2.
    const int32_t fill =
        op == ncclAvg ? (myActiveIndex + 1) * 2 : myActiveIndex + 1;
    const int32_t expected = op == ncclAvg ? 5 : 10;

    int32_t* buff = nullptr;
    if (isActive) {
      HIPCHECK_TEST(hipMalloc(&buff, dataBytes));
    } else {
      HIPCHECK_TEST(hipMalloc(&buff, helperBytes));
    }

    barrierSyncOn(buff);

    if (isActive) {
      std::vector<int32_t> hostData(count, fill);
      HIPCHECK_TEST(
          hipMemcpy(buff, hostData.data(), dataBytes, hipMemcpyHostToDevice));
    } else {
      HIPCHECK_TEST(hipMemset(buff, 0, helperBytes));
    }

    const void* sendPtrs[1] = {buff};
    void* recvPtrs[1] = {buff};
    size_t counts[1] = {count};
    // 64 MiB single-group A=4 sits far above the 1 MiB independent crossover,
    // so this case must exercise the helper offload rather than the
    // pure-direct reduce-scatter + all-gather.
    expectAllReduceRoute(
        nActiveRanksPerGroup,
        nGroups,
        counts,
        sizeof(int32_t),
        rcclx::relay::AllReduceRoute::FlatOffload);

    ncclResult_t result = callAllReduceCompat(
        sendPtrs,
        recvPtrs,
        counts,
        ncclInt32,
        op,
        this->commFor(nActiveRanksPerGroup),
        this->stream,
        allActiveRanks,
        nActiveRanksPerGroup,
        nGroups);
    ASSERT_EQ(result, ncclSuccess);
    HIPCHECK_TEST(hipStreamSynchronize(this->stream));

    if (isActive) {
      verifyDeviceBufferEquals(
          buff, count, expected, 0, "4-active single-group allreduce mismatch");
    }

    HIPCHECK_TEST(hipFree(buff));
  }

  static inline int localRank{0};
  static inline int globalRank{0};
  static inline int numRanks{0};
  static inline ncclComm_t barrierComm{nullptr};
  static inline ncclComm_t commA2{nullptr};
  static inline ncclComm_t commA4{nullptr};
  static inline std::unique_ptr<c10d::TCPStore> server{nullptr};
  cudaStream_t stream;
};

/**
 * Test: Multi-Group Correctness with 4 groups (IN-PLACE)
 *
 * Tests the fused multi-group allreduce with all 4 sparse groups executing
 * simultaneously with phase synchronization.
 *
 * Each rank is:
 * - ACTIVE for exactly one group (uses real tensor with 1s, in-place)
 * - HELPER for the other 3 groups (uses SAME buffer for send/recv with
 *   offset-based access)
 *
 * Buffer requirements:
 * - ACTIVE: sendBuff == recvBuff (in-place)
 * - HELPER: sendBuff == recvBuff (same buffer, offset-based access)
 *
 * Expected result: Active ranks for each group should have sum = 2
 * (sum of the two active ranks in that group)
 */
TEST_F(ShardedRelayMultiGroupAllReduceTest, Correctness_4Groups_InPlace_64MB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t dataBytes = 64ULL * 1024 * 1024;
  const size_t count = dataBytes / sizeof(int32_t);

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  // Determine which group this rank is active for
  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;

  // Allocate buffers for all groups (one per group)
  // For helpers, buffer must be large enough for nActiveRanksPerGroup * count
  // elements to support offset-based access. We use same buffer for send/recv.
  int32_t* buffers[nGroups];

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      // Active rank: buffer for in-place operation
      HIPCHECK_TEST(hipMalloc(&buffers[g], dataBytes));
    } else {
      // Helper rank: buffer needs space for all active ranks' chunks
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMalloc(&buffers[g], helperBufferSize));
    }
  }

  barrierSyncOn(buffers[0]);

  // Initialize buffers:
  // - For the group where this rank is ACTIVE: fill buffer with 1s
  // - For groups where this rank is HELPER: fill buffer with 0s
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      // Active for this group: use actual data (1s)
      std::vector<int32_t> hostData(count, 1);
      HIPCHECK_TEST(hipMemcpy(
          buffers[g], hostData.data(), dataBytes, hipMemcpyHostToDevice));
      std::cout << "R" << this->globalRank << ": ACTIVE for group " << g
                << " - buffer initialized with 1s" << std::endl;
    } else {
      // Helper for this group: initialize buffer to 0s
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMemset(buffers[g], 0, helperBufferSize));
    }
  }

  // Run multi-group allreduce
  // ACTIVE: sendBuff == recvBuff (in-place)
  // HELPER: sendBuff == recvBuff (same buffer, offset-based access)
  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t counts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    // Both active and helper ranks use same buffer for send and recv
    sendPtrs[g] = buffers[g];
    recvPtrs[g] = buffers[g];
    counts[g] = count; // Same count for all groups in this test
  }

  ncclResult_t result = callAllReduceCompat(
      sendPtrs,
      recvPtrs,
      counts,
      ncclInt32,
      ncclSum,
      this->commFor(nActiveRanksPerGroup),
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  // Verify correctness for the group where this rank is active
  // For in-place operation, the result is in buffers (since sendPtrs ==
  // recvPtrs)
  {
    int g = myActiveGroup;
    int errorCount = verifyDeviceBufferEquals(
        buffers[g],
        count,
        nActiveRanksPerGroup,
        g,
        "Found mismatches in multi-group output");

    if (errorCount == 0) {
      std::cout << "R" << this->globalRank << ": Group " << g << " - All "
                << count << " elements verified correctly!" << std::endl;
    }
  }

  // Cleanup
  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(buffers[g]));
  }
}

/**
 * Test: Multi-Group Correctness with 4 groups (OUT-OF-PLACE)
 */
TEST_F(
    ShardedRelayMultiGroupAllReduceTest,
    Correctness_4Groups_OutOfPlace_64MB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t dataBytes = 64ULL * 1024 * 1024;
  const size_t count = dataBytes / sizeof(int32_t);

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;

  // Allocate buffers
  // Active ranks: separate send/recv for out-of-place operation
  // Helper ranks: single buffer with offset-based access
  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      // Active rank: separate buffers for out-of-place
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], dataBytes));
      HIPCHECK_TEST(hipMalloc(&recvBuffs[g], dataBytes));
    } else {
      // Helper rank: single buffer large enough for all active ranks' chunks
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
      recvBuffs[g] = sendBuffs[g]; // Same buffer for send/recv
    }
  }

  barrierSyncOn(recvBuffs[0]);

  // Initialize buffers
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      std::vector<int32_t> hostData(count, 1);
      HIPCHECK_TEST(hipMemcpy(
          sendBuffs[g], hostData.data(), dataBytes, hipMemcpyHostToDevice));
      HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, dataBytes));
    } else {
      // Helper for this group: initialize buffer to 0s
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }

  // Run multi-group allreduce (out-of-place for active, same buffer for
  // helpers)
  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t counts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
    counts[g] = count;
  }

  ncclResult_t result = callAllReduceCompat(
      sendPtrs,
      recvPtrs,
      counts,
      ncclInt32,
      ncclSum,
      this->commFor(nActiveRanksPerGroup),
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  // Verify correctness for the group where this rank is active
  {
    int g = myActiveGroup;
    verifyDeviceBufferEquals(
        recvBuffs[g],
        count,
        nActiveRanksPerGroup,
        g,
        "Found mismatches in out-of-place multi-group output");
  }

  // Cleanup
  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
    if (g == myActiveGroup) {
      // Only free recvBuffs for active groups (helpers share the buffer)
      HIPCHECK_TEST(hipFree(recvBuffs[g]));
    }
  }
}

TEST_F(
    ShardedRelayMultiGroupAllReduceTest,
    Correctness_4Groups_HeterogeneousPositiveSizes) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t largeCount = (4ULL * 1024 * 1024) / sizeof(int32_t);
  const size_t counts[nGroups] = {largeCount, 513, largeCount, 1023};

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;
  const int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  const int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    const size_t dataBytes = counts[g] * sizeof(int32_t);
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], dataBytes));
      if (g < 2) {
        recvBuffs[g] = sendBuffs[g];
      } else {
        HIPCHECK_TEST(hipMalloc(&recvBuffs[g], dataBytes));
      }
    } else {
      const size_t helperBytes = counts[g] < 1024
          ? sizeof(int32_t)
          : static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBytes));
      recvBuffs[g] = sendBuffs[g];
    }
  }

  barrierSyncOn(recvBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    const size_t dataBytes = counts[g] * sizeof(int32_t);
    if (g == myActiveGroup) {
      std::vector<int32_t> hostData(counts[g], myActiveIndex + 1);
      HIPCHECK_TEST(hipMemcpy(
          sendBuffs[g], hostData.data(), dataBytes, hipMemcpyHostToDevice));
      if (recvBuffs[g] != sendBuffs[g]) {
        HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, dataBytes));
      }
    } else {
      const size_t helperBytes = counts[g] < 1024
          ? sizeof(int32_t)
          : static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBytes));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
  }

  ncclResult_t result = callAllReduceCompat(
      sendPtrs,
      recvPtrs,
      counts,
      ncclInt32,
      ncclSum,
      this->commFor(nActiveRanksPerGroup),
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  verifyDeviceBufferEquals(
      recvBuffs[myActiveGroup],
      counts[myActiveGroup],
      3,
      myActiveGroup,
      "Heterogeneous positive-size SUM mismatch (expected 3)");

  for (int g = 0; g < nGroups; g++) {
    if (recvBuffs[g] != sendBuffs[g]) {
      HIPCHECK_TEST(hipFree(recvBuffs[g]));
    }
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
  }
}

/**
 * Test: BusBW with 4-group Multi-Group AllReduce (1GB per group, IN-PLACE)
 *
 * This is the key benchmark for 2D sparse parallelism performance.
 * All 4 groups execute in parallel with phase synchronization.
 */
TEST_F(ShardedRelayMultiGroupAllReduceTest, Z_BusBW_4Groups_InPlace_1GB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  // 1 GB per group keeps the whole 8-rank suite inside a shared devgpu/CI
  // memory budget: a rank is active for one group and a helper for the other
  // three, so it holds dataBytes + 3 x (nActiveRanksPerGroup x dataBytes).
  const size_t dataBytes = 1ULL * 1024 * 1024 * 1024; // 1 GB per group
  const size_t count = dataBytes / sizeof(int32_t);

  // Benchmark configuration: Run 20 iterations total, take the best time.
  // Early iterations serve as warmup (GPU frequency ramp-up, memory paging).
  // Later iterations should reach peak performance.
  const int nIters = 20;

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  // Determine which group this rank is active for
  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;

  // Allocate buffers for all groups
  // For helpers, use same buffer for send/recv with offset-based access
  int32_t* buffers[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      // Active rank: buffer for in-place operation
      HIPCHECK_TEST(hipMalloc(&buffers[g], dataBytes));
    } else {
      // Helper rank: buffer needs space for all active ranks' chunks
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMalloc(&buffers[g], helperBufferSize));
    }
  }

  barrierSyncOn(buffers[0]);

  // Initialize all buffers with pattern
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMemset(buffers[g], 1, dataBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMemset(buffers[g], 0, helperBufferSize));
    }
  }
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  // Prepare pointers
  // Both active and helper ranks use same buffer for send/recv
  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t counts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = buffers[g];
    recvPtrs[g] = buffers[g];
    counts[g] = count;
  }

  // Run all iterations, timing each one
  hipEvent_t startEvent, stopEvent;
  HIPCHECK_TEST(hipEventCreate(&startEvent));
  HIPCHECK_TEST(hipEventCreate(&stopEvent));

  float bestTimeMs = std::numeric_limits<float>::max();
  float totalTimeMs = 0.0f;

  if (this->globalRank == 0) {
    std::cout << "[Benchmark] Running " << nIters << " iterations..."
              << std::endl;
  }

  for (int iter = 0; iter < nIters; iter++) {
    HIPCHECK_TEST(hipEventRecord(startEvent, this->stream));
    ncclResult_t result = callAllReduceCompat(
        sendPtrs,
        recvPtrs,
        counts,
        ncclInt32,
        ncclSum,
        this->commFor(nActiveRanksPerGroup),
        this->stream,
        allActiveRanks,
        nActiveRanksPerGroup,
        nGroups);
    ASSERT_EQ(result, ncclSuccess);
    HIPCHECK_TEST(hipEventRecord(stopEvent, this->stream));
    HIPCHECK_TEST(hipEventSynchronize(stopEvent));

    float elapsedMs = 0.0f;
    HIPCHECK_TEST(hipEventElapsedTime(&elapsedMs, startEvent, stopEvent));

    if (this->globalRank == 0) {
      std::cout << "  Iteration " << (iter + 1) << ": " << elapsedMs << " ms"
                << std::endl;
    }

    if (elapsedMs < bestTimeMs) {
      bestTimeMs = elapsedMs;
    }
    totalTimeMs += elapsedMs;
  }

  float avgTimeMs = totalTimeMs / nIters;

  if (this->globalRank == 0) {
    std::cout << "\n[Benchmark] Best time: " << bestTimeMs << " ms, "
              << "Avg time: " << avgTimeMs << " ms" << std::endl;

    // Report bandwidth using best time (most representative of peak
    // performance)
    ShardedBandwidthResult bwResult = calculateMultiGroupAggregateBandwidth(
        dataBytes, bestTimeMs, nActiveRanksPerGroup, nGroups);
    printMultiGroupBandwidthResults(
        "4-Group IN-PLACE 1GB",
        dataBytes,
        this->numRanks,
        nGroups,
        nActiveRanksPerGroup,
        bwResult,
        true);
  }

  HIPCHECK_TEST(hipEventDestroy(startEvent));
  HIPCHECK_TEST(hipEventDestroy(stopEvent));
  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(buffers[g]));
  }
}

/**
 * Test: BusBW with 4-group Multi-Group AllReduce (1GB per group, OUT-OF-PLACE)
 */
TEST_F(ShardedRelayMultiGroupAllReduceTest, Z_BusBW_4Groups_OutOfPlace_1GB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t dataBytes = 1ULL * 1024 * 1024 * 1024; // 1 GB per group
  const size_t count = dataBytes / sizeof(int32_t);

  // Benchmark configuration: Run 20 iterations total, take the best time.
  // Early iterations serve as warmup (GPU frequency ramp-up, memory paging).
  // Later iterations should reach peak performance.
  const int nIters = 20;

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;

  // Allocate buffers
  // Active ranks: separate send/recv for out-of-place
  // Helper ranks: single buffer large enough for all active ranks' chunks
  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      // Active rank: separate buffers for out-of-place
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], dataBytes));
      HIPCHECK_TEST(hipMalloc(&recvBuffs[g], dataBytes));
    } else {
      // Helper rank: single buffer with offset-based access
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
      recvBuffs[g] = sendBuffs[g]; // Same buffer for send/recv
    }
  }

  barrierSyncOn(recvBuffs[0]);

  // Initialize all buffers
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 1, dataBytes));
      HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, dataBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  // Prepare pointers
  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t counts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
    counts[g] = count;
  }

  // Run all iterations, timing each one
  hipEvent_t startEvent, stopEvent;
  HIPCHECK_TEST(hipEventCreate(&startEvent));
  HIPCHECK_TEST(hipEventCreate(&stopEvent));

  float bestTimeMs = std::numeric_limits<float>::max();
  float totalTimeMs = 0.0f;

  if (this->globalRank == 0) {
    std::cout << "[Benchmark] Running " << nIters << " iterations..."
              << std::endl;
  }

  for (int iter = 0; iter < nIters; iter++) {
    HIPCHECK_TEST(hipEventRecord(startEvent, this->stream));
    ncclResult_t result = callAllReduceCompat(
        sendPtrs,
        recvPtrs,
        counts,
        ncclInt32,
        ncclSum,
        this->commFor(nActiveRanksPerGroup),
        this->stream,
        allActiveRanks,
        nActiveRanksPerGroup,
        nGroups);
    ASSERT_EQ(result, ncclSuccess);
    HIPCHECK_TEST(hipEventRecord(stopEvent, this->stream));
    HIPCHECK_TEST(hipEventSynchronize(stopEvent));

    float elapsedMs = 0.0f;
    HIPCHECK_TEST(hipEventElapsedTime(&elapsedMs, startEvent, stopEvent));

    if (this->globalRank == 0) {
      std::cout << "  Iteration " << (iter + 1) << ": " << elapsedMs << " ms"
                << std::endl;
    }

    if (elapsedMs < bestTimeMs) {
      bestTimeMs = elapsedMs;
    }
    totalTimeMs += elapsedMs;
  }

  float avgTimeMs = totalTimeMs / nIters;

  if (this->globalRank == 0) {
    std::cout << "\n[Benchmark] Best time: " << bestTimeMs << " ms, "
              << "Avg time: " << avgTimeMs << " ms" << std::endl;

    ShardedBandwidthResult bwResult = calculateMultiGroupAggregateBandwidth(
        dataBytes, bestTimeMs, nActiveRanksPerGroup, nGroups);
    printMultiGroupBandwidthResults(
        "4-Group OUT-OF-PLACE 1GB",
        dataBytes,
        this->numRanks,
        nGroups,
        nActiveRanksPerGroup,
        bwResult,
        false);
  }

  HIPCHECK_TEST(hipEventDestroy(startEvent));
  HIPCHECK_TEST(hipEventDestroy(stopEvent));
  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
    if (g == myActiveGroup) {
      // Only free recvBuffs for active groups (helpers share the buffer)
      HIPCHECK_TEST(hipFree(recvBuffs[g]));
    }
  }
}

/**
 * Test: Single group via multi-group API (backward compatibility check)
 *
 * Verifies that the multi-group API works correctly with nGroups=1,
 * which should behave the same as a single sharded relay allreduce.
 *
 * Buffer requirements:
 * - ACTIVE ranks (0, 1): sendBuff == recvBuff (in-place)
 * - HELPER ranks (2-7): sendBuff == recvBuff (same buffer, offset-based access)
 */
TEST_F(ShardedRelayMultiGroupAllReduceTest, Correctness_SingleGroup_64MB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 1;
  const int nActiveRanksPerGroup = 2;
  const size_t dataBytes = 64ULL * 1024 * 1024;
  const size_t count = dataBytes / sizeof(int32_t);

  const int activeRanks[] = {0, 1};
  const int* allActiveRanks[] = {activeRanks};

  // Determine if this rank is active
  bool isActive = (this->globalRank == 0 || this->globalRank == 1);

  // Allocate buffer - helpers use same buffer for send/recv
  int32_t* buff = nullptr;
  if (isActive) {
    // Active rank: buffer for in-place operation
    HIPCHECK_TEST(hipMalloc(&buff, dataBytes));
  } else {
    // Helper rank: buffer needs space for all active ranks' chunks
    size_t helperBufferSize =
        static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
    HIPCHECK_TEST(hipMalloc(&buff, helperBufferSize));
  }

  barrierSyncOn(buff);

  // Initialize: active ranks (0,1) have 1s, helpers have 0s
  if (isActive) {
    std::vector<int32_t> hostData(count, 1);
    HIPCHECK_TEST(
        hipMemcpy(buff, hostData.data(), dataBytes, hipMemcpyHostToDevice));
  } else {
    size_t helperBufferSize =
        static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
    HIPCHECK_TEST(hipMemset(buff, 0, helperBufferSize));
  }

  // Run single-group allreduce via multi-group API
  // Both active and helper ranks use same buffer for send/recv
  const void* sendPtrs[1];
  void* recvPtrs[1];
  size_t counts[] = {count};

  sendPtrs[0] = buff;
  recvPtrs[0] = buff;

  ncclResult_t result = callAllReduceCompat(
      sendPtrs,
      recvPtrs,
      counts,
      ncclInt32,
      ncclSum,
      this->commFor(nActiveRanksPerGroup),
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  // Verify for active ranks
  if (isActive) {
    verifyDeviceBufferEquals(
        buff,
        count,
        nActiveRanksPerGroup,
        0, // single-group test, group index = 0
        "Single group via multi-group API failed");
  }

  HIPCHECK_TEST(hipFree(buff));
}

/**
 * Test: Correctness with minimum passthrough helper buffers (4 groups,
 * IN-PLACE)
 *
 * Validates that the passthrough kernel produces correct results when
 * each helper group's buffer is allocated at the MINIMUM size required
 * by the passthrough design: nActiveRanks × chunkSize elements.
 *
 * Each helper group has its OWN buffer (no aliasing) because all groups
 * are processed simultaneously under phase-sync.
 *
 * Memory layout per rank:
 *   - Active group: distinct buffer with real data (1s)
 *   - Each helper group: separate buffer sized to nActiveRanks × chunkSize
 *
 * Expected: Active ranks for each group have sum = 2 (same as full-sized).
 */
TEST_F(
    ShardedRelayMultiGroupAllReduceTest,
    Correctness_4Groups_PassthroughHelperEquivalence) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t dataBytes = 64ULL * 1024 * 1024;
  const size_t count = dataBytes / sizeof(int32_t);
  const int numHelpers = this->numRanks - nActiveRanksPerGroup;
  const int numChunks = numHelpers + 2;

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;

  // Compute the MINIMUM passthrough helper buffer size.
  size_t chunkSize = count / numChunks;
  chunkSize = (chunkSize / 128) * 128; // CHUNK_ALIGN_ELEMENTS
  if (chunkSize == 0) {
    chunkSize = count;
  }
  size_t minHelperElements =
      std::min(count, static_cast<size_t>(nActiveRanksPerGroup) * chunkSize);
  size_t minHelperBytes = minHelperElements * sizeof(int32_t);

  // Allocate: 1 active buffer + 3 separate minimal helper buffers
  int32_t* activeBuffer = nullptr;
  int32_t* helperBuffers[nGroups];

  HIPCHECK_TEST(hipMalloc(&activeBuffer, dataBytes));
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      helperBuffers[g] = nullptr;
    } else {
      HIPCHECK_TEST(hipMalloc(&helperBuffers[g], minHelperBytes));
    }
  }

  barrierSyncOn(activeBuffer);

  // Initialize: active buffer = 1s, helpers = 0s
  std::vector<int32_t> hostData(count, 1);
  HIPCHECK_TEST(hipMemcpy(
      activeBuffer, hostData.data(), dataBytes, hipMemcpyHostToDevice));
  for (int g = 0; g < nGroups; g++) {
    if (g != myActiveGroup) {
      HIPCHECK_TEST(hipMemset(helperBuffers[g], 0, minHelperBytes));
    }
  }

  // Build pointer arrays: each helper group has its own buffer
  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t counts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      sendPtrs[g] = activeBuffer;
      recvPtrs[g] = activeBuffer;
    } else {
      sendPtrs[g] = helperBuffers[g];
      recvPtrs[g] = helperBuffers[g];
    }
    counts[g] = count;
  }

  ncclResult_t result = callAllReduceCompat(
      sendPtrs,
      recvPtrs,
      counts,
      ncclInt32,
      ncclSum,
      this->commFor(nActiveRanksPerGroup),
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  // Verify correctness for this rank's active group
  {
    int errorCount = verifyDeviceBufferEquals(
        activeBuffer,
        count,
        nActiveRanksPerGroup,
        myActiveGroup,
        "Found mismatches in passthrough-helper-equivalence output");

    if (errorCount == 0) {
      std::cout << "R" << this->globalRank << ": Group " << myActiveGroup
                << " - All " << count
                << " elements verified correctly (min passthrough helper)!"
                << std::endl;
    }
  }

  HIPCHECK_TEST(hipFree(activeBuffer));
  for (int g = 0; g < nGroups; g++) {
    if (g != myActiveGroup) {
      HIPCHECK_TEST(hipFree(helperBuffers[g]));
    }
  }
}

/**
 * Test: Correctness_PartialGroupsZeroCount
 *
 * Regression test for the BM-FM failure:
 *   "RCCLX Sharded Relay Multi-Group AllReduce failed: internal error"
 *
 * When different sparse groups have different numbers of tensors (e.g.
 * [94, 101, 88, 99]), the iterative non-batched loop passes count=0 for
 * groups that have exhausted their tensors while other groups still have
 * data. The kernel must skip those count=0 groups in every phase instead of
 * attempting NCCL operations with zero-element buffers (which trigger an
 * internal error via the scratch buffer cache returning nullptr for size=0).
 *
 * Setup: 4 groups (8 ranks), 2 active ranks per group.
 *   - Groups 0 and 1: count = 16MB (have data, must produce correct sum=2)
 *   - Groups 2 and 3: count = 0    (exhausted, must not corrupt or crash)
 *
 * All ranks call the same API with the same count array (as in production,
 * where counts are synchronized via allgather before the iterative loop).
 */
TEST_F(
    ShardedRelayMultiGroupAllReduceTest,
    Correctness_PartialGroupsZeroCount) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t dataBytes = 16ULL * 1024 * 1024; // 16MB for groups with data
  const size_t count = dataBytes / sizeof(int32_t);

  // Groups 0,1 have data; groups 2,3 are exhausted (count=0)
  const size_t counts[nGroups] = {count, count, 0, 0};

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;

  // Allocate buffers.  For count=0 groups we still allocate a small placeholder
  // (matching how the Python layer passes a 1-element empty tensor), but its
  // content is irrelevant since the kernel will skip the group entirely.
  const size_t placeholderBytes = sizeof(int32_t); // 1 element
  int32_t* buffers[nGroups];

  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0) {
      // count=0 group: allocate a tiny placeholder buffer (kernel skips it)
      HIPCHECK_TEST(hipMalloc(&buffers[g], placeholderBytes));
      HIPCHECK_TEST(hipMemset(buffers[g], 0, placeholderBytes));
      continue;
    }
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(&buffers[g], dataBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMalloc(&buffers[g], helperBufferSize));
    }
  }

  barrierSyncOn(buffers[0]);

  // Initialize: active ranks for groups 0,1 fill with 1s; helpers fill with 0s
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0) {
      continue; // placeholder already zeroed
    }
    if (g == myActiveGroup) {
      std::vector<int32_t> hostData(count, 1);
      HIPCHECK_TEST(hipMemcpy(
          buffers[g], hostData.data(), dataBytes, hipMemcpyHostToDevice));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMemset(buffers[g], 0, helperBufferSize));
    }
  }

  // Build pointer arrays (same buffer for send/recv = in-place)
  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = buffers[g];
    recvPtrs[g] = buffers[g];
  }

  // Execute — must succeed and not trigger internal error for count=0 groups
  ncclResult_t result = callAllReduceCompat(
      sendPtrs,
      recvPtrs,
      counts,
      ncclInt32,
      ncclSum,
      this->commFor(nActiveRanksPerGroup),
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess)
      << "ncclShardedRelayMultiGroupAllReduce failed with partial count=0 groups";
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  // Verify groups with count>0 produced correct sum=2
  if (myActiveGroup < 2) { // groups 0 and 1 had data
    verifyDeviceBufferEquals(
        buffers[myActiveGroup],
        count,
        nActiveRanksPerGroup,
        myActiveGroup,
        "Correctness failed for groups with count>0 when other groups have count=0");
  }
  // Ranks 4-7 (active for groups 2,3 with count=0) have nothing to verify.

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(buffers[g]));
  }
}

/**
 * 4-ACTIVE tests (2 groups of 4 active ranks each).
 *   Group 0: active {0,1,2,3}, helpers {4,5,6,7}
 *   Group 1: active {4,5,6,7}, helpers {0,1,2,3}
 * Each active rank fills its buffer with a distinct value (myActiveIndex+1) so
 * the SUM (1+2+3+4 = 10) detects a missing partner / mis-routed contribution.
 */
/**
 * Test: single group (nGroups == 1) at 4 active ranks, SUM and AVG, both
 * placements.
 *
 * nGroups == 1 makes the active ranks {0,1,2,3} and the helpers {4,5,6,7}
 * disjoint, which is what puts the flat allreduce on its software-pipelined
 * schedule -- the only path that pipelines TWO dependencies at once (the
 * offload's reduce-and-broadcast and the direct region's reduce-scatter into
 * all-gather), and the only one using a tile-major dScratch. 128 MB clears the
 * crossover where that schedule starts beating the two-group one, so this is
 * its only coverage: every other 4-active case runs 2 groups and stays
 * unpipelined.
 *
 * AVG is what pins the divisor, which the pipelined path applies once per tile
 * at the owner and once per tile at the helper. Values are chosen so the
 * average is exact in integers: contributions 1..4 average to 2 with A = 4...
 * deliberately 2 + 8 + 14 + 16 = 40, so SUM is 40 and AVG is exactly 10.
 */
class ShardedRelayAllReduceSingleGroupA4Test
    : public ShardedRelayMultiGroupAllReduceTest {
 protected:
  static int32_t contribution(int activeIndex) {
    static const int32_t values[4] = {2, 8, 14, 16};
    return values[activeIndex];
  }

  void runSingleGroupA4Case(
      bool inPlace,
      ncclRedOp_t op,
      size_t dataBytes = 128ULL * 1024 * 1024) {
    const int nGroups = 1;
    const int nActiveRanksPerGroup = 4;
    const size_t count = dataBytes / sizeof(int32_t);
    const int32_t sum =
        contribution(0) + contribution(1) + contribution(2) + contribution(3);
    const int32_t expected =
        (op == ncclAvg) ? (sum / nActiveRanksPerGroup) : sum;

    const int activeRanks[] = {0, 1, 2, 3};
    const int* allActiveRanks[] = {activeRanks};
    const bool isActive = this->globalRank < nActiveRanksPerGroup;
    const int myActiveIndex = this->globalRank;

    int32_t* sendBuff = nullptr;
    int32_t* recvBuff = nullptr;
    HIPCHECK_TEST(hipMalloc(&sendBuff, dataBytes));
    if (isActive && !inPlace) {
      HIPCHECK_TEST(hipMalloc(&recvBuff, dataBytes));
    }

    barrierSyncOn(sendBuff);

    if (isActive) {
      std::vector<int32_t> hostData(count, contribution(myActiveIndex));
      HIPCHECK_TEST(hipMemcpy(
          sendBuff, hostData.data(), dataBytes, hipMemcpyHostToDevice));
      if (!inPlace) {
        HIPCHECK_TEST(hipMemset(recvBuff, 0, dataBytes));
      }
    } else {
      HIPCHECK_TEST(hipMemset(sendBuff, 0, dataBytes));
    }
    int32_t* out = inPlace ? sendBuff : recvBuff;

    const void* sendPtrs[1] = {sendBuff};
    void* recvPtrs[1] = {isActive ? out : sendBuff};
    size_t counts[1] = {count};

    const ncclResult_t result = callAllReduceCompat(
        sendPtrs,
        recvPtrs,
        counts,
        ncclInt32,
        op,
        this->commFor(nActiveRanksPerGroup),
        this->stream,
        allActiveRanks,
        nActiveRanksPerGroup,
        nGroups);
    ASSERT_EQ(result, ncclSuccess);
    HIPCHECK_TEST(hipStreamSynchronize(this->stream));

    if (isActive) {
      verifyDeviceBufferEquals(
          out, count, expected, 0, "4-active single-group allreduce mismatch");
    }

    HIPCHECK_TEST(hipFree(sendBuff));
    if (recvBuff != nullptr) {
      HIPCHECK_TEST(hipFree(recvBuff));
    }
  }
};

TEST_F(ShardedRelayAllReduceSingleGroupA4Test, Correctness_Sum_OutOfPlace) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runSingleGroupA4Case(/*inPlace=*/false, ncclSum);
}

TEST_F(ShardedRelayAllReduceSingleGroupA4Test, Correctness_Sum_InPlace) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runSingleGroupA4Case(/*inPlace=*/true, ncclSum);
}

TEST_F(ShardedRelayAllReduceSingleGroupA4Test, Correctness_Avg_InPlace) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runSingleGroupA4Case(/*inPlace=*/true, ncclAvg);
}

/**
 * Test: the pipelined allreduce's region split fits inside the count at EVERY
 * geometry buildShardedRelayRankConfig() accepts, not just the A = H = 4 one
 * the 8-rank cases above exercise.
 *
 * The GPU cases here are pinned to 8 ranks, so a single-group A = 4 call always
 * has exactly 4 helpers; the combinations a 9-to-16-rank comm would produce
 * (A = 4 with H = 5..8, or A = 8 with H = 8) cannot be built from this harness.
 * What those geometries break is arithmetic, not scheduling:
 * shardedRelayAllReduceFlatPipelined() takes pO = 2*H*T*y and pD = count - pO,
 * then dShard = pD/A with a last tile of dShard - (T-1)*y. If the unit count
 * per tile is below 2H + A both subtractions underflow as size_t and a wild
 * dShard reaches the reduce kernels and ncclSend. Assert the invariant directly
 * over the whole accepted (A, H) space, which needs no ranks at all.
 */
TEST(ShardedRelayAllReducePipelineUnits, LayoutFitsAtEveryAcceptedGeometry) {
  // Mirrors SHARDED_RELAY_MAX_ACTIVE / SHARDED_RELAY_MAX_HELPERS.
  constexpr int kMaxActive = 8;
  constexpr int kMaxHelpers = 8;

  for (int a = 2; a <= kMaxActive; a *= 2) {
    for (int h = 1; h <= kMaxHelpers; h++) {
      const int unitsPerTile =
          rcclx::relay::relayAllReducePipelineUnitsPerTile(a, h);
      for (int t = 1; t <= 8; t *= 2) {
        // Offload takes 2*H*T units; the direct region needs A*T so every
        // owner's shard can still tile into T pieces of y.
        EXPECT_LE(2 * h * t + a * t, unitsPerTile * t)
            << "pipelined allreduce layout overruns the count at A=" << a
            << " H=" << h << " T=" << t << ": needs " << (2 * h * t + a * t)
            << " units out of " << (unitsPerTile * t);
      }
    }
  }

  // The shipped 8-GPU geometry must still resolve to the 12 the perf numbers
  // were measured at, so the parameterization is a generalization and not a
  // change.
  EXPECT_EQ(rcclx::relay::relayAllReducePipelineUnitsPerTile(4, 4), 12);
}

// The two cases below sit BELOW the 1 MiB A=4 independent crossover, where the
// full-exchange fast path replaces the direct reduce-scatter + all-gather: one
// group in which each active rank ships its whole buffer to the other three,
// then one fused reduce over all four contributions. Both aliasing forms are
// covered because they take different kernels (in-place seeds the reduce from
// the destination, out-of-place seeds it from sendbuff), and AVG is covered
// because the divisor is folded into that single reduce.
TEST_F(
    ShardedRelayAllReduceSingleGroupA4Test,
    Correctness_Sum_OutOfPlace_FullExchange) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runSingleGroupA4Case(
      /*inPlace=*/false, ncclSum, /*dataBytes=*/256ULL * 1024);
}

TEST_F(
    ShardedRelayAllReduceSingleGroupA4Test,
    Correctness_Avg_InPlace_FullExchange) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runSingleGroupA4Case(
      /*inPlace=*/true, ncclAvg, /*dataBytes=*/256ULL * 1024);
}

// The two cases below sit ABOVE kRelayUniformDirectOpMinBytes (256 MiB), where
// the offload is issued as two half-sized operations per link instead of one,
// so that every operation in a group matches the direct tiles. The 128 MB cases
// above stay below that threshold, so between them both sides of the gate are
// covered. AVG is included because the divisor is applied per piece.
TEST_F(
    ShardedRelayAllReduceSingleGroupA4Test,
    Correctness_Sum_OutOfPlace_UniformOps) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runSingleGroupA4Case(
      /*inPlace=*/false, ncclSum, /*dataBytes=*/256ULL * 1024 * 1024);
}

TEST_F(
    ShardedRelayAllReduceSingleGroupA4Test,
    Correctness_Avg_InPlace_UniformOps) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runSingleGroupA4Case(
      /*inPlace=*/true, ncclAvg, /*dataBytes=*/256ULL * 1024 * 1024);
}

TEST_F(
    ShardedRelayMultiGroupAllReduceTest,
    Correctness_4Active_2Groups_InPlace) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int nGroups = 2;
  const int nActiveRanksPerGroup = 4;
  const size_t dataBytes = 64ULL * 1024 * 1024;
  const size_t count = dataBytes / sizeof(int32_t);

  TwoGroupFourActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* buffers[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(&buffers[g], dataBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMalloc(&buffers[g], helperBufferSize));
    }
  }

  barrierSyncOn(buffers[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      std::vector<int32_t> hostData(count, myActiveIndex + 1);
      HIPCHECK_TEST(hipMemcpy(
          buffers[g], hostData.data(), dataBytes, hipMemcpyHostToDevice));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMemset(buffers[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t counts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = buffers[g];
    recvPtrs[g] = buffers[g];
    counts[g] = count;
  }

  ncclResult_t result = callAllReduceCompat(
      sendPtrs,
      recvPtrs,
      counts,
      ncclInt32,
      ncclSum,
      this->commFor(nActiveRanksPerGroup),
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  // Expected SUM across 4 active ranks = 1+2+3+4 = 10.
  verifyDeviceBufferEquals(
      buffers[myActiveGroup],
      count,
      10,
      myActiveGroup,
      "4-active in-place SUM mismatch (expected 10)");

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(buffers[g]));
  }
}

/**
 * Tiny-count regression: forces the relay csz==0 path (no helper sub-sharding)
 * for 4 active ranks. At count=2048: pR=1024, xg=512, csz=align(512/5)=0, so
 * the recursive-halving step exchanges the whole half directly with the
 * partner. The send and recv of that swap MUST share one ncclGroup; a prior bug
 * split them across phases (send in phase 0, recv in phase 1) and deadlocked on
 * GPU.
 */
TEST_F(
    ShardedRelayMultiGroupAllReduceTest,
    Correctness_4Active_2Groups_TinyCsz0_InPlace) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int nGroups = 2;
  const int nActiveRanksPerGroup = 4;
  const size_t count = 2048; // small enough that relay csz == 0
  const size_t dataBytes = count * sizeof(int32_t);

  TwoGroupFourActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* buffers[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(&buffers[g], dataBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMalloc(&buffers[g], helperBufferSize));
    }
  }

  barrierSyncOn(buffers[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      std::vector<int32_t> hostData(count, myActiveIndex + 1);
      HIPCHECK_TEST(hipMemcpy(
          buffers[g], hostData.data(), dataBytes, hipMemcpyHostToDevice));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMemset(buffers[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t counts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = buffers[g];
    recvPtrs[g] = buffers[g];
    counts[g] = count;
  }

  ncclResult_t result = callAllReduceCompat(
      sendPtrs,
      recvPtrs,
      counts,
      ncclInt32,
      ncclSum,
      this->commFor(nActiveRanksPerGroup),
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  // SUM across 4 active ranks = 1+2+3+4 = 10.
  verifyDeviceBufferEquals(
      buffers[myActiveGroup],
      count,
      10,
      myActiveGroup,
      "4-active tiny csz==0 SUM mismatch (expected 10)");

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(buffers[g]));
  }
}

TEST_F(
    ShardedRelayMultiGroupAllReduceTest,
    Correctness_4Active_2Groups_OutOfPlace) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int nGroups = 2;
  const int nActiveRanksPerGroup = 4;
  const size_t dataBytes = 64ULL * 1024 * 1024;
  const size_t count = dataBytes / sizeof(int32_t);

  TwoGroupFourActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], dataBytes));
      HIPCHECK_TEST(hipMalloc(&recvBuffs[g], dataBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
      recvBuffs[g] = sendBuffs[g];
    }
  }

  barrierSyncOn(recvBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      std::vector<int32_t> hostData(count, myActiveIndex + 1);
      HIPCHECK_TEST(hipMemcpy(
          sendBuffs[g], hostData.data(), dataBytes, hipMemcpyHostToDevice));
      HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, dataBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t counts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
    counts[g] = count;
  }

  ncclResult_t result = callAllReduceCompat(
      sendPtrs,
      recvPtrs,
      counts,
      ncclInt32,
      ncclSum,
      this->commFor(nActiveRanksPerGroup),
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  verifyDeviceBufferEquals(
      recvBuffs[myActiveGroup],
      count,
      10,
      myActiveGroup,
      "4-active out-of-place SUM mismatch (expected 10)");

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipFree(recvBuffs[g]));
    }
  }
}

TEST_F(ShardedRelayMultiGroupAllReduceTest, Correctness_4Active_2Groups_Avg) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int nGroups = 2;
  const int nActiveRanksPerGroup = 4;
  const size_t dataBytes = 64ULL * 1024 * 1024;
  const size_t count = dataBytes / sizeof(int32_t);

  TwoGroupFourActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* buffers[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(&buffers[g], dataBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMalloc(&buffers[g], helperBufferSize));
    }
  }

  barrierSyncOn(buffers[0]);

  // Values 2,4,6,8 -> sum 20, AVG (divisor=4) = 5 (exact in int32). A wrong
  // divisor (e.g. 2) would give 10, detecting that AVG uses nActiveRanks.
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      std::vector<int32_t> hostData(count, (myActiveIndex + 1) * 2);
      HIPCHECK_TEST(hipMemcpy(
          buffers[g], hostData.data(), dataBytes, hipMemcpyHostToDevice));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMemset(buffers[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t counts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = buffers[g];
    recvPtrs[g] = buffers[g];
    counts[g] = count;
  }

  ncclResult_t result = callAllReduceCompat(
      sendPtrs,
      recvPtrs,
      counts,
      ncclInt32,
      ncclAvg,
      this->commFor(nActiveRanksPerGroup),
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  verifyDeviceBufferEquals(
      buffers[myActiveGroup],
      count,
      5,
      myActiveGroup,
      "4-active AVG mismatch (expected 5 = (2+4+6+8)/4)");

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(buffers[g]));
  }
}

// Single-group (nGroups=1) A=4 allreduce: ranks {0,1,2,3} active, {4,5,6,7}
// helpers. Exercises the A=4 path without the multi-group fusion the
// 4Active_2Groups tests cover.
TEST_F(
    ShardedRelayMultiGroupAllReduceTest,
    Correctness_4Active_SingleGroup_InPlace) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  runAllReduceA4SingleGroup(ncclSum);
}

TEST_F(
    ShardedRelayMultiGroupAllReduceTest,
    Correctness_4Active_SingleGroup_Avg) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  runAllReduceA4SingleGroup(ncclAvg);
}

TEST_F(
    ShardedRelayMultiGroupAllReduceTest,
    Correctness_4Active_PartialGroupsZeroCount) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int nGroups = 2;
  const int nActiveRanksPerGroup = 4;
  const size_t dataBytes = 16ULL * 1024 * 1024;
  const size_t count = dataBytes / sizeof(int32_t);

  // Group 0 has data; group 1 has count = 0 (skipped).
  const size_t counts[nGroups] = {count, 0};

  TwoGroupFourActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  const size_t placeholderBytes = sizeof(int32_t);
  int32_t* buffers[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0) {
      HIPCHECK_TEST(hipMalloc(&buffers[g], placeholderBytes));
      HIPCHECK_TEST(hipMemset(buffers[g], 0, placeholderBytes));
      continue;
    }
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(&buffers[g], dataBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMalloc(&buffers[g], helperBufferSize));
    }
  }

  barrierSyncOn(buffers[0]);

  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0) {
      continue;
    }
    if (g == myActiveGroup) {
      std::vector<int32_t> hostData(count, myActiveIndex + 1);
      HIPCHECK_TEST(hipMemcpy(
          buffers[g], hostData.data(), dataBytes, hipMemcpyHostToDevice));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMemset(buffers[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = buffers[g];
    recvPtrs[g] = buffers[g];
  }

  ncclResult_t result = callAllReduceCompat(
      sendPtrs,
      recvPtrs,
      counts,
      ncclInt32,
      ncclSum,
      this->commFor(nActiveRanksPerGroup),
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess)
      << "4-active allreduce failed with a partial count=0 group";
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  if (myActiveGroup == 0) { // group 0 had data
    verifyDeviceBufferEquals(
        buffers[myActiveGroup],
        count,
        10,
        myActiveGroup,
        "4-active partial-zero SUM mismatch (expected 10)");
  }

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(buffers[g]));
  }
}

TEST_F(
    ShardedRelayMultiGroupAllReduceTest,
    Correctness_Phase2C02_BFloat16_A2_Independent6MiBBoundary) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  constexpr size_t thresholdCount = (6ULL * 1024 * 1024) / sizeof(uint16_t);
  const int activeRanks[2] = {0, 1};
  const int* allActiveRanks[1] = {activeRanks};
  const std::vector<Bfloat16AllReduceCase> cases = {
      {thresholdCount - 1, ncclSum, "A=2 6 MiB below-threshold BF16 SUM"},
      {thresholdCount - 1, ncclAvg, "A=2 6 MiB below-threshold BF16 AVG"},
      {thresholdCount, ncclSum, "A=2 6 MiB at-threshold BF16 SUM"},
      {thresholdCount, ncclAvg, "A=2 6 MiB at-threshold BF16 AVG"},
      {thresholdCount + 1, ncclSum, "A=2 6 MiB above-threshold BF16 SUM"},
  };

  runBfloat16AllReduceCases(allActiveRanks, 2, 1, cases);
}

TEST_F(
    ShardedRelayMultiGroupAllReduceTest,
    Correctness_Phase2C02_BFloat16_A4_PermutedFused2MiBBoundary) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  constexpr size_t thresholdCount = (2ULL * 1024 * 1024) / sizeof(uint16_t);
  PermutedTwoGroupFourActiveRanks groupConfig;
  const std::vector<Bfloat16AllReduceCase> cases = {
      {thresholdCount - 4,
       ncclSum,
       "A=4 2 MiB below-threshold permuted patterned SUM",
       true},
      {thresholdCount,
       ncclSum,
       "A=4 2 MiB at-threshold permuted patterned SUM",
       true},
      {thresholdCount + 4,
       ncclSum,
       "A=4 2 MiB above-threshold permuted patterned SUM",
       true},
  };

  runBfloat16AllReduceCases(groupConfig.allActiveRanks, 4, 2, cases);
}

// One-shot IPC coverage. Below kRelayOneShotMaxBytes (1 MiB of per-rank count)
// the allreduce runs a single kernel that pushes each rank's whole buffer into
// every peer's staging, handshakes per block, and reduces all A contributions
// -- replacing the group-plus-reduce pair.
//
// useShardPattern is the point: it fingerprints the contribution by active
// index AND element offset. The constant fill the other cases use cannot see an
// offset error inside a staging slot, because every element of a block holds
// the same value, and that offset is exactly what the vectorized bulk / scalar
// tail split could get wrong. 65535 is deliberately not a multiple of
// 16/sizeof(bf16), so it drives the tail and the misaligned fallback rather
// than the vector path.
TEST_F(
    ShardedRelayMultiGroupAllReduceTest,
    Correctness_OneShot_A2_SingleGroup_ShardPattern) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int activeRanks[2] = {0, 1};
  const int* allActiveRanks[1] = {activeRanks};
  const std::vector<Bfloat16AllReduceCase> cases = {
      {65536, ncclSum, "A=2 one-shot BF16 SUM", true},
      {65536, ncclAvg, "A=2 one-shot BF16 AVG", true},
      {65535, ncclSum, "A=2 one-shot BF16 SUM unaligned tail", true},
  };

  runBfloat16AllReduceCases(allActiveRanks, 2, 1, cases);
}

TEST_F(
    ShardedRelayMultiGroupAllReduceTest,
    Correctness_OneShot_A4_SingleGroup_ShardPattern) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int activeRanks[4] = {0, 1, 2, 3};
  const int* allActiveRanks[1] = {activeRanks};
  const std::vector<Bfloat16AllReduceCase> cases = {
      {65536, ncclSum, "A=4 one-shot BF16 SUM", true},
      {65536, ncclAvg, "A=4 one-shot BF16 AVG", true},
      // 65532 is divisible by A (the A>2 allreduce rejects counts that are
      // not) yet not by 8, so rc*sizeof(bf16) is not a multiple of 16 and the
      // scalar fallback runs. 65535 would be rejected as invalid input.
      {65532, ncclSum, "A=4 one-shot BF16 SUM unaligned tail", true},
  };

  runBfloat16AllReduceCases(allActiveRanks, 4, 1, cases);
}

TEST_F(
    ShardedRelayMultiGroupAllReduceTest,
    Correctness_Phase2C02_BFloat16_A4_Independent1MiBBoundary) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  constexpr size_t thresholdCount = (1ULL * 1024 * 1024) / sizeof(uint16_t);
  const int activeRanks[4] = {0, 1, 2, 3};
  const int* allActiveRanks[1] = {activeRanks};
  const std::vector<Bfloat16AllReduceCase> cases = {
      {thresholdCount - 4, ncclSum, "A=4 1 MiB below-threshold BF16 SUM"},
      {thresholdCount, ncclSum, "A=4 1 MiB at-threshold BF16 SUM"},
      {thresholdCount + 4, ncclSum, "A=4 1 MiB above-threshold BF16 SUM"},
  };

  runBfloat16AllReduceCases(allActiveRanks, 4, 1, cases);
}

TEST_F(
    ShardedRelayMultiGroupAllReduceTest,
    Z_BusBW_4Active_2Groups_InPlace_1GB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int nGroups = 2;
  const int nActiveRanksPerGroup = 4;
  const size_t dataBytes = 1ULL * 1024 * 1024 * 1024; // 1 GB per group
  const size_t count = dataBytes / sizeof(int32_t);
  const int nIters = 20;

  TwoGroupFourActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;

  int32_t* buffers[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(&buffers[g], dataBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMalloc(&buffers[g], helperBufferSize));
    }
  }

  barrierSyncOn(buffers[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMemset(buffers[g], 1, dataBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * dataBytes;
      HIPCHECK_TEST(hipMemset(buffers[g], 0, helperBufferSize));
    }
  }
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t counts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = buffers[g];
    recvPtrs[g] = buffers[g];
    counts[g] = count;
  }

  hipEvent_t startEvent, stopEvent;
  HIPCHECK_TEST(hipEventCreate(&startEvent));
  HIPCHECK_TEST(hipEventCreate(&stopEvent));

  float bestTimeMs = std::numeric_limits<float>::max();
  float totalTimeMs = 0.0f;

  for (int iter = 0; iter < nIters; iter++) {
    HIPCHECK_TEST(hipEventRecord(startEvent, this->stream));
    ncclResult_t result = callAllReduceCompat(
        sendPtrs,
        recvPtrs,
        counts,
        ncclInt32,
        ncclSum,
        this->commFor(nActiveRanksPerGroup),
        this->stream,
        allActiveRanks,
        nActiveRanksPerGroup,
        nGroups);
    ASSERT_EQ(result, ncclSuccess);
    HIPCHECK_TEST(hipEventRecord(stopEvent, this->stream));
    HIPCHECK_TEST(hipEventSynchronize(stopEvent));

    float elapsedMs = 0.0f;
    HIPCHECK_TEST(hipEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    if (elapsedMs < bestTimeMs) {
      bestTimeMs = elapsedMs;
    }
    totalTimeMs += elapsedMs;
  }

  if (this->globalRank == 0) {
    ShardedBandwidthResult bwResult = calculateMultiGroupAggregateBandwidth(
        dataBytes, bestTimeMs, nActiveRanksPerGroup, nGroups);
    printMultiGroupBandwidthResults(
        "4-Active 2-Group IN-PLACE 1GB",
        dataBytes,
        this->numRanks,
        nGroups,
        nActiveRanksPerGroup,
        bwResult,
        true);
  }

  HIPCHECK_TEST(hipEventDestroy(startEvent));
  HIPCHECK_TEST(hipEventDestroy(stopEvent));
  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(buffers[g]));
  }
}

// Pins the size-adaptive routing crossovers. Routing is internal to the
// collective and derived only from the message size, so the only thing a test
// can meaningfully assert is that the mapping matches the measured crossovers
// the design specifies. Each block states the intended threshold independently
// and checks below / at / above it, so moving a threshold in the implementation
// fails here and forces a deliberate decision rather than silently reshaping
// the routing. This exercises the selector only -- no collective is issued, so
// it costs nothing on the GPU.
TEST_F(ShardedRelayMultiGroupAllReduceTest, Routing_SizeAdaptiveCrossovers) {
  struct Crossover {
    const char* name;
    int nActiveRanksPerGroup;
    int nGroups;
    size_t thresholdBytes;
    rcclx::relay::AllReduceRoute atOrAbove;
  };
  // A==2: 2 MB fused / 2 MB independent. A>2: 2 MB fused / 1 MB independent.
  const Crossover crossovers[] = {
      {"A2 fused", 2, 4, 2ULL << 20, rcclx::relay::AllReduceRoute::A2Relay},
      {"A2 independent",
       2,
       1,
       2ULL << 20,
       rcclx::relay::AllReduceRoute::A2Relay},
      {"A4 fused", 4, 2, 2ULL << 20, rcclx::relay::AllReduceRoute::FlatOffload},
      {"A4 independent",
       4,
       1,
       1ULL << 20,
       rcclx::relay::AllReduceRoute::FlatOffload},
  };

  for (const auto& crossover : crossovers) {
    const size_t thresholdCount = crossover.thresholdBytes / sizeof(int32_t);
    const size_t belowCount = thresholdCount - 1;
    const size_t aboveCount = thresholdCount + 1;

    // The mapping keys off the largest per-group count, so filling every group
    // with the same count also pins that the maximum is what is consulted.
    const std::vector<size_t> below(crossover.nGroups, belowCount);
    const std::vector<size_t> at(crossover.nGroups, thresholdCount);
    const std::vector<size_t> above(crossover.nGroups, aboveCount);

    SCOPED_TRACE(crossover.name);
    expectAllReduceRoute(
        crossover.nActiveRanksPerGroup,
        crossover.nGroups,
        below.data(),
        sizeof(int32_t),
        rcclx::relay::AllReduceRoute::PureDirect);
    expectAllReduceRoute(
        crossover.nActiveRanksPerGroup,
        crossover.nGroups,
        at.data(),
        sizeof(int32_t),
        crossover.atOrAbove);
    expectAllReduceRoute(
        crossover.nActiveRanksPerGroup,
        crossover.nGroups,
        above.data(),
        sizeof(int32_t),
        crossover.atOrAbove);
  }
}

// ===========================================================================
// LOW PRECISION (fp8e4m3 wire format)
// ===========================================================================
//
// Every case here asserts that low precision actually ENGAGED or actually
// DECLINED, via the counters, rather than trusting the flag. The gate declines
// silently on four independent grounds, so an "LP" run that quietly fell back
// to full precision produces exactly the numbers a passing LP run produces --
// the counter is the only thing that tells them apart.
//
// The comparators stay EXACT. Each active rank fills its buffer with a single
// constant, so every 128-element wire block has that constant as its absmax;
// the power-of-two normalization target then makes quantize/dequantize an
// identity, and a sum of equal values behaves the same way. These cases are
// therefore a genuine detector for a wrong scale, a wrong block boundary or a
// dropped scale. Do not loosen them.
class ShardedRelayAllReduceLowPrecisionTest
    : public ShardedRelayMultiGroupAllReduceTest {
 protected:
  static constexpr int kGroups = 4;
  static constexpr int kActive = 2;

  // 2 Mi elements = 8 MiB in fp32, comfortably above the low-precision size
  // threshold and a whole number of 128-element blocks.
  static constexpr size_t kLpCount = 2ULL * 1024 * 1024;

  struct Buffers {
    std::vector<void*> mem;
    const void* sendPtrs[kGroups];
    void* recvPtrs[kGroups];
    size_t counts[kGroups];
    int myActiveGroup{0};
  };

  // In-place buffers filled so the active group's contribution is a constant
  // (activeIndex + 1), which makes the expected allreduce sum exact.
  template <typename T>
  void makeBuffers(size_t count, T contribution, Buffers& b) {
    b = Buffers{};
    b.myActiveGroup = this->globalRank / kActive;
    b.mem.resize(kGroups);
    for (int g = 0; g < kGroups; g++) {
      const size_t elems =
          (g == b.myActiveGroup) ? count : static_cast<size_t>(kActive) * count;
      HIPCHECK_TEST(hipMalloc(&b.mem[g], elems * sizeof(T)));
      if (g == b.myActiveGroup) {
        const std::vector<T> host(count, contribution);
        HIPCHECK_TEST(hipMemcpy(
            b.mem[g], host.data(), count * sizeof(T), hipMemcpyHostToDevice));
      } else {
        HIPCHECK_TEST(hipMemset(b.mem[g], 0, elems * sizeof(T)));
      }
      b.sendPtrs[g] = b.mem[g];
      b.recvPtrs[g] = b.mem[g];
      b.counts[g] = count;
    }
  }

  void freeBuffers(Buffers& b) {
    for (void* p : b.mem) {
      HIPCHECK_TEST(hipFree(p));
    }
  }

  template <typename T>
  void expectAllEqual(const Buffers& b, size_t count, T want) {
    std::vector<T> got(count);
    HIPCHECK_TEST(hipMemcpy(
        got.data(),
        b.mem[b.myActiveGroup],
        count * sizeof(T),
        hipMemcpyDeviceToHost));
    size_t reported = 0;
    for (size_t i = 0; i < count && reported < 8; i++) {
      if (got[i] != want) {
        reported++;
        ADD_FAILURE() << "element " << i << ": got " << got[i] << ", want "
                      << want;
      }
    }
  }

  ncclResult_t call(const Buffers& b, ncclDataType_t dt, int lowPrecision) {
    return callAllReduceCompat(
        b.sendPtrs,
        b.recvPtrs,
        b.counts,
        dt,
        ncclSum,
        this->commFor(kActive),
        this->stream,
        Standard4GroupActiveRanks().allActiveRanks,
        kActive,
        kGroups,
        lowPrecision);
  }
};

TEST_F(ShardedRelayAllReduceLowPrecisionTest, ConstantBlocksAreBitExact) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  const float contribution =
      static_cast<float>(this->globalRank % kActive) + 1.0f;
  Buffers b;
  makeBuffers<float>(kLpCount, contribution, b);
  barrierSyncOn(static_cast<int32_t*>(b.mem[0]));

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(call(b, ncclFloat32, /*lowPrecision=*/1), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  // 1.0 + 2.0 across the two active ranks, identical in every element.
  expectAllEqual<float>(b, kLpCount, 3.0f);
  EXPECT_GT(rcclx::relay::lpEngageCount(), 0u)
      << "low precision never engaged, so this case proved nothing";
  EXPECT_EQ(rcclx::relay::lpDeclineCount(), 0u);
  freeBuffers(b);
}

TEST_F(ShardedRelayAllReduceLowPrecisionTest, DeclinesOnUnsupportedDtype) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  // ncclInt32 is not a low-precision dtype, so the flag must fall through to
  // full precision and the answer must be exactly the full-precision answer.
  Buffers b;
  makeBuffers<int32_t>(kLpCount, 1, b);
  barrierSyncOn(static_cast<int32_t*>(b.mem[0]));

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(call(b, ncclInt32, /*lowPrecision=*/1), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  expectAllEqual<int32_t>(b, kLpCount, kActive);
  EXPECT_EQ(rcclx::relay::lpEngageCount(), 0u);
  EXPECT_GT(rcclx::relay::lpDeclineCount(rcclx::relay::LpDecline::Dtype), 0u);
  freeBuffers(b);
}

TEST_F(
    ShardedRelayAllReduceLowPrecisionTest,
    DeclinesOnCountThatIsNotWholeBlocks) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  // One element past a whole number of 128-element blocks. bytes() is only
  // additive on whole blocks, so the gate must refuse rather than produce
  // offsets its peers compute differently.
  const size_t count = kLpCount + 1;
  const float contribution =
      static_cast<float>(this->globalRank % kActive) + 1.0f;
  Buffers b;
  makeBuffers<float>(count, contribution, b);
  barrierSyncOn(static_cast<int32_t*>(b.mem[0]));

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(call(b, ncclFloat32, /*lowPrecision=*/1), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  expectAllEqual<float>(b, count, 3.0f);
  EXPECT_EQ(rcclx::relay::lpEngageCount(), 0u);
  EXPECT_GT(
      rcclx::relay::lpDeclineCount(rcclx::relay::LpDecline::Alignment), 0u);
  freeBuffers(b);
}

TEST_F(ShardedRelayAllReduceLowPrecisionTest, InterleavesWithFullPrecision) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  // The behaviour an env var could not express, and the whole point of making
  // this a per-call argument: two calls on the SAME communicator, one low
  // precision and one not, with no state leaking from the first into the
  // second.
  const float contribution =
      static_cast<float>(this->globalRank % kActive) + 1.0f;
  Buffers b;
  makeBuffers<float>(kLpCount, contribution, b);
  barrierSyncOn(static_cast<int32_t*>(b.mem[0]));

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(call(b, ncclFloat32, /*lowPrecision=*/1), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));
  expectAllEqual<float>(b, kLpCount, 3.0f);
  ASSERT_GT(rcclx::relay::lpEngageCount(), 0u);

  // Re-seed and run the same shape at full precision.
  freeBuffers(b);
  makeBuffers<float>(kLpCount, contribution, b);
  barrierSyncOn(static_cast<int32_t*>(b.mem[0]));
  const uint64_t engagedBefore = rcclx::relay::lpEngageCount();
  ASSERT_EQ(call(b, ncclFloat32, /*lowPrecision=*/0), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));
  expectAllEqual<float>(b, kLpCount, 3.0f);
  EXPECT_EQ(rcclx::relay::lpEngageCount(), engagedBefore)
      << "a full-precision call must not engage low precision";
  freeBuffers(b);
}

TEST_F(ShardedRelayAllReduceLowPrecisionTest, SingleGroupPipelinedIsBitExact) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  // The 4-group fixture above cannot reach the pipelined schedule:
  // relayPipelineTiles() returns 1 whenever nGroups != 1. So this case is
  // single-group, at a size large enough that the depth selector picks T > 1 --
  // asserted below rather than assumed, because a size that quietly failed to
  // pipeline would silently re-test the schedule the other cases already cover.
  const int nGroups = 1;
  const size_t count =
      16ULL * 1024 * 1024; // 64 MiB fp32, a whole number of blocks

  ASSERT_GT(
      rcclx::relay::relayPipelineTiles(
          nGroups,
          rcclx::relay::relayShapeA2(this->numRanks - kActive),
          count,
          sizeof(float)),
      1)
      << "chosen count does not pipeline, so this case would not exercise the "
         "pipelined schedule";

  const int activeRanks[] = {0, 1};
  const int* allActiveRanks[] = {activeRanks};
  const bool isActive = (this->globalRank == 0 || this->globalRank == 1);
  const float contribution =
      isActive ? static_cast<float>(this->globalRank % kActive) + 1.0f : 0.0f;

  const size_t elems = isActive ? count : static_cast<size_t>(kActive) * count;
  float* buff = nullptr;
  HIPCHECK_TEST(hipMalloc(&buff, elems * sizeof(float)));
  barrierSyncOn(reinterpret_cast<int32_t*>(buff));
  if (isActive) {
    const std::vector<float> host(count, contribution);
    HIPCHECK_TEST(hipMemcpy(
        buff, host.data(), count * sizeof(float), hipMemcpyHostToDevice));
  } else {
    HIPCHECK_TEST(hipMemset(buff, 0, elems * sizeof(float)));
  }

  const void* sendPtrs[1] = {buff};
  void* recvPtrs[1] = {buff};
  size_t counts[1] = {count};

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(
      callAllReduceCompat(
          sendPtrs,
          recvPtrs,
          counts,
          ncclFloat32,
          ncclSum,
          this->commFor(kActive),
          this->stream,
          allActiveRanks,
          kActive,
          nGroups,
          /*lowPrecision=*/1),
      ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  if (isActive) {
    std::vector<float> got(count);
    HIPCHECK_TEST(hipMemcpy(
        got.data(), buff, count * sizeof(float), hipMemcpyDeviceToHost));
    size_t reported = 0;
    for (size_t i = 0; i < count && reported < 8; i++) {
      if (got[i] != 3.0f) {
        reported++;
        ADD_FAILURE() << "element " << i << ": got " << got[i] << ", want 3";
      }
    }
    EXPECT_GT(rcclx::relay::lpEngageCount(), 0u)
        << "low precision never engaged, so this case proved nothing";
  }
  HIPCHECK_TEST(hipFree(buff));
}

// ---------------------------------------------------------------------------
// 4 active (flat offload schedule)
// ---------------------------------------------------------------------------
class ShardedRelayAllReduceFlatLowPrecisionTest
    : public ShardedRelayMultiGroupAllReduceTest {
 protected:
  static constexpr int kGroups = 2;
  static constexpr int kActive = 4;
  // A multiple of A*128 == 512, which the flat schedule requires: it splits its
  // direct region into A per-owner shards of pD/A, and pD being a whole number
  // of wire blocks does not make pD/A one.
  static constexpr size_t kCount = 2ULL * 1024 * 1024;

  // Rank r is active for group r/4, helper for the other.
  int myGroup() const {
    return this->globalRank / kActive;
  }

  ncclResult_t
  run(void** mem, size_t count, ncclDataType_t dt, int lowPrecision) {
    static const int g0[] = {0, 1, 2, 3};
    static const int g1[] = {4, 5, 6, 7};
    const int* allActiveRanks[] = {g0, g1};
    const void* sendPtrs[kGroups];
    void* recvPtrs[kGroups];
    size_t counts[kGroups];
    for (int g = 0; g < kGroups; g++) {
      sendPtrs[g] = mem[g];
      recvPtrs[g] = mem[g];
      counts[g] = count;
    }
    return callAllReduceCompat(
        sendPtrs,
        recvPtrs,
        counts,
        dt,
        ncclSum,
        this->commFor(kActive),
        this->stream,
        allActiveRanks,
        kActive,
        kGroups,
        lowPrecision);
  }

  void alloc(void** mem, size_t count) {
    for (int g = 0; g < kGroups; g++) {
      const size_t elems =
          (g == myGroup()) ? count : static_cast<size_t>(kActive) * count;
      HIPCHECK_TEST(hipMalloc(&mem[g], elems * sizeof(float)));
      if (g == myGroup()) {
        // Contribution 1..4 by active index, so the sum is exactly 10.
        const std::vector<float> host(
            count, static_cast<float>(this->globalRank % kActive) + 1.0f);
        HIPCHECK_TEST(hipMemcpy(
            mem[g], host.data(), count * sizeof(float), hipMemcpyHostToDevice));
      } else {
        HIPCHECK_TEST(hipMemset(mem[g], 0, elems * sizeof(float)));
      }
    }
  }

  void expectSumIsTen(void** mem, size_t count) {
    std::vector<float> got(count);
    HIPCHECK_TEST(hipMemcpy(
        got.data(),
        mem[myGroup()],
        count * sizeof(float),
        hipMemcpyDeviceToHost));
    size_t reported = 0;
    for (size_t i = 0; i < count && reported < 8; i++) {
      if (got[i] != 10.0f) {
        reported++;
        ADD_FAILURE() << "element " << i << ": got " << got[i] << ", want 10";
      }
    }
  }

  void freeAll(void** mem) {
    for (int g = 0; g < kGroups; g++) {
      HIPCHECK_TEST(hipFree(mem[g]));
    }
  }
};

TEST_F(ShardedRelayAllReduceFlatLowPrecisionTest, ConstantBlocksAreBitExact) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  void* mem[kGroups];
  alloc(mem, kCount);
  barrierSyncOn(static_cast<int32_t*>(mem[0]));

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(run(mem, kCount, ncclFloat32, /*lowPrecision=*/1), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  // 1+2+3+4 == 10, exactly, in every element. This covers both of the flat
  // schedule's regions at once: the direct reduce-scatter/all-gather over
  // [0, pD) and the helper-reduced offload region over [pD, count).
  expectSumIsTen(mem, kCount);
  EXPECT_GT(rcclx::relay::lpEngageCount(), 0u)
      << "low precision never engaged, so this case proved nothing";
  freeAll(mem);
}

TEST_F(
    ShardedRelayAllReduceFlatLowPrecisionTest,
    DeclinesOnCountAlignedToBlocksButNotToShards) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  // The regression test for the reason the gate needs a per-call alignment:
  // this count IS a whole number of 128-element wire blocks, so the 2-active
  // gate would admit it, but it is NOT a multiple of A*128, so pD/A would be a
  // fractional block and every offset in the direct region would be wrong.
  const size_t count =
      kCount + rcclx::relay::kLpBlockElems; // + 128: still blocks, not shards
  ASSERT_EQ(count % 128u, 0u);
  ASSERT_NE(count % (kActive * 128u), 0u);

  void* mem[kGroups];
  alloc(mem, count);
  barrierSyncOn(static_cast<int32_t*>(mem[0]));

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(run(mem, count, ncclFloat32, /*lowPrecision=*/1), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  // Full precision, so still exactly 10.
  expectSumIsTen(mem, count);
  EXPECT_EQ(rcclx::relay::lpEngageCount(), 0u);
  EXPECT_GT(
      rcclx::relay::lpDeclineCount(rcclx::relay::LpDecline::Alignment), 0u);
  freeAll(mem);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
