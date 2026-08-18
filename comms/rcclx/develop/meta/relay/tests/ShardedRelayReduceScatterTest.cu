// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * Unit tests for Fused Multi-Group Sharded Relay Reduce-Scatter
 *
 * Reduce-scatter analogue of ShardedRelayAllReduceTest.cu. Tests the
 * phase-synchronized execution of multiple sharded relay reduce-scatters with
 * passthrough-at-helper design. All groups execute in lockstep phases; helpers
 * forward data without local reduction.
 *
 * Reduce-Scatter Semantics (per group, 2 active ranks):
 * =====================================================
 * Each active rank's sendBuff holds nActiveRanksPerGroup x recvCount elements:
 * block[i] is the slice destined for active index i. Each active recvBuff holds
 * recvCount elements and receives sum_over_active_ranks(sendBuff[block i]).
 *
 * To verify correct block selection, each active rank fills block j with a
 * distinct value blockFillValue(myActiveIndex, j). The expected output for the
 * rank with active index m is then:
 *   blockFillValue(0, m) + blockFillValue(1, m)
 * A wrong block offset in the implementation produces a detectable mismatch.
 *
 * Algorithm Design (Phase-Synchronized, Passthrough Helpers):
 * ===========================================================
 * Phase 1 (active->helpers): Both active ranks send chunks of their sendBlock
 *         (the block destined for the OTHER active rank) to helpers. Helpers
 *         receive into two slots (slot a = data from active rank a).
 * Phase 2 (helpers->active): each helper forwards slot 0 -> a1, slot 1 -> a0.
 * Phase 3 (active reduce): add relay scratch into the seeded output block.
 * Phase 4 (active<->active): direct exchange of the last chunk.
 * Phase 5 (active reduce): final reduction of the direct chunk.
 *
 * Buffer Requirements:
 * ====================
 * ACTIVE ranks: sendBuff holds 2 x recvCount elements. recvBuff holds
 * recvCount elements; in-place is recvBuff == sendBuff + ownBlockOffset.
 * HELPER ranks: two-slot buffer of nActiveRanks x chunkSize elements
 * (chunkSize derived from recvCount). Each helper group MUST have its own
 * buffer (no aliasing across groups).
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
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

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

// Aggregate bandwidth for multi-group reduce-scatter. The size metric is the
// per-group OUTPUT (recvCount) bytes. Bus BW for an n-rank reduce-scatter uses
// the (n-1)/n factor (for 2 ranks this is 0.5).
ShardedBandwidthResult calculateMultiGroupAggregateBandwidth(
    size_t recvBytesPerGroup,
    double elapsedMs,
    int numActiveRanks,
    int numGroups) {
  ShardedBandwidthResult result;
  double elapsedSec = elapsedMs / 1000.0;
  double totalDataSizeGB = static_cast<double>(recvBytesPerGroup) * numGroups /
      (1024.0 * 1024.0 * 1024.0);
  result.algoBW_GBps = totalDataSizeGB / elapsedSec;
  result.busBW_GBps =
      (numActiveRanks - 1.0) / numActiveRanks * totalDataSizeGB / elapsedSec;
  result.latency_us = elapsedMs * 1000.0;
  return result;
}

void printMultiGroupBandwidthResults(
    const std::string& testName,
    size_t recvBytesPerGroup,
    int numRanks,
    int numGroups,
    int activeRanksPerGroup,
    const ShardedBandwidthResult& aggregateResult,
    bool isInPlace) {
  double dataSizePerGroupGB =
      static_cast<double>(recvBytesPerGroup) / (1024.0 * 1024.0 * 1024.0);

  double totalDataSizeGB = dataSizePerGroupGB * numGroups;

  double perGroupAlgoBW = aggregateResult.algoBW_GBps / numGroups;
  double perGroupBusBW = aggregateResult.busBW_GBps / numGroups;

  std::cout << "\n";
  std::cout << "====================================================\n";
  std::cout << "Multi-Group Sharded Relay Reduce-Scatter: " << testName << "\n";
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
  std::cout << "  Output Size per Group: " << dataSizePerGroupGB << " GB\n";
  std::cout << "  Total Output (all groups): " << totalDataSizeGB << " GB\n";
  std::cout << "  Recv Count/Group:      "
            << (recvBytesPerGroup / sizeof(int32_t)) << "\n";
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
// sharded-relay reduce-scatter entry point. The active group's input holds
// nActiveRanks x recvCount elements; its output holds recvCount elements.
static ncclResult_t callReduceScatterCompat(
    const void* const* sendPtrs,
    void* const* recvPtrs,
    const size_t* recvCounts,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    hipStream_t stream,
    const int* const* allActiveRanks,
    int nActiveRanksPerGroup,
    int nGroups) {
  return ncclShardedRelayMultiGroupReduceScatter(
      sendPtrs,
      recvPtrs,
      recvCounts,
      datatype,
      op,
      comm,
      stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
}

class ShardedRelayMultiGroupReduceScatterTest : public ::testing::Test {
 public:
  ShardedRelayMultiGroupReduceScatterTest() = default;

  void SetUp() override {
    int localSize;
    std::tie(this->localRank, this->globalRank, this->numRanks, localSize) =
        getTcpStoreOrMpiInfo();
    bool isServer = (this->globalRank == 0);
    if (checkTcpStoreEnv()) {
      server = createTcpStore(isServer);
    } else if (isServer) {
      server = createTcpStore(true);
    }
    this->comm = createNcclComm(
        this->globalRank,
        this->numRanks,
        this->localRank,
        false,
        nullptr,
        server.get());
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    if (server && checkTcpStoreEnv()) {
      finalizeNcclComm(this->globalRank, server.get());
    }
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
    server.reset();
  }

  // Standard 8-rank, 4-group, 2-active-per-group sparse parallelism layout.
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

  // Distinct per-(activeIndex, block) fill value so that a wrong block offset
  // in the implementation produces a detectable mismatch.
  static int32_t blockFillValue(int activeIndex, int blockIndex) {
    return (activeIndex + 1) * 10 + (blockIndex + 1);
  }

  // Expected reduce-scatter output for the rank with the given active index:
  //   recvBuff[i] = sum over all active ranks r of block[myActiveIndex] from r
  //              = sum_r blockFillValue(r, myActiveIndex)
  static int32_t expectedReduceScatterSum(
      int myActiveIndex,
      int nActiveRanks = 2) {
    int32_t sum = 0;
    for (int r = 0; r < nActiveRanks; r++) {
      sum += blockFillValue(r, myActiveIndex);
    }
    return sum;
  }

  // Initialize an active rank's send buffer (nActiveRanks x recvCount elements)
  // so that block j is uniformly filled with blockFillValue(myActiveIndex, j).
  void initActiveSendBuffer(
      int32_t* deviceBuf,
      size_t recvCount,
      int myActiveIndex,
      int nActiveRanks = 2) {
    std::vector<int32_t> host(static_cast<size_t>(nActiveRanks) * recvCount);
    for (int j = 0; j < nActiveRanks; j++) {
      int32_t v = blockFillValue(myActiveIndex, j);
      std::fill_n(
          host.data() + static_cast<size_t>(j) * recvCount, recvCount, v);
    }
    HIPCHECK_TEST(hipMemcpy(
        deviceBuf,
        host.data(),
        static_cast<size_t>(nActiveRanks) * recvCount * sizeof(int32_t),
        hipMemcpyHostToDevice));
  }

  // Run a 1-element ncclAllReduce on `scratchBuffer` to act as a cross-rank
  // barrier.
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
        this->comm,
        this->stream));
    HIPCHECK_TEST(hipStreamSynchronize(this->stream));
    HIPCHECK_TEST(hipFree(barrierScratch));
  }

  // Copy `count` int32_t elements from `deviceBuffer` to host and verify they
  // all equal `expectedValue`. Returns the number of mismatches (0 on success).
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

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  ncclComm_t comm;
  cudaStream_t stream;
  std::unique_ptr<c10d::TCPStore> server{nullptr};
};

/**
 * Test: Multi-Group Correctness with 4 groups (IN-PLACE)
 *
 * In-place reduce-scatter: recvBuff aliases sendBuff + ownBlockOffset.
 * Active sendBuff holds 2 x recvCount elements with distinct per-block fill.
 * Expected output for active index m: blockFillValue(0,m)+blockFillValue(1,m).
 */
TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_4Groups_InPlace_64MB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t recvBytes = 64ULL * 1024 * 1024;
  const size_t recvCount = recvBytes / sizeof(int32_t);

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  // Active rank: sendBuff holds 2 x recvCount elements; recvBuff aliases the
  // ownBlock region (in-place). Helpers: two-slot buffer of 2 x recvCount.
  int32_t* sendBuffs[nGroups];
  void* recvPtrs[nGroups];
  const void* sendPtrs[nGroups];
  size_t recvCounts[nGroups];

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(
          &sendBuffs[g], static_cast<size_t>(2) * recvBytes)); // 2 blocks
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
    }
  }

  barrierSyncOn(sendBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      initActiveSendBuffer(sendBuffs[g], recvCount, myActiveIndex);
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }

  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    if (g == myActiveGroup) {
      // In-place: recvBuff == sendBuff + ownBlockOffset
      size_t ownBlockOffset = static_cast<size_t>(myActiveIndex) * recvCount;
      recvPtrs[g] = sendBuffs[g] + ownBlockOffset;
    } else {
      recvPtrs[g] = sendBuffs[g];
    }
    recvCounts[g] = recvCount;
  }

  ncclResult_t result = callReduceScatterCompat(
      sendPtrs,
      recvPtrs,
      recvCounts,
      ncclInt32,
      ncclSum,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  {
    int g = myActiveGroup;
    int errorCount = verifyDeviceBufferEquals(
        static_cast<const int32_t*>(recvPtrs[g]),
        recvCount,
        expectedReduceScatterSum(myActiveIndex),
        g,
        "Found mismatches in in-place reduce-scatter output");

    if (errorCount == 0) {
      std::cout << "R" << this->globalRank << ": Group " << g << " - All "
                << recvCount << " elements verified correctly!" << std::endl;
    }
  }

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
  }
}

/**
 * Test: Multi-Group Correctness with 4 groups (OUT-OF-PLACE)
 */
TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_4Groups_OutOfPlace_64MB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t recvBytes = 64ULL * 1024 * 1024;
  const size_t recvCount = recvBytes / sizeof(int32_t);

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      // Active rank: sendBuff is 2 blocks, recvBuff is 1 block (separate)
      HIPCHECK_TEST(
          hipMalloc(&sendBuffs[g], static_cast<size_t>(2) * recvBytes));
      HIPCHECK_TEST(hipMalloc(&recvBuffs[g], recvBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
      recvBuffs[g] = sendBuffs[g]; // helper: same buffer for send/recv
    }
  }

  barrierSyncOn(recvBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      initActiveSendBuffer(sendBuffs[g], recvCount, myActiveIndex);
      HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, recvBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t recvCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
    recvCounts[g] = recvCount;
  }

  ncclResult_t result = callReduceScatterCompat(
      sendPtrs,
      recvPtrs,
      recvCounts,
      ncclInt32,
      ncclSum,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  {
    int g = myActiveGroup;
    verifyDeviceBufferEquals(
        recvBuffs[g],
        recvCount,
        expectedReduceScatterSum(myActiveIndex),
        g,
        "Found mismatches in out-of-place reduce-scatter output");
  }

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipFree(recvBuffs[g]));
    }
  }
}

/**
 * Test: AVG correctness with 4 groups (OUT-OF-PLACE)
 *
 * Uses ncclAvg so the output is the average of the two active ranks' blocks.
 * Fill values are chosen so the average is exact in integer arithmetic.
 */
TEST_F(ShardedRelayMultiGroupReduceScatterTest, Correctness_4Groups_Avg_64MB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t recvBytes = 64ULL * 1024 * 1024;
  const size_t recvCount = recvBytes / sizeof(int32_t);

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(
          hipMalloc(&sendBuffs[g], static_cast<size_t>(2) * recvBytes));
      HIPCHECK_TEST(hipMalloc(&recvBuffs[g], recvBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
      recvBuffs[g] = sendBuffs[g];
    }
  }

  barrierSyncOn(recvBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      initActiveSendBuffer(sendBuffs[g], recvCount, myActiveIndex);
      HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, recvBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t recvCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
    recvCounts[g] = recvCount;
  }

  ncclResult_t result = callReduceScatterCompat(
      sendPtrs,
      recvPtrs,
      recvCounts,
      ncclInt32,
      ncclAvg,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  {
    int g = myActiveGroup;
    // AVG = sum / nActiveRanksPerGroup
    int32_t expectedAvg =
        expectedReduceScatterSum(myActiveIndex) / nActiveRanksPerGroup;
    verifyDeviceBufferEquals(
        recvBuffs[g],
        recvCount,
        expectedAvg,
        g,
        "Found mismatches in AVG reduce-scatter output");
  }

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipFree(recvBuffs[g]));
    }
  }
}

/**
 * Test: BusBW with 4-group Multi-Group Reduce-Scatter (1GB output, IN-PLACE)
 *
 * A reduce-scatter active rank's sendBuff holds 2 x recvBytes (two blocks), so
 * the per-rank footprint (active 2S + helper groups + relay scratch) is ~9x the
 * per-group output S. 1GB keeps the suite inside a shared devgpu/CI budget.
 */
TEST_F(ShardedRelayMultiGroupReduceScatterTest, Z_BusBW_4Groups_InPlace_1GB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t recvBytes = 1ULL * 1024 * 1024 * 1024; // 1 GB output per group
  const size_t recvCount = recvBytes / sizeof(int32_t);
  const int nIters = 20;

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(
          &sendBuffs[g], static_cast<size_t>(2) * recvBytes)); // 2 blocks
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
    }
  }

  barrierSyncOn(sendBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(
          hipMemset(sendBuffs[g], 1, static_cast<size_t>(2) * recvBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t recvCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    if (g == myActiveGroup) {
      size_t ownBlockOffset = static_cast<size_t>(myActiveIndex) * recvCount;
      recvPtrs[g] = sendBuffs[g] + ownBlockOffset;
    } else {
      recvPtrs[g] = sendBuffs[g];
    }
    recvCounts[g] = recvCount;
  }

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
    ncclResult_t result = callReduceScatterCompat(
        sendPtrs,
        recvPtrs,
        recvCounts,
        ncclInt32,
        ncclSum,
        this->comm,
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
        recvBytes, bestTimeMs, nActiveRanksPerGroup, nGroups);
    printMultiGroupBandwidthResults(
        "4-Group IN-PLACE 1GB",
        recvBytes,
        this->numRanks,
        nGroups,
        nActiveRanksPerGroup,
        bwResult,
        true);
  }

  HIPCHECK_TEST(hipEventDestroy(startEvent));
  HIPCHECK_TEST(hipEventDestroy(stopEvent));
  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
  }
}

/**
 * Test: BusBW with 4-group Multi-Group Reduce-Scatter (1GB, OUT-OF-PLACE)
 *
 * See the in-place benchmark above for why the output size is 1GB (the
 * reduce-scatter send buffer is doubled relative to allreduce).
 */
TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Z_BusBW_4Groups_OutOfPlace_1GB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t recvBytes = 1ULL * 1024 * 1024 * 1024; // 1 GB output per group
  const size_t recvCount = recvBytes / sizeof(int32_t);
  const int nIters = 20;

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(
          hipMalloc(&sendBuffs[g], static_cast<size_t>(2) * recvBytes));
      HIPCHECK_TEST(hipMalloc(&recvBuffs[g], recvBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
      recvBuffs[g] = sendBuffs[g];
    }
  }

  barrierSyncOn(recvBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(
          hipMemset(sendBuffs[g], 1, static_cast<size_t>(2) * recvBytes));
      HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, recvBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t recvCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
    recvCounts[g] = recvCount;
  }

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
    ncclResult_t result = callReduceScatterCompat(
        sendPtrs,
        recvPtrs,
        recvCounts,
        ncclInt32,
        ncclSum,
        this->comm,
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
        recvBytes, bestTimeMs, nActiveRanksPerGroup, nGroups);
    printMultiGroupBandwidthResults(
        "4-Group OUT-OF-PLACE 1GB",
        recvBytes,
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
      HIPCHECK_TEST(hipFree(recvBuffs[g]));
    }
  }
}

/**
 * Test: Single group via multi-group API (backward compatibility check)
 */
TEST_F(ShardedRelayMultiGroupReduceScatterTest, Correctness_SingleGroup_64MB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 1;
  const int nActiveRanksPerGroup = 2;
  const size_t recvBytes = 64ULL * 1024 * 1024;
  const size_t recvCount = recvBytes / sizeof(int32_t);

  const int activeRanks[] = {0, 1};
  const int* allActiveRanks[] = {activeRanks};

  bool isActive = (this->globalRank == 0 || this->globalRank == 1);
  int myActiveIndex = this->globalRank; // 0 or 1 for the active ranks

  int32_t* sendBuff = nullptr;
  int32_t* recvBuff = nullptr;
  if (isActive) {
    HIPCHECK_TEST(
        hipMalloc(&sendBuff, static_cast<size_t>(2) * recvBytes)); // 2 blocks
    HIPCHECK_TEST(hipMalloc(&recvBuff, recvBytes));
  } else {
    size_t helperBufferSize =
        static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
    HIPCHECK_TEST(hipMalloc(&sendBuff, helperBufferSize));
    recvBuff = sendBuff;
  }

  barrierSyncOn(sendBuff);

  if (isActive) {
    initActiveSendBuffer(sendBuff, recvCount, myActiveIndex);
    HIPCHECK_TEST(hipMemset(recvBuff, 0, recvBytes));
  } else {
    size_t helperBufferSize =
        static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
    HIPCHECK_TEST(hipMemset(sendBuff, 0, helperBufferSize));
  }

  const void* sendPtrs[1];
  void* recvPtrs[1];
  size_t recvCounts[] = {recvCount};

  sendPtrs[0] = sendBuff;
  recvPtrs[0] = recvBuff;

  ncclResult_t result = callReduceScatterCompat(
      sendPtrs,
      recvPtrs,
      recvCounts,
      ncclInt32,
      ncclSum,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  if (isActive) {
    verifyDeviceBufferEquals(
        recvBuff,
        recvCount,
        expectedReduceScatterSum(myActiveIndex),
        0,
        "Single group via multi-group API failed");
  }

  HIPCHECK_TEST(hipFree(sendBuff));
  if (isActive) {
    HIPCHECK_TEST(hipFree(recvBuff));
  }
}

/**
 * Test: Correctness with minimum passthrough helper buffers (4 groups,
 * OUT-OF-PLACE)
 *
 * Validates correctness when each helper group's buffer is allocated at the
 * MINIMUM size required by the passthrough design: nActiveRanks x chunkSize
 * elements (chunkSize derived from recvCount).
 */
TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_4Groups_PassthroughHelperEquivalence) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t recvBytes = 64ULL * 1024 * 1024;
  const size_t recvCount = recvBytes / sizeof(int32_t);
  const int numHelpers = this->numRanks - nActiveRanksPerGroup;
  const int numChunks = numHelpers + 1;

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  // Compute the MINIMUM passthrough helper buffer size from recvCount.
  size_t chunkSize = recvCount / numChunks;
  chunkSize = (chunkSize / 128) * 128; // CHUNK_ALIGN_ELEMENTS
  if (chunkSize == 0) {
    chunkSize = recvCount;
  }
  size_t minHelperElements = std::min(
      recvCount, static_cast<size_t>(nActiveRanksPerGroup) * chunkSize);
  size_t minHelperBytes = minHelperElements * sizeof(int32_t);

  // 1 active send buffer (2 blocks) + separate recv buffer; 3 minimal helpers.
  int32_t* activeSendBuffer = nullptr;
  int32_t* activeRecvBuffer = nullptr;
  int32_t* helperBuffers[nGroups];

  HIPCHECK_TEST(
      hipMalloc(&activeSendBuffer, static_cast<size_t>(2) * recvBytes));
  HIPCHECK_TEST(hipMalloc(&activeRecvBuffer, recvBytes));
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      helperBuffers[g] = nullptr;
    } else {
      HIPCHECK_TEST(hipMalloc(&helperBuffers[g], minHelperBytes));
    }
  }

  barrierSyncOn(activeSendBuffer);

  initActiveSendBuffer(activeSendBuffer, recvCount, myActiveIndex);
  HIPCHECK_TEST(hipMemset(activeRecvBuffer, 0, recvBytes));
  for (int g = 0; g < nGroups; g++) {
    if (g != myActiveGroup) {
      HIPCHECK_TEST(hipMemset(helperBuffers[g], 0, minHelperBytes));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t recvCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      sendPtrs[g] = activeSendBuffer;
      recvPtrs[g] = activeRecvBuffer;
    } else {
      sendPtrs[g] = helperBuffers[g];
      recvPtrs[g] = helperBuffers[g];
    }
    recvCounts[g] = recvCount;
  }

  ncclResult_t result = callReduceScatterCompat(
      sendPtrs,
      recvPtrs,
      recvCounts,
      ncclInt32,
      ncclSum,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  {
    int errorCount = verifyDeviceBufferEquals(
        activeRecvBuffer,
        recvCount,
        expectedReduceScatterSum(myActiveIndex),
        myActiveGroup,
        "Found mismatches in passthrough-helper-equivalence output");

    if (errorCount == 0) {
      std::cout << "R" << this->globalRank << ": Group " << myActiveGroup
                << " - All " << recvCount
                << " elements verified correctly (min passthrough helper)!"
                << std::endl;
    }
  }

  HIPCHECK_TEST(hipFree(activeSendBuffer));
  HIPCHECK_TEST(hipFree(activeRecvBuffer));
  for (int g = 0; g < nGroups; g++) {
    if (g != myActiveGroup) {
      HIPCHECK_TEST(hipFree(helperBuffers[g]));
    }
  }
}

/**
 * Test: Correctness_PartialGroupsZeroCount
 *
 * Reduce-scatter analogue of the allreduce partial-zero-count regression: when
 * different sparse groups have different numbers of tensors, exhausted groups
 * pass recvCount=0. The kernel must skip those groups in every phase instead
 * of attempting NCCL ops with zero-element buffers.
 *
 * Setup: 4 groups (8 ranks), 2 active ranks per group.
 *   - Groups 0 and 1: recvCount = 16MB (must produce correct sum)
 *   - Groups 2 and 3: recvCount = 0    (must not corrupt or crash)
 */
TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_PartialGroupsZeroCount) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t recvBytes = 16ULL * 1024 * 1024; // 16MB for groups with data
  const size_t recvCount = recvBytes / sizeof(int32_t);

  const size_t recvCounts[nGroups] = {recvCount, recvCount, 0, 0};

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  const size_t placeholderBytes = sizeof(int32_t); // 1 element
  int32_t* buffers[nGroups];

  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] == 0) {
      HIPCHECK_TEST(hipMalloc(&buffers[g], placeholderBytes));
      HIPCHECK_TEST(hipMemset(buffers[g], 0, placeholderBytes));
      continue;
    }
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(
          &buffers[g], static_cast<size_t>(2) * recvBytes)); // 2 blocks
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMalloc(&buffers[g], helperBufferSize));
    }
  }

  barrierSyncOn(buffers[0]);

  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] == 0) {
      continue; // placeholder already zeroed
    }
    if (g == myActiveGroup) {
      initActiveSendBuffer(buffers[g], recvCount, myActiveIndex);
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMemset(buffers[g], 0, helperBufferSize));
    }
  }

  // In-place: recvBuff == sendBuff + ownBlockOffset for active groups.
  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = buffers[g];
    if (recvCounts[g] != 0 && g == myActiveGroup) {
      size_t ownBlockOffset = static_cast<size_t>(myActiveIndex) * recvCount;
      recvPtrs[g] = buffers[g] + ownBlockOffset;
    } else {
      recvPtrs[g] = buffers[g];
    }
  }

  ncclResult_t result = callReduceScatterCompat(
      sendPtrs,
      recvPtrs,
      recvCounts,
      ncclInt32,
      ncclSum,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess)
      << "ncclShardedRelayMultiGroupReduceScatter failed with partial recvCount=0 groups";
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  // Verify groups with recvCount>0 produced the correct reduce-scatter output.
  if (myActiveGroup < 2) { // groups 0 and 1 had data
    verifyDeviceBufferEquals(
        static_cast<const int32_t*>(recvPtrs[myActiveGroup]),
        recvCount,
        expectedReduceScatterSum(myActiveIndex),
        myActiveGroup,
        "Correctness failed for groups with recvCount>0 when other groups have recvCount=0");
  }

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(buffers[g]));
  }
}

/**
 * 4-ACTIVE tests (2 groups of 4 active ranks each).
 *   Group 0: active {0,1,2,3}, helpers {4,5,6,7}
 *   Group 1: active {4,5,6,7}, helpers {0,1,2,3}
 *
 * Each active rank's sendBuff holds A=4 blocks of recvCount; block j is filled
 * with blockFillValue(myActiveIndex, j). The reduce-scatter output for owner mi
 * is sum over the 4 active ranks r of blockFillValue(r, mi), so a wrong block
 * mapping (e.g. the bit-reversed-permutation bug) is detectable.
 */
TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_4Active_2Groups_InPlace) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int nGroups = 2;
  const int nActiveRanksPerGroup = 4;
  const size_t recvBytes = 64ULL * 1024 * 1024;
  const size_t recvCount = recvBytes / sizeof(int32_t);

  TwoGroupFourActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(
          &sendBuffs[g],
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes)); // A blocks
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
    }
  }

  barrierSyncOn(sendBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      initActiveSendBuffer(
          sendBuffs[g], recvCount, myActiveIndex, nActiveRanksPerGroup);
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t recvCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    if (g == myActiveGroup) {
      // In-place: recvBuff == sendBuff + ownBlockOffset.
      size_t ownBlockOffset = static_cast<size_t>(myActiveIndex) * recvCount;
      recvPtrs[g] = sendBuffs[g] + ownBlockOffset;
    } else {
      recvPtrs[g] = sendBuffs[g];
    }
    recvCounts[g] = recvCount;
  }

  ncclResult_t result = callReduceScatterCompat(
      sendPtrs,
      recvPtrs,
      recvCounts,
      ncclInt32,
      ncclSum,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  verifyDeviceBufferEquals(
      static_cast<const int32_t*>(recvPtrs[myActiveGroup]),
      recvCount,
      expectedReduceScatterSum(myActiveIndex, nActiveRanksPerGroup),
      myActiveGroup,
      "4-active in-place reduce-scatter SUM mismatch");

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
  }
}

TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_4Active_2Groups_OutOfPlace) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int nGroups = 2;
  const int nActiveRanksPerGroup = 4;
  const size_t recvBytes = 64ULL * 1024 * 1024;
  const size_t recvCount = recvBytes / sizeof(int32_t);

  TwoGroupFourActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(
          &sendBuffs[g],
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes));
      HIPCHECK_TEST(hipMalloc(&recvBuffs[g], recvBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
      recvBuffs[g] = sendBuffs[g];
    }
  }

  barrierSyncOn(recvBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      initActiveSendBuffer(
          sendBuffs[g], recvCount, myActiveIndex, nActiveRanksPerGroup);
      HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, recvBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t recvCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
    recvCounts[g] = recvCount;
  }

  ncclResult_t result = callReduceScatterCompat(
      sendPtrs,
      recvPtrs,
      recvCounts,
      ncclInt32,
      ncclSum,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  verifyDeviceBufferEquals(
      recvBuffs[myActiveGroup],
      recvCount,
      expectedReduceScatterSum(myActiveIndex, nActiveRanksPerGroup),
      myActiveGroup,
      "4-active out-of-place reduce-scatter SUM mismatch");

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipFree(recvBuffs[g]));
    }
  }
}

/**
 * AVG correctness with 4 active ranks. block j of rank r holds
 * blockFillValue(r, j) = (r+1)*10 + (j+1). For owner mi the sum over r=0..3 is
 * 100 + 4*(mi+1), so AVG (divisor = nActiveRanks = 4) = 25 + (mi+1), exact in
 * int32. A wrong divisor (e.g. 2) would give 50 + 2*(mi+1), detecting that AVG
 * uses nActiveRanks rather than a hardcoded 2.
 */
TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_4Active_2Groups_Avg) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int nGroups = 2;
  const int nActiveRanksPerGroup = 4;
  const size_t recvBytes = 64ULL * 1024 * 1024;
  const size_t recvCount = recvBytes / sizeof(int32_t);

  TwoGroupFourActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(
          &sendBuffs[g],
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes));
      HIPCHECK_TEST(hipMalloc(&recvBuffs[g], recvBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
      recvBuffs[g] = sendBuffs[g];
    }
  }

  barrierSyncOn(recvBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      initActiveSendBuffer(
          sendBuffs[g], recvCount, myActiveIndex, nActiveRanksPerGroup);
      HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, recvBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t recvCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
    recvCounts[g] = recvCount;
  }

  ncclResult_t result = callReduceScatterCompat(
      sendPtrs,
      recvPtrs,
      recvCounts,
      ncclInt32,
      ncclAvg,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  int32_t expectedAvg =
      expectedReduceScatterSum(myActiveIndex, nActiveRanksPerGroup) /
      nActiveRanksPerGroup;
  verifyDeviceBufferEquals(
      recvBuffs[myActiveGroup],
      recvCount,
      expectedAvg,
      myActiveGroup,
      "4-active AVG reduce-scatter mismatch");

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipFree(recvBuffs[g]));
    }
  }
}

/**
 * Tiny-count regression: forces the relay csz==0 path for 4 active ranks. With
 * recvCount=512 the working buffer count = A*recvCount = 2048: pR=1024, the
 * first relay half xg=512, csz=align(512/5)=0, so the recursive-halving step
 * exchanges the whole half directly with the partner. The send and recv of that
 * swap MUST share one ncclGroup; splitting them across phases would deadlock.
 * The direct all-to-all D region (pD=1024) is also exercised.
 */
TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_4Active_2Groups_TinyCsz0_InPlace) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int nGroups = 2;
  const int nActiveRanksPerGroup = 4;
  const size_t recvCount = 512; // count = A*recvCount = 2048 -> relay csz == 0
  const size_t recvBytes = recvCount * sizeof(int32_t);

  TwoGroupFourActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    size_t bytes = static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
    HIPCHECK_TEST(hipMalloc(&sendBuffs[g], bytes));
  }

  barrierSyncOn(sendBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      initActiveSendBuffer(
          sendBuffs[g], recvCount, myActiveIndex, nActiveRanksPerGroup);
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t recvCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    if (g == myActiveGroup) {
      size_t ownBlockOffset = static_cast<size_t>(myActiveIndex) * recvCount;
      recvPtrs[g] = sendBuffs[g] + ownBlockOffset;
    } else {
      recvPtrs[g] = sendBuffs[g];
    }
    recvCounts[g] = recvCount;
  }

  ncclResult_t result = callReduceScatterCompat(
      sendPtrs,
      recvPtrs,
      recvCounts,
      ncclInt32,
      ncclSum,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  verifyDeviceBufferEquals(
      static_cast<const int32_t*>(recvPtrs[myActiveGroup]),
      recvCount,
      expectedReduceScatterSum(myActiveIndex, nActiveRanksPerGroup),
      myActiveGroup,
      "4-active tiny csz==0 reduce-scatter SUM mismatch");

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
  }
}

/**
 * Partial-zero-count regression for 4 active ranks: group 0 has data, group 1
 * passes recvCount=0 and must be skipped without crash/corruption.
 */
TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_4Active_PartialGroupsZeroCount) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int nGroups = 2;
  const int nActiveRanksPerGroup = 4;
  const size_t recvBytes = 16ULL * 1024 * 1024;
  const size_t recvCount = recvBytes / sizeof(int32_t);

  const size_t recvCounts[nGroups] = {recvCount, 0};

  TwoGroupFourActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  const size_t placeholderBytes = sizeof(int32_t);
  int32_t* buffers[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] == 0) {
      HIPCHECK_TEST(hipMalloc(&buffers[g], placeholderBytes));
      HIPCHECK_TEST(hipMemset(buffers[g], 0, placeholderBytes));
      continue;
    }
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(
          &buffers[g], static_cast<size_t>(nActiveRanksPerGroup) * recvBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMalloc(&buffers[g], helperBufferSize));
    }
  }

  barrierSyncOn(buffers[0]);

  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] == 0) {
      continue;
    }
    if (g == myActiveGroup) {
      initActiveSendBuffer(
          buffers[g], recvCount, myActiveIndex, nActiveRanksPerGroup);
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
      HIPCHECK_TEST(hipMemset(buffers[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = buffers[g];
    if (recvCounts[g] != 0 && g == myActiveGroup) {
      size_t ownBlockOffset = static_cast<size_t>(myActiveIndex) * recvCount;
      recvPtrs[g] = buffers[g] + ownBlockOffset;
    } else {
      recvPtrs[g] = buffers[g];
    }
  }

  ncclResult_t result = callReduceScatterCompat(
      sendPtrs,
      recvPtrs,
      recvCounts,
      ncclInt32,
      ncclSum,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess)
      << "4-active reduce-scatter failed with a partial recvCount=0 group";
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  if (myActiveGroup == 0) { // group 0 had data
    verifyDeviceBufferEquals(
        static_cast<const int32_t*>(recvPtrs[myActiveGroup]),
        recvCount,
        expectedReduceScatterSum(myActiveIndex, nActiveRanksPerGroup),
        myActiveGroup,
        "4-active partial-zero reduce-scatter SUM mismatch");
  }

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(buffers[g]));
  }
}

/**
 * BusBW with 4-active 2-group reduce-scatter (1GB output, IN-PLACE).
 *
 * The reduce-scatter active rank holds A=4 blocks (A x recvBytes send buffer)
 * plus a working buffer of A x recvBytes; 1GB keeps the per-rank footprint
 * inside a shared devgpu/CI memory budget.
 */
TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Z_BusBW_4Active_2Groups_InPlace_1GB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int nGroups = 2;
  const int nActiveRanksPerGroup = 4;
  const size_t recvBytes = 1ULL * 1024 * 1024 * 1024; // 1 GB output per group
  const size_t recvCount = recvBytes / sizeof(int32_t);
  const int nIters = 20;

  TwoGroupFourActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    size_t bytes = static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
    HIPCHECK_TEST(hipMalloc(&sendBuffs[g], bytes));
  }

  barrierSyncOn(sendBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    size_t bytes = static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;
    HIPCHECK_TEST(hipMemset(sendBuffs[g], (g == myActiveGroup) ? 1 : 0, bytes));
  }
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t recvCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    if (g == myActiveGroup) {
      size_t ownBlockOffset = static_cast<size_t>(myActiveIndex) * recvCount;
      recvPtrs[g] = sendBuffs[g] + ownBlockOffset;
    } else {
      recvPtrs[g] = sendBuffs[g];
    }
    recvCounts[g] = recvCount;
  }

  hipEvent_t startEvent, stopEvent;
  HIPCHECK_TEST(hipEventCreate(&startEvent));
  HIPCHECK_TEST(hipEventCreate(&stopEvent));

  float bestTimeMs = std::numeric_limits<float>::max();
  float totalTimeMs = 0.0f;

  for (int iter = 0; iter < nIters; iter++) {
    HIPCHECK_TEST(hipEventRecord(startEvent, this->stream));
    ncclResult_t result = callReduceScatterCompat(
        sendPtrs,
        recvPtrs,
        recvCounts,
        ncclInt32,
        ncclSum,
        this->comm,
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
        recvBytes, bestTimeMs, nActiveRanksPerGroup, nGroups);
    printMultiGroupBandwidthResults(
        "4-Active 2-Group IN-PLACE 1GB",
        recvBytes,
        this->numRanks,
        nGroups,
        nActiveRanksPerGroup,
        bwResult,
        true);
  }

  HIPCHECK_TEST(hipEventDestroy(startEvent));
  HIPCHECK_TEST(hipEventDestroy(stopEvent));
  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
