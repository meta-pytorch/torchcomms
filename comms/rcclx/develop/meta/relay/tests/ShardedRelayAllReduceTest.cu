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

class ShardedRelayMultiGroupAllReduceTest : public ::testing::Test {
 public:
  ShardedRelayMultiGroupAllReduceTest() = default;

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

  // Run a 1-element ncclAllReduce on `scratchBuffer` to act as a cross-rank
  // barrier (ensuring all ranks reach this point before proceeding with
  // benchmark / correctness work).
  void barrierSyncOn(int32_t* scratchBuffer) {
    HIPCHECK_TEST(hipMemset(scratchBuffer, 0, sizeof(int32_t)));
    NCCLCHECK_TEST(ncclAllReduce(
        scratchBuffer,
        scratchBuffer,
        1,
        ncclInt32,
        ncclSum,
        this->comm,
        this->stream));
    HIPCHECK_TEST(hipStreamSynchronize(this->stream));
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

  ncclResult_t result = ncclShardedRelayMultiGroupAllReduce(
      sendPtrs,
      recvPtrs,
      counts,
      ncclInt32,
      ncclSum,
      this->comm,
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

  ncclResult_t result = ncclShardedRelayMultiGroupAllReduce(
      sendPtrs,
      recvPtrs,
      counts,
      ncclInt32,
      ncclSum,
      this->comm,
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

/**
 * Test: BusBW with 4-group Multi-Group AllReduce (24GB per group, IN-PLACE)
 *
 * This is the key benchmark for 2D sparse parallelism performance.
 * All 4 groups execute in parallel with phase synchronization.
 */
TEST_F(ShardedRelayMultiGroupAllReduceTest, Z_BusBW_4Groups_InPlace_24GB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t dataBytes = 24ULL * 1024 * 1024 * 1024; // 24 GB per group
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
    ncclResult_t result = ncclShardedRelayMultiGroupAllReduce(
        sendPtrs,
        recvPtrs,
        counts,
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

    // Report bandwidth using best time (most representative of peak
    // performance)
    ShardedBandwidthResult bwResult = calculateMultiGroupAggregateBandwidth(
        dataBytes, bestTimeMs, nActiveRanksPerGroup, nGroups);
    printMultiGroupBandwidthResults(
        "4-Group IN-PLACE 24GB",
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
 * Test: BusBW with 4-group Multi-Group AllReduce (24GB per group, OUT-OF-PLACE)
 */
TEST_F(ShardedRelayMultiGroupAllReduceTest, Z_BusBW_4Groups_OutOfPlace_24GB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t dataBytes = 24ULL * 1024 * 1024 * 1024; // 24 GB per group
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
    ncclResult_t result = ncclShardedRelayMultiGroupAllReduce(
        sendPtrs,
        recvPtrs,
        counts,
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
        dataBytes, bestTimeMs, nActiveRanksPerGroup, nGroups);
    printMultiGroupBandwidthResults(
        "4-Group OUT-OF-PLACE 24GB",
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

  ncclResult_t result = ncclShardedRelayMultiGroupAllReduce(
      sendPtrs,
      recvPtrs,
      counts,
      ncclInt32,
      ncclSum,
      this->comm,
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
  const int numChunks = numHelpers + 1;

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

  ncclResult_t result = ncclShardedRelayMultiGroupAllReduce(
      sendPtrs,
      recvPtrs,
      counts,
      ncclInt32,
      ncclSum,
      this->comm,
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
  ncclResult_t result = ncclShardedRelayMultiGroupAllReduce(
      sendPtrs,
      recvPtrs,
      counts,
      ncclInt32,
      ncclSum,
      this->comm,
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

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
