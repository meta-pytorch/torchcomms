// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * Unit tests for Fused Multi-Group Sharded Relay All-Gather
 *
 * All-gather analogue of ShardedRelayReduceScatterTest.cu (and the dual of
 * reduce-scatter). Tests the phase-synchronized execution of multiple sharded
 * relay all-gathers with passthrough-at-helper design. All-gather performs NO
 * reduction; helpers forward data and active ranks place it. Both in-place and
 * out-of-place are supported.
 *
 * All-Gather Semantics (per group, 2 active ranks):
 * =================================================
 * Each active rank's sendBuff holds sendCount elements (its contribution);
 * recvBuff holds nActiveRanksPerGroup x sendCount elements, where
 * recvBuff[i x sendCount] receives the contribution from active index i.
 *
 * To verify the gather, each active rank fills its sendBuff with a distinct
 * value rankFillValue(myActiveIndex). After the all-gather, every active rank's
 * recvBuff slot i must equal rankFillValue(i). A wrong slot offset in the
 * implementation produces a detectable mismatch.
 *
 * In-place is detected when sendBuff == recvBuff + myActiveIndex x sendCount.
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

// Aggregate bandwidth for multi-group all-gather. The size metric is the
// per-group per-rank contribution (sendCount) bytes. Bus BW for an n-rank
// all-gather uses the (n-1)/n factor (for 2 ranks this is 0.5).
ShardedBandwidthResult calculateMultiGroupAggregateBandwidth(
    size_t sendBytesPerGroup,
    double elapsedMs,
    int numActiveRanks,
    int numGroups) {
  ShardedBandwidthResult result;
  double elapsedSec = elapsedMs / 1000.0;
  double totalDataSizeGB = static_cast<double>(sendBytesPerGroup) * numGroups /
      (1024.0 * 1024.0 * 1024.0);
  result.algoBW_GBps = totalDataSizeGB / elapsedSec;
  result.busBW_GBps =
      (numActiveRanks - 1.0) / numActiveRanks * totalDataSizeGB / elapsedSec;
  result.latency_us = elapsedMs * 1000.0;
  return result;
}

void printMultiGroupBandwidthResults(
    const std::string& testName,
    size_t sendBytesPerGroup,
    int numRanks,
    int numGroups,
    int activeRanksPerGroup,
    const ShardedBandwidthResult& aggregateResult,
    bool isInPlace) {
  double dataSizePerGroupGB =
      static_cast<double>(sendBytesPerGroup) / (1024.0 * 1024.0 * 1024.0);
  double totalDataSizeGB = dataSizePerGroupGB * numGroups;
  double perGroupAlgoBW = aggregateResult.algoBW_GBps / numGroups;
  double perGroupBusBW = aggregateResult.busBW_GBps / numGroups;

  std::cout << "\n";
  std::cout << "====================================================\n";
  std::cout << "Multi-Group Sharded Relay All-Gather: " << testName << "\n";
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
  std::cout << "  Send Size/Group:       " << dataSizePerGroupGB << " GB\n";
  std::cout << "  Total Gathered (grps): " << totalDataSizeGB << " GB\n";
  std::cout << "  Send Count/Group:      "
            << (sendBytesPerGroup / sizeof(int32_t)) << "\n";
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
// sharded-relay all-gather entry point. Active group input = sendCount; output
// = nActiveRanks x sendCount. All-gather may be in-place
// (sendPtrs[g] aliases recvPtrs[g] + myActiveIndex x sendCount) or
// out-of-place.
static ncclResult_t callAllGatherCompat(
    const void* const* sendPtrs,
    void* const* recvPtrs,
    const size_t* sendCounts,
    ncclDataType_t datatype,
    ncclComm_t comm,
    hipStream_t stream,
    const int* const* allActiveRanks,
    int nActiveRanksPerGroup,
    int nGroups) {
  return ncclShardedRelayMultiGroupAllGather(
      sendPtrs,
      recvPtrs,
      sendCounts,
      datatype,
      comm,
      stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
}

class ShardedRelayMultiGroupAllGatherTest : public ::testing::Test {
 public:
  ShardedRelayMultiGroupAllGatherTest() = default;

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

  // 8-rank, 2-group, 4-active-per-group layout for the 4-active recursive path:
  //   Group 0: activeRanks = {0, 1, 2, 3}, helpers = {4, 5, 6, 7}
  //   Group 1: activeRanks = {4, 5, 6, 7}, helpers = {0, 1, 2, 3}
  struct TwoGroupFourActiveRanks {
    int storage[2][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}};
    const int* allActiveRanks[2] = {storage[0], storage[1]};
  };

  // Distinct per-active-rank fill value so that a wrong slot offset in the
  // implementation produces a detectable mismatch.
  static int32_t rankFillValue(int activeIndex) {
    return (activeIndex + 1) * 100 + 7;
  }

  // Fill `count` int32_t elements of a device region with `value`.
  void fillDeviceRegion(int32_t* devicePtr, size_t count, int32_t value) {
    std::vector<int32_t> host(count, value);
    HIPCHECK_TEST(hipMemcpy(
        devicePtr,
        host.data(),
        count * sizeof(int32_t),
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

  // Verify all gathered slots of an active rank's recvBuff:
  //   recvBuff[i x sendCount] == rankFillValue(i).
  void verifyAllGatherOutput(
      const int32_t* recvBuff,
      size_t sendCount,
      int groupIndex,
      int nActiveRanks = 2) {
    for (int i = 0; i < nActiveRanks; i++) {
      verifyDeviceBufferEquals(
          recvBuff + static_cast<size_t>(i) * sendCount,
          sendCount,
          rankFillValue(i),
          groupIndex,
          "Found mismatches in all-gather slot[i]");
    }
  }

  static const char* allGatherRouteName(rcclx::relay::AllGatherRoute route) {
    switch (route) {
      case rcclx::relay::AllGatherRoute::PureDirect:
        return "PureDirect";
      case rcclx::relay::AllGatherRoute::A2Relay:
        return "A2Relay";
      case rcclx::relay::AllGatherRoute::FlatOffload:
        return "FlatOffload";
    }
    return "unknown";
  }

  void runOutOfPlaceRoutingCases(
      const int* const* allActiveRanks,
      int nActiveRanksPerGroup,
      int nGroups,
      const std::vector<size_t>& caseSendCounts,
      size_t pureDirectMaxCount) {
    int myActiveGroup = -1;
    int myActiveIndex = -1;
    for (int g = 0; g < nGroups; g++) {
      for (int i = 0; i < nActiveRanksPerGroup; i++) {
        if (allActiveRanks[g][i] == this->globalRank) {
          myActiveGroup = g;
          myActiveIndex = i;
        }
      }
    }

    constexpr int32_t kHelperSentinel = -1;
    for (size_t sendCount : caseSendCounts) {
      const size_t sendBytes = sendCount * sizeof(int32_t);
      const size_t helperElements =
          static_cast<size_t>(nActiveRanksPerGroup) * sendCount;
      // Which route runs is internal to the collective and derived only from
      // the message size. These cases sweep just below/at/above a crossover, so
      // assert the collective's own selector actually flips there: helpers now
      // stage into kernel-owned internal scratch and never write the caller's
      // buffer on any route, so without this the sweep would pass even if the
      // routing collapsed to a single path.
      const std::vector<size_t> routeSendCounts(nGroups, sendCount);
      const rcclx::relay::AllGatherRoute expectedRoute =
          sendCount < pureDirectMaxCount
          ? rcclx::relay::AllGatherRoute::PureDirect
          : (nActiveRanksPerGroup == 2
                 ? rcclx::relay::AllGatherRoute::A2Relay
                 : rcclx::relay::AllGatherRoute::FlatOffload);
      const rcclx::relay::AllGatherRoute actualRoute =
          rcclx::relay::selectAllGatherRoute(
              nActiveRanksPerGroup,
              this->numRanks - nActiveRanksPerGroup,
              nGroups,
              routeSendCounts.data(),
              sizeof(int32_t));
      EXPECT_EQ(actualRoute, expectedRoute)
          << "internal route selection resolved to "
          << allGatherRouteName(actualRoute) << " but sendCount=" << sendCount
          << " against a pure-direct max of " << pureDirectMaxCount
          << " (A=" << nActiveRanksPerGroup << ", nGroups=" << nGroups
          << ") is written for " << allGatherRouteName(expectedRoute);
      std::vector<int32_t*> sendBuffs(nGroups);
      std::vector<int32_t*> recvBuffs(nGroups);

      for (int g = 0; g < nGroups; g++) {
        if (g == myActiveGroup) {
          HIPCHECK_TEST(hipMalloc(&sendBuffs[g], sendBytes));
          HIPCHECK_TEST(hipMalloc(
              &recvBuffs[g],
              static_cast<size_t>(nActiveRanksPerGroup) * sendBytes));
        } else {
          HIPCHECK_TEST(
              hipMalloc(&sendBuffs[g], helperElements * sizeof(int32_t)));
          recvBuffs[g] = sendBuffs[g];
        }
      }

      barrierSyncOn(nullptr);

      for (int g = 0; g < nGroups; g++) {
        if (g == myActiveGroup) {
          fillDeviceRegion(
              sendBuffs[g], sendCount, rankFillValue(myActiveIndex));
          HIPCHECK_TEST(hipMemset(
              recvBuffs[g],
              0,
              static_cast<size_t>(nActiveRanksPerGroup) * sendBytes));
        } else {
          fillDeviceRegion(sendBuffs[g], helperElements, kHelperSentinel);
        }
      }

      std::vector<const void*> sendPtrs(nGroups);
      std::vector<void*> recvPtrs(nGroups);
      std::vector<size_t> sendCounts(nGroups, sendCount);
      for (int g = 0; g < nGroups; g++) {
        sendPtrs[g] = sendBuffs[g];
        recvPtrs[g] = recvBuffs[g];
      }

      ncclResult_t result = callAllGatherCompat(
          sendPtrs.data(),
          recvPtrs.data(),
          sendCounts.data(),
          ncclInt32,
          this->comm,
          this->stream,
          allActiveRanks,
          nActiveRanksPerGroup,
          nGroups);
      ASSERT_EQ(result, ncclSuccess) << "sendCount=" << sendCount;
      HIPCHECK_TEST(hipStreamSynchronize(this->stream));

      if (myActiveGroup >= 0) {
        verifyAllGatherOutput(
            recvBuffs[myActiveGroup],
            sendCount,
            myActiveGroup,
            nActiveRanksPerGroup);
      }
      for (int g = 0; g < nGroups; g++) {
        if (g == myActiveGroup) {
          continue;
        }
        verifyDeviceBufferEquals(
            recvBuffs[g],
            1,
            kHelperSentinel,
            g,
            "Helper caller buffer must stay untouched (internal scratch)");
      }

      for (int g = 0; g < nGroups; g++) {
        HIPCHECK_TEST(hipFree(sendBuffs[g]));
        if (g == myActiveGroup) {
          HIPCHECK_TEST(hipFree(recvBuffs[g]));
        }
      }
    }
  }

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  ncclComm_t comm;
  cudaStream_t stream;
  std::unique_ptr<c10d::TCPStore> server{nullptr};
};

TEST_F(
    ShardedRelayMultiGroupAllGatherTest,
    Correctness_4Groups_A2_SmallPureDirect_Unaligned) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  constexpr size_t kFusedA2ThresholdElements =
      (2ULL * 1024 * 1024) / sizeof(int32_t);
  Standard4GroupActiveRanks groupConfig;
  runOutOfPlaceRoutingCases(
      groupConfig.allActiveRanks, 2, 4, {1025}, kFusedA2ThresholdElements);
}

TEST_F(
    ShardedRelayMultiGroupAllGatherTest,
    Correctness_4Groups_A2_FusedRoutingThresholds) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  constexpr size_t kThresholdElements = (2ULL * 1024 * 1024) / sizeof(int32_t);
  Standard4GroupActiveRanks groupConfig;
  runOutOfPlaceRoutingCases(
      groupConfig.allActiveRanks,
      2,
      4,
      {kThresholdElements - 1, kThresholdElements, kThresholdElements + 1},
      kThresholdElements);
}

TEST_F(
    ShardedRelayMultiGroupAllGatherTest,
    Correctness_SingleGroup_A2_RoutingThresholds) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  constexpr size_t kThresholdElements = (9ULL * 1024 * 1024) / sizeof(int32_t);
  const int activeRanks[] = {0, 1};
  const int* allActiveRanks[] = {activeRanks};
  runOutOfPlaceRoutingCases(
      allActiveRanks,
      2,
      1,
      {kThresholdElements - 1, kThresholdElements, kThresholdElements + 1},
      kThresholdElements);
}

TEST_F(
    ShardedRelayMultiGroupAllGatherTest,
    Correctness_2Groups_A4_FusedRoutingThresholds) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  constexpr size_t kThresholdElements = (12ULL * 1024 * 1024) / sizeof(int32_t);
  TwoGroupFourActiveRanks groupConfig;
  runOutOfPlaceRoutingCases(
      groupConfig.allActiveRanks,
      4,
      2,
      {kThresholdElements - 1, kThresholdElements, kThresholdElements + 1},
      kThresholdElements);
}

TEST_F(
    ShardedRelayMultiGroupAllGatherTest,
    Correctness_SingleGroup_A4_RoutingThresholds) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  constexpr size_t kThresholdElements = (8ULL * 1024 * 1024) / sizeof(int32_t);
  const int activeRanks[] = {0, 1, 2, 3};
  const int* allActiveRanks[] = {activeRanks};
  runOutOfPlaceRoutingCases(
      allActiveRanks,
      4,
      1,
      {kThresholdElements - 1, kThresholdElements, kThresholdElements + 1},
      kThresholdElements);
}

/**
 * Test: Multi-Group Correctness with 4 groups (OUT-OF-PLACE)
 *
 * Active sendBuff holds sendCount elements filled with rankFillValue(m);
 * recvBuff holds 2 x sendCount. Expected recvBuff slot i = rankFillValue(i).
 */
TEST_F(
    ShardedRelayMultiGroupAllGatherTest,
    Correctness_4Groups_OutOfPlace_64MB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t sendBytes = 64ULL * 1024 * 1024;
  const size_t sendCount = sendBytes / sizeof(int32_t);

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], sendBytes)); // 1 segment
      HIPCHECK_TEST(
          hipMalloc(&recvBuffs[g], static_cast<size_t>(2) * sendBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
      recvBuffs[g] = sendBuffs[g]; // helper: same buffer for send/recv
    }
  }

  barrierSyncOn(recvBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      fillDeviceRegion(sendBuffs[g], sendCount, rankFillValue(myActiveIndex));
      HIPCHECK_TEST(
          hipMemset(recvBuffs[g], 0, static_cast<size_t>(2) * sendBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t sendCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
    sendCounts[g] = sendCount;
  }

  ncclResult_t result = callAllGatherCompat(
      sendPtrs,
      recvPtrs,
      sendCounts,
      ncclInt32,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  verifyAllGatherOutput(recvBuffs[myActiveGroup], sendCount, myActiveGroup);

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipFree(recvBuffs[g]));
    }
  }
}

/**
 * Test: Multi-Group Correctness with 4 groups (IN-PLACE)
 *
 * In-place all-gather: sendBuff aliases recvBuff + myActiveIndex x sendCount.
 */
TEST_F(ShardedRelayMultiGroupAllGatherTest, Correctness_4Groups_InPlace_64MB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t sendBytes = 64ULL * 1024 * 1024;
  const size_t sendCount = sendBytes / sizeof(int32_t);

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  // Active rank: one recvBuff (2 x sendBytes); sendBuff aliases its own slot.
  // Helpers: two-slot scratch buffer.
  int32_t* recvBuffs[nGroups];

  for (int g = 0; g < nGroups; g++) {
    size_t bufSize = static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
    HIPCHECK_TEST(hipMalloc(&recvBuffs[g], bufSize));
  }

  barrierSyncOn(recvBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      // Zero the whole recvBuff, then fill my own slot in place.
      HIPCHECK_TEST(
          hipMemset(recvBuffs[g], 0, static_cast<size_t>(2) * sendBytes));
      size_t ownSlotOffset = static_cast<size_t>(myActiveIndex) * sendCount;
      fillDeviceRegion(
          recvBuffs[g] + ownSlotOffset,
          sendCount,
          rankFillValue(myActiveIndex));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
      HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t sendCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    recvPtrs[g] = recvBuffs[g];
    if (g == myActiveGroup) {
      // In-place: sendBuff == recvBuff + myActiveIndex x sendCount
      size_t ownSlotOffset = static_cast<size_t>(myActiveIndex) * sendCount;
      sendPtrs[g] = recvBuffs[g] + ownSlotOffset;
    } else {
      sendPtrs[g] = recvBuffs[g];
    }
    sendCounts[g] = sendCount;
  }

  ncclResult_t result = callAllGatherCompat(
      sendPtrs,
      recvPtrs,
      sendCounts,
      ncclInt32,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  verifyAllGatherOutput(recvBuffs[myActiveGroup], sendCount, myActiveGroup);

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(recvBuffs[g]));
  }
}

/**
 * Test: Single group via multi-group API (OUT-OF-PLACE)
 */
TEST_F(ShardedRelayMultiGroupAllGatherTest, Correctness_SingleGroup_64MB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 1;
  const int nActiveRanksPerGroup = 2;
  const size_t sendBytes = 64ULL * 1024 * 1024;
  const size_t sendCount = sendBytes / sizeof(int32_t);

  const int activeRanks[] = {0, 1};
  const int* allActiveRanks[] = {activeRanks};

  bool isActive = (this->globalRank == 0 || this->globalRank == 1);
  int myActiveIndex = this->globalRank; // 0 or 1 for active ranks

  int32_t* sendBuff = nullptr;
  int32_t* recvBuff = nullptr;
  if (isActive) {
    HIPCHECK_TEST(hipMalloc(&sendBuff, sendBytes));
    HIPCHECK_TEST(hipMalloc(&recvBuff, static_cast<size_t>(2) * sendBytes));
  } else {
    size_t helperBufferSize =
        static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
    HIPCHECK_TEST(hipMalloc(&sendBuff, helperBufferSize));
    recvBuff = sendBuff;
  }

  barrierSyncOn(sendBuff);

  if (isActive) {
    fillDeviceRegion(sendBuff, sendCount, rankFillValue(myActiveIndex));
    HIPCHECK_TEST(hipMemset(recvBuff, 0, static_cast<size_t>(2) * sendBytes));
  } else {
    size_t helperBufferSize =
        static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
    HIPCHECK_TEST(hipMemset(sendBuff, 0, helperBufferSize));
  }

  const void* sendPtrs[1];
  void* recvPtrs[1];
  size_t sendCounts[] = {sendCount};
  sendPtrs[0] = sendBuff;
  recvPtrs[0] = recvBuff;

  ncclResult_t result = callAllGatherCompat(
      sendPtrs,
      recvPtrs,
      sendCounts,
      ncclInt32,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  if (isActive) {
    verifyAllGatherOutput(recvBuff, sendCount, 0);
  }

  HIPCHECK_TEST(hipFree(sendBuff));
  if (isActive) {
    HIPCHECK_TEST(hipFree(recvBuff));
  }
}

/**
 * Test: Correctness with minimum passthrough helper buffers (OUT-OF-PLACE)
 */
TEST_F(
    ShardedRelayMultiGroupAllGatherTest,
    Correctness_4Groups_PassthroughHelperEquivalence) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t sendBytes = 64ULL * 1024 * 1024;
  const size_t sendCount = sendBytes / sizeof(int32_t);
  const int numHelpers = this->numRanks - nActiveRanksPerGroup;
  const int numChunks = numHelpers + 2;

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  // Minimum passthrough helper buffer size from sendCount.
  size_t chunkSize = sendCount / numChunks;
  chunkSize = (chunkSize / 128) * 128; // CHUNK_ALIGN_ELEMENTS
  if (chunkSize == 0) {
    chunkSize = sendCount;
  }
  size_t minHelperElements = std::min(
      sendCount, static_cast<size_t>(nActiveRanksPerGroup) * chunkSize);
  size_t minHelperBytes = minHelperElements * sizeof(int32_t);

  int32_t* activeSendBuffer = nullptr;
  int32_t* activeRecvBuffer = nullptr;
  int32_t* helperBuffers[nGroups];

  HIPCHECK_TEST(hipMalloc(&activeSendBuffer, sendBytes));
  HIPCHECK_TEST(
      hipMalloc(&activeRecvBuffer, static_cast<size_t>(2) * sendBytes));
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      helperBuffers[g] = nullptr;
    } else {
      HIPCHECK_TEST(hipMalloc(&helperBuffers[g], minHelperBytes));
    }
  }

  barrierSyncOn(activeSendBuffer);

  fillDeviceRegion(activeSendBuffer, sendCount, rankFillValue(myActiveIndex));
  HIPCHECK_TEST(
      hipMemset(activeRecvBuffer, 0, static_cast<size_t>(2) * sendBytes));
  for (int g = 0; g < nGroups; g++) {
    if (g != myActiveGroup) {
      HIPCHECK_TEST(hipMemset(helperBuffers[g], 0, minHelperBytes));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t sendCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      sendPtrs[g] = activeSendBuffer;
      recvPtrs[g] = activeRecvBuffer;
    } else {
      sendPtrs[g] = helperBuffers[g];
      recvPtrs[g] = helperBuffers[g];
    }
    sendCounts[g] = sendCount;
  }

  ncclResult_t result = callAllGatherCompat(
      sendPtrs,
      recvPtrs,
      sendCounts,
      ncclInt32,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  verifyAllGatherOutput(activeRecvBuffer, sendCount, myActiveGroup);

  HIPCHECK_TEST(hipFree(activeSendBuffer));
  HIPCHECK_TEST(hipFree(activeRecvBuffer));
  for (int g = 0; g < nGroups; g++) {
    if (g != myActiveGroup) {
      HIPCHECK_TEST(hipFree(helperBuffers[g]));
    }
  }
}

TEST_F(
    ShardedRelayMultiGroupAllGatherTest,
    Correctness_4Groups_HeterogeneousRelayAndTiny) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t sendCounts[nGroups] = {
      (4ULL * 1024 * 1024) / sizeof(int32_t), 511, 777, 1023};
  const bool inPlace[nGroups] = {false, true, false, true};
  const int numChunks = this->numRanks - nActiveRanksPerGroup + 2;
  const int32_t helperSentinel = -1;

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    size_t sendBytes = sendCounts[g] * sizeof(int32_t);
    size_t recvBytes = static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
    if (g != myActiveGroup) {
      size_t chunkSize = sendCounts[g] / numChunks;
      chunkSize = (chunkSize / 128) * 128;
      size_t helperElements = chunkSize == 0
          ? 1
          : static_cast<size_t>(nActiveRanksPerGroup) * chunkSize;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperElements * sizeof(int32_t)));
      recvBuffs[g] = sendBuffs[g];
    } else if (inPlace[g]) {
      HIPCHECK_TEST(hipMalloc(&recvBuffs[g], recvBytes));
      sendBuffs[g] =
          recvBuffs[g] + static_cast<size_t>(myActiveIndex) * sendCounts[g];
    } else {
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], sendBytes));
      HIPCHECK_TEST(hipMalloc(&recvBuffs[g], recvBytes));
    }
  }

  barrierSyncOn(recvBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      size_t recvBytes = static_cast<size_t>(nActiveRanksPerGroup) *
          sendCounts[g] * sizeof(int32_t);
      HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, recvBytes));
      fillDeviceRegion(
          sendBuffs[g], sendCounts[g], rankFillValue(myActiveIndex));
      continue;
    }

    size_t chunkSize = sendCounts[g] / numChunks;
    chunkSize = (chunkSize / 128) * 128;
    size_t helperElements = chunkSize == 0
        ? 1
        : static_cast<size_t>(nActiveRanksPerGroup) * chunkSize;
    fillDeviceRegion(recvBuffs[g], helperElements, helperSentinel);
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
  }

  ncclResult_t result = callAllGatherCompat(
      sendPtrs,
      recvPtrs,
      sendCounts,
      ncclInt32,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  verifyAllGatherOutput(
      recvBuffs[myActiveGroup], sendCounts[myActiveGroup], myActiveGroup);
  for (int g = 0; g < nGroups; g++) {
    if (g != myActiveGroup) {
      verifyDeviceBufferEquals(
          recvBuffs[g],
          1,
          helperSentinel,
          g,
          "Helper caller buffer must stay untouched (internal scratch)");
    }
  }

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup && !inPlace[g]) {
      HIPCHECK_TEST(hipFree(sendBuffs[g]));
      HIPCHECK_TEST(hipFree(recvBuffs[g]));
    } else {
      HIPCHECK_TEST(hipFree(recvBuffs[g]));
    }
  }
}

/**
 * Test: Correctness_PartialGroupsZeroCount (OUT-OF-PLACE)
 *
 * Groups 0 and 1 have data; groups 2 and 3 pass sendCount=0 and must be skipped
 * without crash or corruption.
 */
TEST_F(
    ShardedRelayMultiGroupAllGatherTest,
    Correctness_PartialGroupsZeroCount) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t sendBytes = 16ULL * 1024 * 1024;
  const size_t sendCount = sendBytes / sizeof(int32_t);

  const size_t sendCounts[nGroups] = {sendCount, sendCount, 0, 0};

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  const size_t placeholderBytes = sizeof(int32_t); // 1 element
  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];

  for (int g = 0; g < nGroups; g++) {
    if (sendCounts[g] == 0) {
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], placeholderBytes));
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, placeholderBytes));
      recvBuffs[g] = sendBuffs[g];
      continue;
    }
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], sendBytes));
      HIPCHECK_TEST(
          hipMalloc(&recvBuffs[g], static_cast<size_t>(2) * sendBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
      recvBuffs[g] = sendBuffs[g];
    }
  }

  barrierSyncOn(sendBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    if (sendCounts[g] == 0) {
      continue;
    }
    if (g == myActiveGroup) {
      fillDeviceRegion(sendBuffs[g], sendCount, rankFillValue(myActiveIndex));
      HIPCHECK_TEST(
          hipMemset(recvBuffs[g], 0, static_cast<size_t>(2) * sendBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
  }

  ncclResult_t result = callAllGatherCompat(
      sendPtrs,
      recvPtrs,
      sendCounts,
      ncclInt32,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess)
      << "ncclShardedRelayMultiGroupAllGather failed with partial sendCount=0 groups";
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  if (myActiveGroup < 2) { // groups 0 and 1 had data
    verifyAllGatherOutput(recvBuffs[myActiveGroup], sendCount, myActiveGroup);
  }

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
    if (sendCounts[g] != 0 && g == myActiveGroup) {
      HIPCHECK_TEST(hipFree(recvBuffs[g]));
    }
  }
}

/**
 * Test: BusBW with 4-group Multi-Group All-Gather (1GB send, OUT-OF-PLACE)
 */
TEST_F(ShardedRelayMultiGroupAllGatherTest, Z_BusBW_4Groups_OutOfPlace_1GB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t sendBytes = 1ULL * 1024 * 1024 * 1024; // 1 GB send/group
  const size_t sendCount = sendBytes / sizeof(int32_t);
  const int nIters = 20;

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], sendBytes));
      HIPCHECK_TEST(
          hipMalloc(&recvBuffs[g], static_cast<size_t>(2) * sendBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
      recvBuffs[g] = sendBuffs[g];
    }
  }

  barrierSyncOn(recvBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 1, sendBytes));
      HIPCHECK_TEST(
          hipMemset(recvBuffs[g], 0, static_cast<size_t>(2) * sendBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t sendCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
    sendCounts[g] = sendCount;
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
    ncclResult_t result = callAllGatherCompat(
        sendPtrs,
        recvPtrs,
        sendCounts,
        ncclInt32,
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
        sendBytes, bestTimeMs, nActiveRanksPerGroup, nGroups);
    printMultiGroupBandwidthResults(
        "4-Group OUT-OF-PLACE 1GB",
        sendBytes,
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
 * Test: BusBW with 4-group Multi-Group All-Gather (1GB send, IN-PLACE)
 */
TEST_F(ShardedRelayMultiGroupAllGatherTest, Z_BusBW_4Groups_InPlace_1GB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t sendBytes = 1ULL * 1024 * 1024 * 1024; // 1 GB send/group
  const size_t sendCount = sendBytes / sizeof(int32_t);
  const int nIters = 20;

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* recvBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    size_t bufSize = static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
    HIPCHECK_TEST(hipMalloc(&recvBuffs[g], bufSize));
  }

  barrierSyncOn(recvBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    size_t bufSize = static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
    HIPCHECK_TEST(hipMemset(recvBuffs[g], 1, bufSize));
  }
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t sendCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    recvPtrs[g] = recvBuffs[g];
    if (g == myActiveGroup) {
      size_t ownSlotOffset = static_cast<size_t>(myActiveIndex) * sendCount;
      sendPtrs[g] = recvBuffs[g] + ownSlotOffset;
    } else {
      sendPtrs[g] = recvBuffs[g];
    }
    sendCounts[g] = sendCount;
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
    ncclResult_t result = callAllGatherCompat(
        sendPtrs,
        recvPtrs,
        sendCounts,
        ncclInt32,
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
        sendBytes, bestTimeMs, nActiveRanksPerGroup, nGroups);
    printMultiGroupBandwidthResults(
        "4-Group IN-PLACE 1GB",
        sendBytes,
        this->numRanks,
        nGroups,
        nActiveRanksPerGroup,
        bwResult,
        true);
  }

  HIPCHECK_TEST(hipEventDestroy(startEvent));
  HIPCHECK_TEST(hipEventDestroy(stopEvent));
  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(recvBuffs[g]));
  }
}

/**
 * 4-ACTIVE tests (2 groups of 4 active ranks each).
 *   Group 0: active {0,1,2,3}, helpers {4,5,6,7}
 *   Group 1: active {4,5,6,7}, helpers {0,1,2,3}
 *
 * Each active rank's sendBuff (sendCount) is filled with rankFillValue(mi);
 * recvBuff holds A=4 slots of sendCount. After the all-gather, every active
 * rank's recvBuff slot i must equal rankFillValue(i), so a wrong source->slot
 * mapping in the flat scatter->forward relay is detected.
 */
TEST_F(
    ShardedRelayMultiGroupAllGatherTest,
    Correctness_4Active_2Groups_OutOfPlace) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int nGroups = 2;
  const int nActiveRanksPerGroup = 4;
  const size_t sendBytes = 64ULL * 1024 * 1024;
  const size_t sendCount = sendBytes / sizeof(int32_t);

  TwoGroupFourActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], sendBytes)); // 1 segment
      HIPCHECK_TEST(hipMalloc(
          &recvBuffs[g],
          static_cast<size_t>(nActiveRanksPerGroup) * sendBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
      recvBuffs[g] = sendBuffs[g];
    }
  }

  barrierSyncOn(recvBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    if (g == myActiveGroup) {
      fillDeviceRegion(sendBuffs[g], sendCount, rankFillValue(myActiveIndex));
      HIPCHECK_TEST(hipMemset(
          recvBuffs[g],
          0,
          static_cast<size_t>(nActiveRanksPerGroup) * sendBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t sendCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
    sendCounts[g] = sendCount;
  }

  ncclResult_t result = callAllGatherCompat(
      sendPtrs,
      recvPtrs,
      sendCounts,
      ncclInt32,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  verifyAllGatherOutput(
      recvBuffs[myActiveGroup], sendCount, myActiveGroup, nActiveRanksPerGroup);

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipFree(recvBuffs[g]));
    }
  }
}

TEST_F(
    ShardedRelayMultiGroupAllGatherTest,
    Correctness_4Active_2Groups_InPlace) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int nGroups = 2;
  const int nActiveRanksPerGroup = 4;
  const size_t sendBytes = 64ULL * 1024 * 1024;
  const size_t sendCount = sendBytes / sizeof(int32_t);

  TwoGroupFourActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  // Active rank: one recvBuff (A x sendBytes); sendBuff aliases its own slot.
  int32_t* recvBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    size_t bufSize = static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
    HIPCHECK_TEST(hipMalloc(&recvBuffs[g], bufSize));
  }

  barrierSyncOn(recvBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    size_t bufSize = static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, bufSize));
      size_t ownSlotOffset = static_cast<size_t>(myActiveIndex) * sendCount;
      fillDeviceRegion(
          recvBuffs[g] + ownSlotOffset,
          sendCount,
          rankFillValue(myActiveIndex));
    } else {
      HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, bufSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t sendCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    recvPtrs[g] = recvBuffs[g];
    if (g == myActiveGroup) {
      // In-place: sendBuff == recvBuff + myActiveIndex x sendCount
      size_t ownSlotOffset = static_cast<size_t>(myActiveIndex) * sendCount;
      sendPtrs[g] = recvBuffs[g] + ownSlotOffset;
    } else {
      sendPtrs[g] = recvBuffs[g];
    }
    sendCounts[g] = sendCount;
  }

  ncclResult_t result = callAllGatherCompat(
      sendPtrs,
      recvPtrs,
      sendCounts,
      ncclInt32,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  verifyAllGatherOutput(
      recvBuffs[myActiveGroup], sendCount, myActiveGroup, nActiveRanksPerGroup);

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(recvBuffs[g]));
  }
}

/**
 * Tiny-segment regression for the flat A>2 all-gather. sendCount=512 keeps the
 * per-helper relay chunk tiny (128-aligned), so it floors toward zero and the
 * slots are delivered mostly over the direct chunks woven into each relay
 * group. All sends and recvs share one ncclGroup per group, so the exchange is
 * deadlock-safe by construction.
 */
TEST_F(
    ShardedRelayMultiGroupAllGatherTest,
    Correctness_4Active_2Groups_TinyCsz0_OutOfPlace) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int nGroups = 2;
  const int nActiveRanksPerGroup = 4;
  const size_t sendCount = 512; // tiny send: relay chunk floors toward zero
  const size_t sendBytes = sendCount * sizeof(int32_t);

  TwoGroupFourActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    size_t helperBufferSize =
        static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], sendBytes));
      HIPCHECK_TEST(hipMalloc(&recvBuffs[g], helperBufferSize));
    } else {
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
      recvBuffs[g] = sendBuffs[g];
    }
  }

  barrierSyncOn(recvBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    size_t helperBufferSize =
        static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
    if (g == myActiveGroup) {
      fillDeviceRegion(sendBuffs[g], sendCount, rankFillValue(myActiveIndex));
      HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, helperBufferSize));
    } else {
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t sendCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
    sendCounts[g] = sendCount;
  }

  ncclResult_t result = callAllGatherCompat(
      sendPtrs,
      recvPtrs,
      sendCounts,
      ncclInt32,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  verifyAllGatherOutput(
      recvBuffs[myActiveGroup], sendCount, myActiveGroup, nActiveRanksPerGroup);

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipFree(recvBuffs[g]));
    }
  }
}

/**
 * Partial-zero-count regression for 4 active ranks: group 0 has data, group 1
 * passes sendCount=0 and must be skipped without crash/corruption.
 */
TEST_F(
    ShardedRelayMultiGroupAllGatherTest,
    Correctness_4Active_PartialGroupsZeroCount) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int nGroups = 2;
  const int nActiveRanksPerGroup = 4;
  const size_t sendBytes = 16ULL * 1024 * 1024;
  const size_t sendCount = sendBytes / sizeof(int32_t);

  const size_t sendCounts[nGroups] = {sendCount, 0};

  TwoGroupFourActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  const size_t placeholderBytes = sizeof(int32_t);
  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    if (sendCounts[g] == 0) {
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], placeholderBytes));
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, placeholderBytes));
      recvBuffs[g] = sendBuffs[g];
      continue;
    }
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], sendBytes));
      HIPCHECK_TEST(hipMalloc(
          &recvBuffs[g],
          static_cast<size_t>(nActiveRanksPerGroup) * sendBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
      recvBuffs[g] = sendBuffs[g];
    }
  }

  barrierSyncOn(sendBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    if (sendCounts[g] == 0) {
      continue;
    }
    if (g == myActiveGroup) {
      fillDeviceRegion(sendBuffs[g], sendCount, rankFillValue(myActiveIndex));
      HIPCHECK_TEST(hipMemset(
          recvBuffs[g],
          0,
          static_cast<size_t>(nActiveRanksPerGroup) * sendBytes));
    } else {
      size_t helperBufferSize =
          static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
  }

  ncclResult_t result = callAllGatherCompat(
      sendPtrs,
      recvPtrs,
      sendCounts,
      ncclInt32,
      this->comm,
      this->stream,
      allActiveRanks,
      nActiveRanksPerGroup,
      nGroups);
  ASSERT_EQ(result, ncclSuccess)
      << "4-active all-gather failed with a partial sendCount=0 group";
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  if (myActiveGroup == 0) { // group 0 had data
    verifyAllGatherOutput(
        recvBuffs[myActiveGroup],
        sendCount,
        myActiveGroup,
        nActiveRanksPerGroup);
  }

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
    if (sendCounts[g] != 0 && g == myActiveGroup) {
      HIPCHECK_TEST(hipFree(recvBuffs[g]));
    }
  }
}

/**
 * BusBW with 4-active 2-group all-gather (1GB send, OUT-OF-PLACE).
 *
 * A 4-active all-gather active rank holds an A-slot recvBuff (A x sendCount);
 * 1GB keeps the per-rank footprint inside a shared devgpu/CI memory budget.
 */
TEST_F(
    ShardedRelayMultiGroupAllGatherTest,
    Z_BusBW_4Active_2Groups_OutOfPlace_1GB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  const int nGroups = 2;
  const int nActiveRanksPerGroup = 4;
  const size_t sendBytes = 1ULL * 1024 * 1024 * 1024; // 1 GB send/group
  const size_t sendCount = sendBytes / sizeof(int32_t);
  const int nIters = 20;

  TwoGroupFourActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  int myActiveGroup = this->globalRank / nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    size_t helperBufferSize =
        static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], sendBytes));
      HIPCHECK_TEST(hipMalloc(&recvBuffs[g], helperBufferSize));
    } else {
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], helperBufferSize));
      recvBuffs[g] = sendBuffs[g];
    }
  }

  barrierSyncOn(recvBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    size_t helperBufferSize =
        static_cast<size_t>(nActiveRanksPerGroup) * sendBytes;
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 1, sendBytes));
      HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, helperBufferSize));
    } else {
      HIPCHECK_TEST(hipMemset(sendBuffs[g], 0, helperBufferSize));
    }
  }
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  size_t sendCounts[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
    sendCounts[g] = sendCount;
  }

  hipEvent_t startEvent, stopEvent;
  HIPCHECK_TEST(hipEventCreate(&startEvent));
  HIPCHECK_TEST(hipEventCreate(&stopEvent));

  float bestTimeMs = std::numeric_limits<float>::max();
  float totalTimeMs = 0.0f;

  for (int iter = 0; iter < nIters; iter++) {
    HIPCHECK_TEST(hipEventRecord(startEvent, this->stream));
    ncclResult_t result = callAllGatherCompat(
        sendPtrs,
        recvPtrs,
        sendCounts,
        ncclInt32,
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
        sendBytes, bestTimeMs, nActiveRanksPerGroup, nGroups);
    printMultiGroupBandwidthResults(
        "4-Active 2-Group OUT-OF-PLACE 1GB",
        sendBytes,
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

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
