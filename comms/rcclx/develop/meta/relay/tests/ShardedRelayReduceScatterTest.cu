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
#include <cstdint>
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
    int nGroups,
    int lowPrecision = 0) {
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
      nGroups,
      lowPrecision);
}

class ShardedRelayMultiGroupReduceScatterTest : public ::testing::Test {
 public:
  ShardedRelayMultiGroupReduceScatterTest() = default;

  // One comm per active-rank shape, created once for the whole binary and
  // reused by every case, plus a dedicated one for the rank barrier. These used
  // to be created in SetUp and destroyed in TearDown, so each of the 37 cases
  // freed everything an 8-rank comm owns. On MI350 freeing VRAM makes amdgpu
  // wipe it (amdgpu_bo_release_notify -> amdgpu_fill_buffer) while holding
  // mmap_lock for write, so 8 ranks cycling multi-GB comms serialise into a
  // stall that takes the whole host down. Reusing also matches how comms are
  // really used: a handful, kept for the life of the process.
  //
  // Shapes are kept apart because relay state is per-comm and a comm cannot be
  // shared between the 2- and 4-active-rank configurations. One store serves
  // all three, with incrTestCount() between them: the unique-ID rendezvous key
  // is derived from that counter, so without bumping it the second comm would
  // read the first one's stale ID. Built eagerly in a fixed order so every rank
  // consumes the same keys in the same sequence.
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

  static uint16_t bfloat16Bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
  }

  void initActiveBfloat16SendBuffer(
      uint16_t* deviceBuf,
      size_t recvCount,
      int myActiveIndex) {
    constexpr int nActiveRanks = 2;
    std::vector<uint16_t> host(nActiveRanks * recvCount);
    for (int block = 0; block < nActiveRanks; block++) {
      const uint16_t value = bfloat16Bits(
          static_cast<float>(blockFillValue(myActiveIndex, block)));
      std::fill_n(host.data() + block * recvCount, recvCount, value);
    }
    HIPCHECK_TEST(hipMemcpy(
        deviceBuf,
        host.data(),
        host.size() * sizeof(uint16_t),
        hipMemcpyHostToDevice));
  }

  void verifyBfloat16BufferEquals(
      const uint16_t* deviceBuffer,
      size_t count,
      uint16_t expectedValue,
      int groupIndex,
      const char* boundary) {
    std::vector<uint16_t> hostOutput(count);
    HIPCHECK_TEST(hipMemcpy(
        hostOutput.data(),
        deviceBuffer,
        count * sizeof(uint16_t),
        hipMemcpyDeviceToHost));

    int errorCount = 0;
    for (size_t i = 0; i < count && errorCount < 10; ++i) {
      if (hostOutput[i] != expectedValue) {
        std::cout << "R" << this->globalRank << ": Group " << groupIndex
                  << " BF16 mismatch at index " << i << ": expected bits "
                  << expectedValue << " but got " << hostOutput[i] << std::endl;
        errorCount++;
      }
    }
    EXPECT_EQ(errorCount, 0) << "BF16 reduce-scatter mismatch at the "
                             << boundary << " routing-threshold case";
  }

  void verifyBfloat16BufferEquals(
      const uint16_t* deviceBuffer,
      const std::vector<uint16_t>& expected,
      const char* context) {
    std::vector<uint16_t> actual(expected.size());
    HIPCHECK_TEST(hipMemcpy(
        actual.data(),
        deviceBuffer,
        actual.size() * sizeof(uint16_t),
        hipMemcpyDeviceToHost));

    int errorCount = 0;
    for (size_t i = 0; i < actual.size() && errorCount < 10; ++i) {
      if (actual[i] != expected[i]) {
        std::cout << "R" << this->globalRank << ": " << context << " at index "
                  << i << ": expected bits " << expected[i] << " but got "
                  << actual[i] << std::endl;
        errorCount++;
      }
    }
    EXPECT_EQ(errorCount, 0) << context;
  }

  static const char* reduceScatterRouteName(
      rcclx::relay::ReduceScatterRoute route) {
    switch (route) {
      case rcclx::relay::ReduceScatterRoute::PureDirect:
        return "PureDirect";
      case rcclx::relay::ReduceScatterRoute::A2Relay:
        return "A2Relay";
      case rcclx::relay::ReduceScatterRoute::FlatOffload:
        return "FlatOffload";
    }
    return "unknown";
  }

  void runBfloat16A2RoutingThreshold(
      int nGroups,
      size_t thresholdBytes,
      ncclRedOp_t op) {
    constexpr int nActiveRanksPerGroup = 2;
    constexpr uint16_t helperSentinel = 0xffff;
    const size_t thresholdCount =
        thresholdBytes / nActiveRanksPerGroup / sizeof(uint16_t);
    const size_t boundaryCounts[] = {
        thresholdCount - 1, thresholdCount, thresholdCount + 1};
    const char* boundaries[] = {"below", "at", "above"};

    Standard4GroupActiveRanks fourGroupConfig;
    const int singleGroupStorage[] = {0, 1};
    const int* singleGroupActiveRanks[] = {singleGroupStorage};
    const int* const* allActiveRanks =
        nGroups == 1 ? singleGroupActiveRanks : fourGroupConfig.allActiveRanks;

    const int myActiveGroup = nGroups == 1
        ? (this->globalRank < nActiveRanksPerGroup ? 0 : -1)
        : this->globalRank / nActiveRanksPerGroup;
    const int myActiveIndex = nGroups == 1
        ? this->globalRank
        : this->globalRank % nActiveRanksPerGroup;

    for (int boundaryIndex = 0; boundaryIndex < 3; boundaryIndex++) {
      const size_t recvCount = boundaryCounts[boundaryIndex];
      const size_t recvBytes = recvCount * sizeof(uint16_t);
      std::vector<uint16_t*> sendBuffs(nGroups);
      std::vector<uint16_t*> recvBuffs(nGroups);

      for (int g = 0; g < nGroups; g++) {
        HIPCHECK_TEST(
            hipMalloc(&sendBuffs[g], nActiveRanksPerGroup * recvBytes));
        if (g == myActiveGroup) {
          HIPCHECK_TEST(hipMalloc(&recvBuffs[g], recvBytes));
        } else {
          recvBuffs[g] = sendBuffs[g];
        }
      }

      barrierSyncOn(reinterpret_cast<int32_t*>(sendBuffs[0]));

      const std::vector<uint16_t> helperSentinels(
          nActiveRanksPerGroup * recvCount, helperSentinel);
      for (int g = 0; g < nGroups; g++) {
        if (g == myActiveGroup) {
          initActiveBfloat16SendBuffer(sendBuffs[g], recvCount, myActiveIndex);
          HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, recvBytes));
        } else {
          HIPCHECK_TEST(hipMemcpy(
              sendBuffs[g],
              helperSentinels.data(),
              nActiveRanksPerGroup * recvBytes,
              hipMemcpyHostToDevice));
        }
      }
      HIPCHECK_TEST(hipStreamSynchronize(this->stream));

      std::vector<const void*> sendPtrs(nGroups);
      std::vector<void*> recvPtrs(nGroups);
      std::vector<size_t> recvCounts(nGroups, recvCount);
      for (int g = 0; g < nGroups; g++) {
        sendPtrs[g] = sendBuffs[g];
        recvPtrs[g] = recvBuffs[g];
      }

      const ncclResult_t result = callReduceScatterCompat(
          sendPtrs.data(),
          recvPtrs.data(),
          recvCounts.data(),
          ncclBfloat16,
          op,
          this->commFor(nActiveRanksPerGroup),
          this->stream,
          allActiveRanks,
          nActiveRanksPerGroup,
          nGroups);
      ASSERT_EQ(result, ncclSuccess)
          << "BF16 reduce-scatter failed at the " << boundaries[boundaryIndex]
          << " routing-threshold case";
      HIPCHECK_TEST(hipStreamSynchronize(this->stream));

      if (myActiveGroup >= 0) {
        const int32_t expectedSum =
            expectedReduceScatterSum(myActiveIndex, nActiveRanksPerGroup);
        const float expected = op == ncclAvg
            ? static_cast<float>(expectedSum) / nActiveRanksPerGroup
            : static_cast<float>(expectedSum);
        verifyBfloat16BufferEquals(
            recvBuffs[myActiveGroup],
            recvCount,
            bfloat16Bits(expected),
            myActiveGroup,
            boundaries[boundaryIndex]);
      }

      // Which route runs is internal to the collective and derived only from
      // the message size. Helper participation used to make that visible, but
      // helpers now stage into kernel-owned internal scratch and never write
      // the caller's buffer on any route, so assert the collective's own
      // selector instead: boundary 0 sits below the crossover and must stay
      // pure-direct, the other two must take the relay.
      const std::vector<size_t> routeRecvCounts(nGroups, recvCount);
      const rcclx::relay::ReduceScatterRoute expectedRoute = boundaryIndex == 0
          ? rcclx::relay::ReduceScatterRoute::PureDirect
          : rcclx::relay::ReduceScatterRoute::A2Relay;
      const rcclx::relay::ReduceScatterRoute actualRoute =
          rcclx::relay::selectReduceScatterRoute(
              nActiveRanksPerGroup,
              this->numRanks - nActiveRanksPerGroup,
              nGroups,
              routeRecvCounts.data(),
              sizeof(uint16_t));
      EXPECT_EQ(actualRoute, expectedRoute)
          << "internal route selection resolved to "
          << reduceScatterRouteName(actualRoute) << " at the "
          << boundaries[boundaryIndex]
          << " routing-threshold case (recvCount=" << recvCount
          << ", nGroups=" << nGroups << "), expected "
          << reduceScatterRouteName(expectedRoute);

      for (int g = 0; g < nGroups; g++) {
        HIPCHECK_TEST(hipFree(sendBuffs[g]));
        if (g == myActiveGroup) {
          HIPCHECK_TEST(hipFree(recvBuffs[g]));
        }
      }
    }
  }

  // Single-group (nGroups=1) reduce-scatter with 4 active ranks {0,1,2,3} and
  // ranks {4,5,6,7} acting as passthrough helpers. Exercises the A=4 kernel
  // path WITHOUT multi-group fusion (the single-group counterpart to the
  // 4Active_2Groups tests). Covers SUM/AVG and in-place/out-of-place.
  void runReduceScatterA4SingleGroup(ncclRedOp_t op, bool inPlace) {
    constexpr int nGroups = 1;
    constexpr int nActiveRanksPerGroup = 4;
    const size_t recvBytes = 64ULL * 1024 * 1024;
    const size_t recvCount = recvBytes / sizeof(int32_t);

    const int activeRanks[] = {0, 1, 2, 3};
    const int* allActiveRanks[] = {activeRanks};

    const bool isActive = this->globalRank < nActiveRanksPerGroup;
    const int myActiveIndex = this->globalRank; // 0..3 for active ranks
    const size_t helperBytes =
        static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;

    int32_t* sendBuff = nullptr;
    int32_t* recvBuff = nullptr;
    if (isActive) {
      HIPCHECK_TEST(hipMalloc(&sendBuff, helperBytes)); // A blocks
      if (inPlace) {
        // In-place: recvBuff aliases the owner's block inside sendBuff.
        recvBuff = sendBuff + static_cast<size_t>(myActiveIndex) * recvCount;
      } else {
        HIPCHECK_TEST(hipMalloc(&recvBuff, recvBytes));
      }
    } else {
      HIPCHECK_TEST(hipMalloc(&sendBuff, helperBytes));
      recvBuff = sendBuff;
    }

    barrierSyncOn(sendBuff);

    if (isActive) {
      initActiveSendBuffer(
          sendBuff, recvCount, myActiveIndex, nActiveRanksPerGroup);
      if (!inPlace) {
        HIPCHECK_TEST(hipMemset(recvBuff, 0, recvBytes));
      }
    } else {
      HIPCHECK_TEST(hipMemset(sendBuff, 0, helperBytes));
    }

    const void* sendPtrs[1] = {sendBuff};
    void* recvPtrs[1] = {recvBuff};
    size_t recvCounts[1] = {recvCount};

    ncclResult_t result = callReduceScatterCompat(
        sendPtrs,
        recvPtrs,
        recvCounts,
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
      const int32_t sum =
          expectedReduceScatterSum(myActiveIndex, nActiveRanksPerGroup);
      const int32_t expected = op == ncclAvg ? sum / nActiveRanksPerGroup : sum;
      verifyDeviceBufferEquals(
          recvBuff,
          recvCount,
          expected,
          0,
          "4-active single-group reduce-scatter mismatch");
    }

    HIPCHECK_TEST(hipFree(sendBuff));
    if (isActive && !inPlace) {
      HIPCHECK_TEST(hipFree(recvBuff));
    }
  }

  void runBfloat16A4SeededDirect(ncclRedOp_t op) {
    constexpr int nGroups = 2;
    constexpr int nActiveRanksPerGroup = 4;
    constexpr size_t recvCount = 4099;
    const size_t recvBytes = recvCount * sizeof(uint16_t);
    const size_t inputCount = nActiveRanksPerGroup * recvCount;

    TwoGroupFourActiveRanks groupConfig;
    const int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
    const int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

    uint16_t* sendBuffs[nGroups];
    uint16_t* recvBuffs[nGroups];
    for (int g = 0; g < nGroups; g++) {
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], inputCount * sizeof(uint16_t)));
      if (g == myActiveGroup) {
        HIPCHECK_TEST(hipMalloc(&recvBuffs[g], recvBytes));
      } else {
        recvBuffs[g] = sendBuffs[g];
      }
    }

    barrierSyncOn(reinterpret_cast<int32_t*>(sendBuffs[0]));

    std::vector<uint16_t> expectedInput(inputCount);
    for (int owner = 0; owner < nActiveRanksPerGroup; owner++) {
      for (size_t i = 0; i < recvCount; i++) {
        // At this magnitude BF16 has a 2-unit ULP. The even-mantissa seed
        // rounds each subsequent +1 back to itself, while reassociating the
        // three peer contributions would produce seed + 4.
        const float seed =
            256.0f + owner * 16.0f + static_cast<float>(i % 2) * 4.0f;
        const float value = myActiveIndex == 0 ? seed : 1.0f;
        expectedInput[static_cast<size_t>(owner) * recvCount + i] =
            bfloat16Bits(value);
      }
    }

    for (int g = 0; g < nGroups; g++) {
      if (g == myActiveGroup) {
        HIPCHECK_TEST(hipMemcpy(
            sendBuffs[g],
            expectedInput.data(),
            expectedInput.size() * sizeof(uint16_t),
            hipMemcpyHostToDevice));
        HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, recvBytes));
      } else {
        HIPCHECK_TEST(
            hipMemset(sendBuffs[g], 0, inputCount * sizeof(uint16_t)));
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

    const ncclResult_t result = callReduceScatterCompat(
        sendPtrs,
        recvPtrs,
        recvCounts,
        ncclBfloat16,
        op,
        this->commFor(nActiveRanksPerGroup),
        this->stream,
        groupConfig.allActiveRanks,
        nActiveRanksPerGroup,
        nGroups);
    ASSERT_EQ(result, ncclSuccess);
    HIPCHECK_TEST(hipStreamSynchronize(this->stream));

    std::vector<uint16_t> expectedOutput(recvCount);
    for (size_t i = 0; i < recvCount; i++) {
      float expected =
          256.0f + myActiveIndex * 16.0f + static_cast<float>(i % 2) * 4.0f;
      if (op == ncclAvg) {
        expected /= nActiveRanksPerGroup;
      }
      expectedOutput[i] = bfloat16Bits(expected);
    }

    verifyBfloat16BufferEquals(
        recvBuffs[myActiveGroup],
        expectedOutput,
        op == ncclAvg ? "A=4 seeded direct BF16 AVG mismatch"
                      : "A=4 seeded direct BF16 SUM mismatch");
    verifyBfloat16BufferEquals(
        sendBuffs[myActiveGroup],
        expectedInput,
        "A=4 seeded direct modified the out-of-place input");

    for (int g = 0; g < nGroups; g++) {
      HIPCHECK_TEST(hipFree(sendBuffs[g]));
      if (g == myActiveGroup) {
        HIPCHECK_TEST(hipFree(recvBuffs[g]));
      }
    }
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
        this->barrierComm,
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
      this->commFor(nActiveRanksPerGroup),
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
      this->commFor(nActiveRanksPerGroup),
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
      this->commFor(nActiveRanksPerGroup),
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
      this->commFor(nActiveRanksPerGroup),
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
  const int numChunks = numHelpers + 2;

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
      this->commFor(nActiveRanksPerGroup),
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

TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_4Groups_Bfloat16_Sum_FusedRoutingThreshold) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  runBfloat16A2RoutingThreshold(4, static_cast<size_t>(2) << 20, ncclSum);
}

TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_4Groups_Bfloat16_Avg_FusedRoutingThreshold) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  runBfloat16A2RoutingThreshold(4, static_cast<size_t>(2) << 20, ncclAvg);
}

TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_SingleGroup_Bfloat16_Sum_IndependentRoutingThreshold) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  runBfloat16A2RoutingThreshold(1, static_cast<size_t>(3) << 20, ncclSum);
}

TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_SingleGroup_Bfloat16_Avg_IndependentRoutingThreshold) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  runBfloat16A2RoutingThreshold(1, static_cast<size_t>(3) << 20, ncclAvg);
}

TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_4Groups_HeterogeneousRelayAndSmallPositive) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks
                 << " available";
  }

  const int nGroups = 4;
  const int nActiveRanksPerGroup = 2;
  const size_t relayThresholdCount =
      (static_cast<size_t>(2) << 20) / nActiveRanksPerGroup / sizeof(int32_t);
  const size_t recvCounts[nGroups] = {
      relayThresholdCount, relayThresholdCount, 513, 1023};

  Standard4GroupActiveRanks groupConfig;
  const int* const* allActiveRanks = groupConfig.allActiveRanks;

  const int myActiveGroup = this->globalRank / nActiveRanksPerGroup;
  const int myActiveIndex = this->globalRank % nActiveRanksPerGroup;

  int32_t* sendBuffs[nGroups];
  int32_t* recvBuffs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    const size_t recvBytes = recvCounts[g] * sizeof(int32_t);
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], nActiveRanksPerGroup * recvBytes));
      HIPCHECK_TEST(hipMalloc(&recvBuffs[g], recvBytes));
    } else {
      HIPCHECK_TEST(hipMalloc(&sendBuffs[g], nActiveRanksPerGroup * recvBytes));
      recvBuffs[g] = sendBuffs[g];
    }
  }

  barrierSyncOn(sendBuffs[0]);

  for (int g = 0; g < nGroups; g++) {
    const size_t recvBytes = recvCounts[g] * sizeof(int32_t);
    if (g == myActiveGroup) {
      initActiveSendBuffer(sendBuffs[g], recvCounts[g], myActiveIndex);
      HIPCHECK_TEST(hipMemset(recvBuffs[g], 0, recvBytes));
    } else {
      HIPCHECK_TEST(
          hipMemset(sendBuffs[g], 0, nActiveRanksPerGroup * recvBytes));
    }
  }

  const void* sendPtrs[nGroups];
  void* recvPtrs[nGroups];
  for (int g = 0; g < nGroups; g++) {
    sendPtrs[g] = sendBuffs[g];
    recvPtrs[g] = recvBuffs[g];
  }

  ncclResult_t result = callReduceScatterCompat(
      sendPtrs,
      recvPtrs,
      recvCounts,
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
      recvCounts[myActiveGroup],
      expectedReduceScatterSum(myActiveIndex),
      myActiveGroup,
      "Heterogeneous relay/direct reduce-scatter SUM mismatch");

  for (int g = 0; g < nGroups; g++) {
    HIPCHECK_TEST(hipFree(sendBuffs[g]));
    if (g == myActiveGroup) {
      HIPCHECK_TEST(hipFree(recvBuffs[g]));
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
      this->commFor(nActiveRanksPerGroup),
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
      this->commFor(nActiveRanksPerGroup),
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
    Correctness_4Active_2Groups_Bfloat16_SeededDirect_Sum) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  runBfloat16A4SeededDirect(ncclSum);
}

TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_4Active_2Groups_Bfloat16_SeededDirect_Avg) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  runBfloat16A4SeededDirect(ncclAvg);
}

/**
 * Test: single group (nGroups == 1) at 4 active ranks, both placements.
 *
 * nGroups == 1 makes the active ranks {0,1,2,3} and the helpers {4,5,6,7}
 * disjoint, which is what puts the flat relay on its software-pipelined
 * schedule: the offload scatter and the helper's reduced return share a group
 * so each cross link runs duplex, and the helper's reduce runs per tile between
 * the group that receives it and the group that ships it. 64 MB is above the
 * offload threshold and deep enough for the cost model to tile, so this is the
 * only coverage of that path -- every other 4-active case runs 2 groups, where
 * the relay stays unpipelined.
 */
class ShardedRelayReduceScatterSingleGroupA4Test
    : public ShardedRelayMultiGroupReduceScatterTest {
 protected:
  void runSingleGroupA4Case(bool inPlace) {
    const int nGroups = 1;
    const int nActiveRanksPerGroup = 4;
    const size_t recvBytes = 64ULL * 1024 * 1024;
    const size_t recvCount = recvBytes / sizeof(int32_t);
    const size_t inBytes =
        static_cast<size_t>(nActiveRanksPerGroup) * recvBytes;

    const int activeRanks[] = {0, 1, 2, 3};
    const int* allActiveRanks[] = {activeRanks};
    const bool isActive = this->globalRank < nActiveRanksPerGroup;
    const int myActiveIndex = this->globalRank;

    // Helpers hand in a placeholder; the kernel stages into its own scratch.
    int32_t* sendBuff = nullptr;
    int32_t* recvBuff = nullptr;
    HIPCHECK_TEST(hipMalloc(&sendBuff, inBytes));
    if (isActive && !inPlace) {
      HIPCHECK_TEST(hipMalloc(&recvBuff, recvBytes));
    }

    barrierSyncOn(sendBuff);

    HIPCHECK_TEST(hipMemset(sendBuff, 0, inBytes));
    if (isActive) {
      initActiveSendBuffer(
          sendBuff, recvCount, myActiveIndex, nActiveRanksPerGroup);
      if (!inPlace) {
        HIPCHECK_TEST(hipMemset(recvBuff, 0, recvBytes));
      }
    }
    // In-place: the output aliases this rank's own contribution block.
    int32_t* out = inPlace
        ? sendBuff + static_cast<size_t>(myActiveIndex) * recvCount
        : recvBuff;

    const void* sendPtrs[1] = {sendBuff};
    void* recvPtrs[1] = {isActive ? out : sendBuff};
    size_t recvCounts[1] = {recvCount};

    const ncclResult_t result = callReduceScatterCompat(
        sendPtrs,
        recvPtrs,
        recvCounts,
        ncclInt32,
        ncclSum,
        this->commFor(nActiveRanksPerGroup),
        this->stream,
        allActiveRanks,
        nActiveRanksPerGroup,
        nGroups);
    ASSERT_EQ(result, ncclSuccess);
    HIPCHECK_TEST(hipStreamSynchronize(this->stream));

    if (isActive) {
      verifyDeviceBufferEquals(
          out,
          recvCount,
          expectedReduceScatterSum(myActiveIndex, nActiveRanksPerGroup),
          0,
          "4-active single-group reduce-scatter SUM mismatch");
    }

    HIPCHECK_TEST(hipFree(sendBuff));
    if (recvBuff != nullptr) {
      HIPCHECK_TEST(hipFree(recvBuff));
    }
  }
};

TEST_F(
    ShardedRelayReduceScatterSingleGroupA4Test,
    Correctness_OutOfPlace_64MB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runSingleGroupA4Case(/*inPlace=*/false);
}

TEST_F(ShardedRelayReduceScatterSingleGroupA4Test, Correctness_InPlace_64MB) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runSingleGroupA4Case(/*inPlace=*/true);
}

/**
 * Test: the pipelined relay's block layout fits inside recvCount at EVERY
 * geometry buildShardedRelayRankConfig() accepts, not just the A = H = 4 one
 * the 8-rank cases above exercise.
 *
 * The GPU cases here are pinned to 8 ranks, so a single-group A = 4 call always
 * has exactly 4 helpers; the combinations a 9-to-16-rank comm would produce
 * (A = 4 with H = 5..8, or A = 8 with H = 8) cannot be built from this harness.
 * What those geometries break is arithmetic, not scheduling:
 * shardedRelayReduceScatterFlatPipelined() takes directSz = rc - H*T*w and then
 * directSize(T) = directSz - T*(A-1)*w, so if totalUnits(T) is smaller than
 * H*T + (A-1)*T + 1 both subtractions underflow as size_t and a wild count
 * reaches ncclSend/ncclRecv. Assert the invariant against the shape directly
 * over the whole accepted (A, H) space, which needs no ranks at all.
 */
TEST(ShardedRelayReduceScatterFanoutShape, LayoutFitsAtEveryAcceptedGeometry) {
  // Mirrors SHARDED_RELAY_MAX_ACTIVE / SHARDED_RELAY_MAX_HELPERS.
  constexpr int kMaxActive = 8;
  constexpr int kMaxHelpers = 8;

  for (int a = 2; a <= kMaxActive; a *= 2) {
    for (int h = 1; h <= kMaxHelpers; h++) {
      const rcclx::relay::RelayPipelineShape shape =
          rcclx::relay::relayShapeFanout(a, h);
      for (int t = 1; t <= 8; t *= 2) {
        const int totalUnits = shape.totalPerTile * t + shape.totalFixed;
        // Offload region, plus the T heavy direct chunks, plus at least one
        // unit for the last chunk to absorb.
        const int consumed = h * t + (a - 1) * t + 1;
        EXPECT_LE(consumed, totalUnits)
            << "pipelined block layout overruns recvCount at A=" << a
            << " H=" << h << " T=" << t << ": needs " << consumed
            << " units out of " << totalUnits;
      }
    }
  }

  // The shipped 8-GPU geometry must still resolve to the constants the perf
  // numbers were measured at, so the parameterization is a generalization and
  // not a change.
  const rcclx::relay::RelayPipelineShape shipped =
      rcclx::relay::relayShapeFanout(4, 4);
  EXPECT_EQ(shipped.linkPerTile, 3);
  EXPECT_EQ(shipped.linkFixed, 1);
  EXPECT_EQ(shipped.totalPerTile, 7);
  EXPECT_EQ(shipped.totalFixed, 1);
}

/**
 * Test: single group at 2 active ranks, large enough to overlap the owner
 * reduce, with a POSITION-DEPENDENT fill.
 *
 * Above kRelayOverlapReduceMinBytes the pipelined 2-active relay stops reducing
 * once at the end and instead reduces each pipeline region on a side stream as
 * that region lands. That required re-indexing the shipped block from
 * helper-major to stage-major, so region k holds exactly what group k receives.
 *
 * Every other reduce-scatter test fills a contribution block with a CONSTANT
 * value, which means a region that arrives at the wrong offset still sums to
 * the right number and the bug is invisible. These cases fill position
 * dependently instead, so any permutation, shift, overlap or omission of a
 * region changes the output and fails.
 *
 * Element i of every block on active rank r holds (r + 1) * kBase +
 * (i % kPeriod). Owner m's output element i is the sum over r, which at A == 2
 * is 3 * kBase + 2 * (i % kPeriod) -- always even, so AVG is exact in integer
 * arithmetic and a divisor applied twice, or to the wrong region, is caught.
 * kPeriod is coprime with the 128-element chunk alignment, so a misplacement by
 * any whole number of chunks shifts the pattern rather than aliasing onto it.
 */
class ShardedRelayReduceScatterOverlapA2Test
    : public ShardedRelayMultiGroupReduceScatterTest {
 protected:
  static constexpr int32_t kBase = 1000;
  static constexpr int32_t kPeriod = 977;

  static int32_t fillValue(int activeIndex, size_t i) {
    return static_cast<int32_t>(activeIndex + 1) * kBase +
        static_cast<int32_t>(i % static_cast<size_t>(kPeriod));
  }

  void runOverlapCase(
      bool inPlace,
      ncclRedOp_t op,
      size_t recvCountOverride = 0,
      size_t misalignElemsOnOddRanks = 0,
      int activeRanksPerGroup = 2) {
    const int nGroups = 1;
    const int A = activeRanksPerGroup;
    // Default: A * recvBytes reaches kRelayOverlapReduceMinBytes (256 MiB) so
    // the side-stream path engages. recvCountOverride instead pins a small
    // count so the one-shot IPC path is exercised (it needs A * recvCount *
    // elemSize <= kRelayOneShotMaxBytes).
    const size_t recvCount = (recvCountOverride != 0)
        ? recvCountOverride
        : (128ULL * 1024 * 1024) / sizeof(int32_t);
    const size_t recvBytes = recvCount * sizeof(int32_t);
    const size_t inBytes = static_cast<size_t>(A) * recvBytes;

    const int activeRanks[] = {0, 1, 2, 3};
    const int* allActiveRanks[] = {activeRanks};
    const bool isActive = this->globalRank < A;
    const int myActiveIndex = this->globalRank;

    // Shift only the ODD active rank's buffers, so the two peers disagree on
    // whether 16-byte accesses are usable. hipMalloc always returns a
    // 16-byte-aligned pointer, so every other case here has both ranks aligned.
    const size_t skew =
        (misalignElemsOnOddRanks != 0 && (myActiveIndex % 2) == 1)
        ? misalignElemsOnOddRanks
        : 0;
    const size_t skewBytes = skew * sizeof(int32_t);

    int32_t* sendAlloc = nullptr;
    int32_t* recvAlloc = nullptr;
    HIPCHECK_TEST(hipMalloc(&sendAlloc, inBytes + skewBytes));
    if (isActive && !inPlace) {
      HIPCHECK_TEST(hipMalloc(&recvAlloc, recvBytes + skewBytes));
    }
    int32_t* sendBuff = sendAlloc + skew;
    int32_t* recvBuff = (recvAlloc != nullptr) ? recvAlloc + skew : nullptr;

    barrierSyncOn(sendBuff);

    HIPCHECK_TEST(hipMemset(sendBuff, 0, inBytes));
    if (isActive) {
      std::vector<int32_t> host(static_cast<size_t>(A) * recvCount);
      for (int j = 0; j < A; j++) {
        int32_t* block = host.data() + static_cast<size_t>(j) * recvCount;
        for (size_t i = 0; i < recvCount; i++) {
          block[i] = fillValue(myActiveIndex, i);
        }
      }
      HIPCHECK_TEST(
          hipMemcpy(sendBuff, host.data(), inBytes, hipMemcpyHostToDevice));
      if (!inPlace) {
        HIPCHECK_TEST(hipMemset(recvBuff, 0, recvBytes));
      }
    }

    int32_t* out = inPlace
        ? sendBuff + static_cast<size_t>(myActiveIndex) * recvCount
        : recvBuff;

    const void* sendPtrs[1] = {sendBuff};
    void* recvPtrs[1] = {isActive ? out : sendBuff};
    size_t recvCounts[1] = {recvCount};

    const ncclResult_t result = callReduceScatterCompat(
        sendPtrs,
        recvPtrs,
        recvCounts,
        ncclInt32,
        op,
        this->commFor(activeRanksPerGroup),
        this->stream,
        allActiveRanks,
        A,
        nGroups);
    ASSERT_EQ(result, ncclSuccess);
    HIPCHECK_TEST(hipStreamSynchronize(this->stream));

    if (isActive) {
      std::vector<int32_t> got(recvCount);
      HIPCHECK_TEST(
          hipMemcpy(got.data(), out, recvBytes, hipMemcpyDeviceToHost));
      size_t mismatches = 0;
      size_t firstBad = 0;
      for (size_t i = 0; i < recvCount; i++) {
        int32_t sum = 0;
        for (int r = 0; r < A; r++) {
          sum += fillValue(r, i);
        }
        const int32_t want = (op == ncclAvg) ? sum / A : sum;
        if (got[i] != want) {
          if (mismatches == 0) {
            firstBad = i;
          }
          mismatches++;
        }
      }
      ASSERT_EQ(mismatches, 0u)
          << A << "-active single-group reduce-scatter mismatch: " << mismatches
          << " of " << recvCount << " elements, first at index " << firstBad;
    }

    HIPCHECK_TEST(hipFree(sendBuff));
    if (recvBuff != nullptr) {
      HIPCHECK_TEST(hipFree(recvBuff));
    }
  }
};

// The four cases below sit BELOW kRelayOneShotMaxBytes (1 MiB of
// per-active-rank input), where the one-shot IPC kernel replaces the ncclGroup
// + reduce pair entirely: it pushes into the peer's staging, handshakes on a
// per-block flag, and reduces in registers, in a single launch.
//
// This fixture is the right home because its fill is POSITION-DEPENDENT. The
// older A=2 tests fill each block with a constant, which cannot catch an offset
// or slot permutation -- and the one-shot path is exactly where such a bug
// would live, since it indexes peer staging by active index.
//
// 65536 int32 = 256 KiB per rank, so 512 KiB of input: inside the gate, and
// 16-byte aligned so the vectorized bulk path runs. 65535 is deliberately NOT a
// multiple of 16/sizeof(int32_t), so it drives the scalar tail and the
// misaligned-pointer fallback instead.
TEST_F(
    ShardedRelayReduceScatterOverlapA2Test,
    Correctness_OneShot_OutOfPlace_Sum) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runOverlapCase(/*inPlace=*/false, ncclSum, /*recvCountOverride=*/65536);
}

TEST_F(
    ShardedRelayReduceScatterOverlapA2Test,
    Correctness_OneShot_InPlace_Sum) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runOverlapCase(/*inPlace=*/true, ncclSum, /*recvCountOverride=*/65536);
}

TEST_F(
    ShardedRelayReduceScatterOverlapA2Test,
    Correctness_OneShot_InPlace_Avg) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runOverlapCase(/*inPlace=*/true, ncclAvg, /*recvCountOverride=*/65536);
}

TEST_F(
    ShardedRelayReduceScatterOverlapA2Test,
    Correctness_OneShot_OutOfPlace_Sum_UnalignedTail) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runOverlapCase(/*inPlace=*/false, ncclSum, /*recvCountOverride=*/65535);
}

// The two cases below give ONE of the two peers 16-byte-misaligned caller
// buffers and leave the other aligned, so the peers disagree about whether
// vectorized accesses are usable. hipMalloc always returns a 16-byte-aligned
// pointer, so in every other case here both ranks vectorize and that mixed
// state is never reached at all.
//
// The count is chosen so the two ranks would also PARTITION differently if the
// alignment decision were allowed to pick the block range. Vector partitioning
// splits rc/kEvec vectors across gridDim blocks, element partitioning splits
// rc, and the two agree whenever ceil((rc/kEvec)/G)*kEvec == ceil(rc/G) --
// which is the case at 65536 and 65535 (both 1024). At 16388 int32 with
// kEvec = 4 and G = 64 the vector stride is ceil(4097/64)*4 = 260 against an
// element stride of ceil(16388/64) = 257, so the ranges diverge from block 1
// onward. 16389 adds a ragged rc % kEvec tail on top of the same skew.
//
// WHAT THESE DO AND DO NOT GUARANTEE. They cover the mixed-alignment path --
// one rank on the vector loops, its peer on the element-wise loops -- and they
// would fail outright on a partition that overlapped or left a gap. They are
// NOT a regression test for the ordering hazard that motivated deriving the
// range from rank-agreed values only. With a divergent partition, block b
// reduces a range wider than the one block b's flag covers, so it can read
// staging its peer has not pushed yet. That read is a race and it is not
// reliably observable here: the window is a few elements at a block seam, the
// peer's block is doing the same work at the same time, and this fixture writes
// the same fill on every call, so a premature read returns the value that is
// about to be written anyway. Confirmed empirically -- with the
// alignment-dependent partition restored, both cases below still pass. The
// invariant is held by the kernel deriving [begin, end) from rc, kEvec and
// gridDim alone; do not read a green run here as licence to relax that.
TEST_F(
    ShardedRelayReduceScatterOverlapA2Test,
    Correctness_OneShot_OutOfPlace_Sum_AsymmetricAlignment) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runOverlapCase(
      /*inPlace=*/false,
      ncclSum,
      /*recvCountOverride=*/16388,
      /*misalignElemsOnOddRanks=*/1);
}

TEST_F(
    ShardedRelayReduceScatterOverlapA2Test,
    Correctness_OneShot_InPlace_Avg_AsymmetricAlignmentUnalignedTail) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runOverlapCase(
      /*inPlace=*/true,
      ncclAvg,
      /*recvCountOverride=*/16389,
      /*misalignElemsOnOddRanks=*/1);
}

// A=4 counterparts of the one-shot cases above. The one-shot kernel indexes
// peer staging by ACTIVE INDEX, so a slot permutation is the characteristic
// bug, and at A=4 there are three sources to get wrong instead of one. The A=4
// fixture elsewhere in this file fills each block with a constant, which cannot
// see such a bug -- these reuse this fixture's position-dependent fill instead.
//
// AVG stays integer-exact at A=4: the per-element sum is
// kBase*A*(A+1)/2 + A*(i%kPeriod), so dividing by 4 gives 2500 + (i%kPeriod).
TEST_F(
    ShardedRelayReduceScatterOverlapA2Test,
    Correctness_OneShot_A4_OutOfPlace_Sum) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runOverlapCase(
      /*inPlace=*/false,
      ncclSum,
      /*recvCountOverride=*/32768,
      /*misalignElemsOnOddRanks=*/0,
      /*activeRanksPerGroup=*/4);
}

TEST_F(
    ShardedRelayReduceScatterOverlapA2Test,
    Correctness_OneShot_A4_InPlace_Avg) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runOverlapCase(
      /*inPlace=*/true,
      ncclAvg,
      /*recvCountOverride=*/32768,
      /*misalignElemsOnOddRanks=*/0,
      /*activeRanksPerGroup=*/4);
}

TEST_F(
    ShardedRelayReduceScatterOverlapA2Test,
    Correctness_OneShot_A4_OutOfPlace_Sum_UnalignedTail) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runOverlapCase(
      /*inPlace=*/false,
      ncclSum,
      /*recvCountOverride=*/32767,
      /*misalignElemsOnOddRanks=*/0,
      /*activeRanksPerGroup=*/4);
}

TEST_F(ShardedRelayReduceScatterOverlapA2Test, Correctness_OutOfPlace_Sum) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runOverlapCase(/*inPlace=*/false, ncclSum);
}

TEST_F(ShardedRelayReduceScatterOverlapA2Test, Correctness_InPlace_Sum) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runOverlapCase(/*inPlace=*/true, ncclSum);
}

TEST_F(ShardedRelayReduceScatterOverlapA2Test, Correctness_OutOfPlace_Avg) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runOverlapCase(/*inPlace=*/false, ncclAvg);
}

TEST_F(ShardedRelayReduceScatterOverlapA2Test, Correctness_InPlace_Avg) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }
  runOverlapCase(/*inPlace=*/true, ncclAvg);
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
      this->commFor(nActiveRanksPerGroup),
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
      this->commFor(nActiveRanksPerGroup),
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

// Single-group (nGroups=1) A=4 reduce-scatter. Ranks {0,1,2,3} are active and
// {4,5,6,7} are passthrough helpers, exercising the A=4 path without the
// multi-group fusion the 4Active_2Groups tests cover.
TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_4Active_SingleGroup_InPlace) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  runReduceScatterA4SingleGroup(ncclSum, /*inPlace=*/true);
}

TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_4Active_SingleGroup_OutOfPlace) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  runReduceScatterA4SingleGroup(ncclSum, /*inPlace=*/false);
}

TEST_F(
    ShardedRelayMultiGroupReduceScatterTest,
    Correctness_4Active_SingleGroup_Avg) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks, but got " << this->numRanks;
  }

  runReduceScatterA4SingleGroup(ncclAvg, /*inPlace=*/false);
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
      this->commFor(nActiveRanksPerGroup),
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
      this->commFor(nActiveRanksPerGroup),
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

// ===========================================================================
// LOW PRECISION (fp8e4m3 wire format)
// ===========================================================================
//
// Every case here asserts that low precision actually ENGAGED or actually
// DECLINED, via the counters, rather than trusting the flag. The gate declines
// silently on several independent grounds, so an "LP" run that quietly fell
// back to full precision produces exactly the numbers a passing LP run
// produces -- the counter is the only thing that tells them apart.
//
// The comparators stay EXACT. Each active rank fills every block it contributes
// with a single constant, so every 128-element wire block has that constant as
// its absmax; the power-of-two normalization target then makes
// quantize/dequantize an identity, and a sum of equal values behaves the same
// way. These cases are therefore a genuine detector for a wrong scale, a wrong
// block boundary or a dropped scale. Do not loosen them.
//
// Reduce-scatter's own hazard, which no allreduce case can catch: the halves of
// the send buffer are DIFFERENT blocks bound for different places, and
// sendBlockOffset selects the one shipped to the partner. Filling each block
// with a different constant means a wire offset computed against the wrong
// block, or a quantize hoisted over the wrong span, changes the answer instead
// of cancelling out.
class ShardedRelayReduceScatterLowPrecisionTest
    : public ShardedRelayMultiGroupReduceScatterTest {
 protected:
  static constexpr int kActive = 2;
  static constexpr int kGroups = 4;

  // 2 Mi elements per output block = 16 MiB of fp32 input per active rank,
  // comfortably above the low-precision size threshold and a whole number of
  // 128-element blocks.
  static constexpr size_t kLpCount = 2ULL * 1024 * 1024;

  // bf16 is carried as raw bits, the convention the rest of this file uses, so
  // that no host-side bf16 arithmetic or comparison is involved. These
  // overloads are selected by a value-initialized tag of the storage type.
  static float encodeAs(float v, float) {
    return v;
  }
  static int32_t encodeAs(float v, int32_t) {
    return static_cast<int32_t>(v);
  }
  static uint16_t encodeAs(float v, uint16_t) {
    return bfloat16Bits(v);
  }

  template <typename T>
  static T encode(float v) {
    return encodeAs(v, T{});
  }

  // Active rank r's input block b holds this value, so the expected output for
  // whichever rank owns block b is the sum over r -- a different constant per
  // owner, which is what makes a swapped block detectable. All four values
  // (5, 6, 9, 10) and both sums (14, 16) are exactly representable in fp32,
  // bf16 and, after the power-of-two normalization, e4m3.
  static float blockValue(int activeIndex, int blockIndex) {
    return static_cast<float>((activeIndex + 1) * 4 + (blockIndex + 1));
  }

  static float expectedOwnerValue(int blockIndex) {
    float sum = 0.0f;
    for (int r = 0; r < kActive; r++) {
      sum += blockValue(r, blockIndex);
    }
    return sum;
  }

  struct Buffers {
    std::vector<void*> sendMem;
    void* recvMem{nullptr};
    const void* sendPtrs[kGroups];
    void* recvPtrs[kGroups];
    size_t counts[kGroups];
    int nGroups{kGroups};
    int myActiveGroup{0};
    int myActiveIndex{0};
  };

  // Out-of-place buffers, one contiguous send/recv pair per group. Helper
  // groups pass their send allocation as the output placeholder, as the rest of
  // this suite does; only the active group's output is read.
  template <typename T>
  void makeBuffers(size_t count, int nGroups, Buffers& b) {
    b = Buffers{};
    b.nGroups = nGroups;
    b.myActiveGroup = this->globalRank / kActive;
    b.myActiveIndex = this->globalRank % kActive;
    b.sendMem.resize(nGroups);
    const size_t elems = static_cast<size_t>(kActive) * count;
    for (int g = 0; g < nGroups; g++) {
      HIPCHECK_TEST(hipMalloc(&b.sendMem[g], elems * sizeof(T)));
      if (g == b.myActiveGroup) {
        std::vector<T> host(elems);
        for (int blk = 0; blk < kActive; blk++) {
          std::fill(
              host.begin() + static_cast<ptrdiff_t>(blk * count),
              host.begin() + static_cast<ptrdiff_t>((blk + 1) * count),
              encode<T>(blockValue(b.myActiveIndex, blk)));
        }
        HIPCHECK_TEST(hipMemcpy(
            b.sendMem[g],
            host.data(),
            elems * sizeof(T),
            hipMemcpyHostToDevice));
      } else {
        HIPCHECK_TEST(hipMemset(b.sendMem[g], 0, elems * sizeof(T)));
      }
      b.sendPtrs[g] = b.sendMem[g];
      b.counts[g] = count;
    }
    HIPCHECK_TEST(hipMalloc(&b.recvMem, count * sizeof(T)));
    HIPCHECK_TEST(hipMemset(b.recvMem, 0, count * sizeof(T)));
    for (int g = 0; g < nGroups; g++) {
      b.recvPtrs[g] = (g == b.myActiveGroup) ? b.recvMem : b.sendMem[g];
    }
  }

  void freeBuffers(Buffers& b) {
    for (void* p : b.sendMem) {
      HIPCHECK_TEST(hipFree(p));
    }
    HIPCHECK_TEST(hipFree(b.recvMem));
    b.sendMem.clear();
    b.recvMem = nullptr;
  }

  // Exact, element by element. Only the ranks that own an output block check
  // it; for a single-group call the rest are helpers with no output of their
  // own.
  template <typename T>
  void expectOutputEquals(const Buffers& b, size_t count, float wantValue) {
    if (b.myActiveGroup >= b.nGroups) {
      return;
    }
    const T want = encode<T>(wantValue);
    std::vector<T> got(count);
    HIPCHECK_TEST(hipMemcpy(
        got.data(), b.recvMem, count * sizeof(T), hipMemcpyDeviceToHost));
    size_t reported = 0;
    for (size_t i = 0; i < count && reported < 8; i++) {
      if (got[i] != want) {
        reported++;
        ADD_FAILURE() << "R" << this->globalRank << ": element " << i
                      << " differs (raw comparison against the encoding of "
                      << wantValue << ")";
      }
    }
  }

  ncclResult_t
  call(const Buffers& b, ncclDataType_t dt, ncclRedOp_t op, int lowPrecision) {
    Standard4GroupActiveRanks layout;
    return callReduceScatterCompat(
        b.sendPtrs,
        b.recvPtrs,
        b.counts,
        dt,
        op,
        this->commFor(kActive),
        this->stream,
        layout.allActiveRanks,
        kActive,
        b.nGroups,
        lowPrecision);
  }
};

TEST_F(ShardedRelayReduceScatterLowPrecisionTest, ConstantBlocksAreBitExact) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  Buffers b;
  makeBuffers<float>(kLpCount, kGroups, b);
  barrierSyncOn(nullptr);

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(call(b, ncclFloat32, ncclSum, /*lowPrecision=*/1), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  expectOutputEquals<float>(b, kLpCount, expectedOwnerValue(b.myActiveIndex));
  EXPECT_GT(rcclx::relay::lpEngageCount(), 0u)
      << "low precision never engaged, so this case proved nothing";
  EXPECT_EQ(rcclx::relay::lpDeclineCount(), 0u);
  freeBuffers(b);
}

TEST_F(ShardedRelayReduceScatterLowPrecisionTest, Bfloat16IsBitExact) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  // The other supported dtype, and the one the wire format was built for: bf16
  // in, fp8 on the wire, bf16 out, with a single rounding on the whole path
  // because reduce-scatter's helper never requantizes.
  Buffers b;
  makeBuffers<uint16_t>(kLpCount, kGroups, b);
  barrierSyncOn(nullptr);

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(call(b, ncclBfloat16, ncclSum, /*lowPrecision=*/1), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  expectOutputEquals<uint16_t>(
      b, kLpCount, expectedOwnerValue(b.myActiveIndex));
  EXPECT_GT(rcclx::relay::lpEngageCount(), 0u);
  EXPECT_EQ(rcclx::relay::lpDeclineCount(), 0u);
  freeBuffers(b);
}

TEST_F(ShardedRelayReduceScatterLowPrecisionTest, AvgAppliesTheDivisorOnce) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  // ncclAvg is where reduce-scatter parts company with allreduce. Its helper is
  // a PURE RELAY, so the divisor belongs solely to the active rank's closing
  // reduce. A divisor also applied at the helper -- which is correct for
  // allreduce and wrong here -- would halve the relayed chunks and leave the
  // two direct chunks right, and the exact comparator catches that at the first
  // relayed element. 14 / 2 and 16 / 2 are both exact.
  Buffers b;
  makeBuffers<float>(kLpCount, kGroups, b);
  barrierSyncOn(nullptr);

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(call(b, ncclFloat32, ncclAvg, /*lowPrecision=*/1), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  expectOutputEquals<float>(
      b,
      kLpCount,
      expectedOwnerValue(b.myActiveIndex) / static_cast<float>(kActive));
  EXPECT_GT(rcclx::relay::lpEngageCount(), 0u);
  freeBuffers(b);
}

TEST_F(ShardedRelayReduceScatterLowPrecisionTest, DeclinesOnUnsupportedDtype) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  // ncclInt32 is not a low-precision dtype, so the flag must fall through to
  // full precision and the answer must be exactly the full-precision answer.
  Buffers b;
  makeBuffers<int32_t>(kLpCount, kGroups, b);
  barrierSyncOn(nullptr);

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(call(b, ncclInt32, ncclSum, /*lowPrecision=*/1), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  expectOutputEquals<int32_t>(b, kLpCount, expectedOwnerValue(b.myActiveIndex));
  EXPECT_EQ(rcclx::relay::lpEngageCount(), 0u);
  EXPECT_GT(rcclx::relay::lpDeclineCount(rcclx::relay::LpDecline::Dtype), 0u);
  freeBuffers(b);
}

TEST_F(
    ShardedRelayReduceScatterLowPrecisionTest,
    DeclinesOnCountThatIsNotWholeBlocks) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  // One element past a whole number of 128-element blocks. sendBlockOffset is a
  // raw per-group count here, so an unaligned count breaks additivity in the
  // MIDDLE of the send buffer, not only in its tail -- the gate must refuse.
  const size_t count = kLpCount + 1;
  Buffers b;
  makeBuffers<float>(count, kGroups, b);
  barrierSyncOn(nullptr);

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(call(b, ncclFloat32, ncclSum, /*lowPrecision=*/1), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  expectOutputEquals<float>(b, count, expectedOwnerValue(b.myActiveIndex));
  EXPECT_EQ(rcclx::relay::lpEngageCount(), 0u);
  EXPECT_GT(
      rcclx::relay::lpDeclineCount(rcclx::relay::LpDecline::Alignment), 0u);
  freeBuffers(b);
}

TEST_F(
    ShardedRelayReduceScatterLowPrecisionTest,
    InterleavesWithFullPrecision) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  // The behaviour an env var could not express, and the whole point of making
  // this a per-call argument: two calls on the SAME communicator, one low
  // precision and one not, with no state leaking from the first into the
  // second.
  Buffers b;
  makeBuffers<float>(kLpCount, kGroups, b);
  barrierSyncOn(nullptr);

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(call(b, ncclFloat32, ncclSum, /*lowPrecision=*/1), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));
  expectOutputEquals<float>(b, kLpCount, expectedOwnerValue(b.myActiveIndex));
  ASSERT_GT(rcclx::relay::lpEngageCount(), 0u);

  const uint64_t engagedBefore = rcclx::relay::lpEngageCount();
  HIPCHECK_TEST(hipMemset(b.recvMem, 0, kLpCount * sizeof(float)));
  barrierSyncOn(nullptr);
  ASSERT_EQ(call(b, ncclFloat32, ncclSum, /*lowPrecision=*/0), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));
  expectOutputEquals<float>(b, kLpCount, expectedOwnerValue(b.myActiveIndex));
  EXPECT_EQ(rcclx::relay::lpEngageCount(), engagedBefore)
      << "a full-precision call must not engage low precision";
  freeBuffers(b);
}

TEST_F(
    ShardedRelayReduceScatterLowPrecisionTest,
    SingleGroupPipelinedIsBitExact) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  // The 4-group fixture above cannot reach the pipelined schedule:
  // relayPipelineTiles() returns 1 whenever nGroups != 1. So this case is
  // single-group, at a size the depth selector pipelines -- asserted rather
  // than assumed, because a size that quietly failed to pipeline would silently
  // re-test the schedule the other cases already cover.
  const size_t count = 4ULL * 1024 * 1024;
  const int numHelpers = 8 - kActive;
  ASSERT_GT(
      rcclx::relay::relayPipelineTiles(
          1, rcclx::relay::relayShapeA2(numHelpers), count, sizeof(float)),
      1);

  Buffers b;
  makeBuffers<float>(count, /*nGroups=*/1, b);
  barrierSyncOn(nullptr);

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(call(b, ncclFloat32, ncclSum, /*lowPrecision=*/1), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  expectOutputEquals<float>(b, count, expectedOwnerValue(b.myActiveIndex));
  EXPECT_GT(rcclx::relay::lpEngageCount(), 0u)
      << "low precision never engaged, so this case proved nothing";
  EXPECT_EQ(rcclx::relay::lpDeclineCount(), 0u);
  freeBuffers(b);
}

TEST_F(
    ShardedRelayReduceScatterLowPrecisionTest,
    SingleGroupPipelinedAvgIsBitExact) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  // The pipelined schedule reduces PER REGION on a side stream rather than once
  // at the end, so the divisor is applied T + 1 times over disjoint spans. This
  // pins that it is applied exactly once per element: a divisor that also
  // landed at the helper, or one applied twice to an overlapping span, breaks
  // the exact comparator.
  const size_t count = 4ULL * 1024 * 1024;
  Buffers b;
  makeBuffers<float>(count, /*nGroups=*/1, b);
  barrierSyncOn(nullptr);

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(call(b, ncclFloat32, ncclAvg, /*lowPrecision=*/1), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  expectOutputEquals<float>(
      b,
      count,
      expectedOwnerValue(b.myActiveIndex) / static_cast<float>(kActive));
  EXPECT_GT(rcclx::relay::lpEngageCount(), 0u);
  freeBuffers(b);
}

// ===========================================================================
// LOW PRECISION, FLAT A>2 PATH
// ===========================================================================
//
// Same exactness discipline as the 2-active fixture above, at the 2-group
// 4-active geometry, which is what reaches shardedRelayReduceScatterFlat.
//
// Every value here is a distinct per-(source, owner) constant, so a wrong block
// index, a wrong helper slot or a wrong owner changes the answer. Exactness
// survives even though the offload region rounds TWICE -- once on the way up
// and again when the helper requantizes its sum -- because each 128-element
// wire block is constant, and a constant block's absmax normalization makes the
// code exactly 128 and the round trip exact for ANY value. That is the property
// kLpNormalizeMax being a power of two buys, and it is what makes these exact
// comparators legitimate rather than lucky.
class ShardedRelayReduceScatterFlatLowPrecisionTest
    : public ShardedRelayMultiGroupReduceScatterTest {
 protected:
  static constexpr int kActive = 4;
  static constexpr int kGroups = 2;

  // 4 Mi elements per output block = 64 MiB of fp32 input per rank. The flat
  // route's own metric is A * recvCount * elementSize = 64 MiB, which is above
  // both the offload crossover (~48 MB) and the low-precision threshold; the
  // test asserts the offload route rather than assuming it.
  static constexpr size_t kLpCount = 4ULL * 1024 * 1024;

  // Source s's contribution to owner j. Distinct in both indices.
  static float blockValue(int sourceIndex, int ownerIndex) {
    return static_cast<float>((sourceIndex + 1) * 10 + (ownerIndex + 1));
  }

  static float expectedOwnerValue(int ownerIndex) {
    float sum = 0.0f;
    for (int s = 0; s < kActive; s++) {
      sum += blockValue(s, ownerIndex);
    }
    return sum;
  }

  struct Buffers {
    void* sendMem[kGroups]{};
    void* recvMem{nullptr};
    const void* sendPtrs[kGroups]{};
    void* recvPtrs[kGroups]{};
    size_t counts[kGroups]{};
    int myActiveGroup{0};
    int myActiveIndex{0};
  };

  void makeBuffers(size_t count, Buffers& b) {
    b = Buffers{};
    b.myActiveGroup = this->globalRank / kActive;
    b.myActiveIndex = this->globalRank % kActive;
    const size_t inputCount = static_cast<size_t>(kActive) * count;

    for (int g = 0; g < kGroups; g++) {
      HIPCHECK_TEST(hipMalloc(&b.sendMem[g], inputCount * sizeof(float)));
      if (g == b.myActiveGroup) {
        std::vector<float> host(inputCount);
        for (int owner = 0; owner < kActive; owner++) {
          std::fill(
              host.begin() + static_cast<ptrdiff_t>(owner * count),
              host.begin() + static_cast<ptrdiff_t>((owner + 1) * count),
              blockValue(b.myActiveIndex, owner));
        }
        HIPCHECK_TEST(hipMemcpy(
            b.sendMem[g],
            host.data(),
            inputCount * sizeof(float),
            hipMemcpyHostToDevice));
      } else {
        HIPCHECK_TEST(hipMemset(b.sendMem[g], 0, inputCount * sizeof(float)));
      }
      b.sendPtrs[g] = b.sendMem[g];
      b.counts[g] = count;
    }

    HIPCHECK_TEST(hipMalloc(&b.recvMem, count * sizeof(float)));
    HIPCHECK_TEST(hipMemset(b.recvMem, 0, count * sizeof(float)));
    for (int g = 0; g < kGroups; g++) {
      b.recvPtrs[g] = (g == b.myActiveGroup) ? b.recvMem : b.sendMem[g];
    }
  }

  void freeBuffers(Buffers& b) {
    for (int g = 0; g < kGroups; g++) {
      HIPCHECK_TEST(hipFree(b.sendMem[g]));
    }
    HIPCHECK_TEST(hipFree(b.recvMem));
  }

  void expectOutputEquals(const Buffers& b, size_t count, float wantValue) {
    std::vector<float> got(count);
    HIPCHECK_TEST(hipMemcpy(
        got.data(), b.recvMem, count * sizeof(float), hipMemcpyDeviceToHost));
    size_t reported = 0;
    for (size_t i = 0; i < count && reported < 8; i++) {
      if (got[i] != wantValue) {
        reported++;
        ADD_FAILURE() << "R" << this->globalRank << ": element " << i
                      << ": got " << got[i] << ", want " << wantValue;
      }
    }
  }

  ncclResult_t call(const Buffers& b, ncclRedOp_t op, int lowPrecision) {
    TwoGroupFourActiveRanks layout;
    return callReduceScatterCompat(
        b.sendPtrs,
        b.recvPtrs,
        b.counts,
        ncclFloat32,
        op,
        this->commFor(kActive),
        this->stream,
        layout.allActiveRanks,
        kActive,
        kGroups,
        lowPrecision);
  }

  // The flat schedule only routes through the helpers above the offload
  // crossover; below it the call degenerates to a pure-direct all-to-all with
  // the helpers idle, which low precision declines. Asserting the route keeps a
  // resized test from silently covering the wrong schedule.
  void assertOffloadRouteSelected(size_t count) {
    size_t counts[kGroups];
    for (int g = 0; g < kGroups; g++) {
      counts[g] = count;
    }
    ASSERT_EQ(
        rcclx::relay::selectReduceScatterRoute(
            kActive, 8 - kActive, kGroups, counts, sizeof(float)),
        rcclx::relay::ReduceScatterRoute::FlatOffload);
  }
};

TEST_F(
    ShardedRelayReduceScatterFlatLowPrecisionTest,
    ConstantBlocksAreBitExact) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  assertOffloadRouteSelected(kLpCount);

  Buffers b;
  makeBuffers(kLpCount, b);
  barrierSyncOn(nullptr);

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(call(b, ncclSum, /*lowPrecision=*/1), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  // Covers both regions of the schedule at once: the direct all-to-all over the
  // intra links and the reduce-at-helper offload over the cross links. Their
  // boundary is interior to the output block, so a wire offset that disagrees
  // with a peer's shows up as a mismatch partway through rather than
  // everywhere.
  expectOutputEquals(b, kLpCount, expectedOwnerValue(b.myActiveIndex));
  EXPECT_GT(rcclx::relay::lpEngageCount(), 0u)
      << "low precision never engaged, so this case proved nothing";
  EXPECT_EQ(rcclx::relay::lpDeclineCount(), 0u);
  freeBuffers(b);
}

TEST_F(
    ShardedRelayReduceScatterFlatLowPrecisionTest,
    AvgAppliesTheDivisorOnce) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  // The divisor placement differs BETWEEN THE TWO REGIONS of this one schedule.
  // The offload region's helper reduces and returns a plain sum, so the divisor
  // waits for the owner's fold; the direct region has no helper at all and gets
  // it in the same owner-side reduce. A divisor applied at the helper as well
  // would scale only the offload region, which the exact comparator catches at
  // the region boundary rather than everywhere.
  assertOffloadRouteSelected(kLpCount);

  Buffers b;
  makeBuffers(kLpCount, b);
  barrierSyncOn(nullptr);

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(call(b, ncclAvg, /*lowPrecision=*/1), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  expectOutputEquals(
      b,
      kLpCount,
      expectedOwnerValue(b.myActiveIndex) / static_cast<float>(kActive));
  EXPECT_GT(rcclx::relay::lpEngageCount(), 0u);
  freeBuffers(b);
}

TEST_F(
    ShardedRelayReduceScatterFlatLowPrecisionTest,
    InterleavesWithFullPrecision) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  assertOffloadRouteSelected(kLpCount);

  Buffers b;
  makeBuffers(kLpCount, b);
  barrierSyncOn(nullptr);

  rcclx::relay::lpResetCounters();
  ASSERT_EQ(call(b, ncclSum, /*lowPrecision=*/1), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));
  expectOutputEquals(b, kLpCount, expectedOwnerValue(b.myActiveIndex));
  ASSERT_GT(rcclx::relay::lpEngageCount(), 0u);

  const uint64_t engagedBefore = rcclx::relay::lpEngageCount();
  HIPCHECK_TEST(hipMemset(b.recvMem, 0, kLpCount * sizeof(float)));
  barrierSyncOn(nullptr);
  ASSERT_EQ(call(b, ncclSum, /*lowPrecision=*/0), ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));
  expectOutputEquals(b, kLpCount, expectedOwnerValue(b.myActiveIndex));
  EXPECT_EQ(rcclx::relay::lpEngageCount(), engagedBefore)
      << "a full-precision call must not engage low precision";
  freeBuffers(b);
}

TEST_F(
    ShardedRelayReduceScatterFlatLowPrecisionTest,
    SingleGroupPipelinedIsBitExact) {
  if (this->numRanks != 8) {
    GTEST_SKIP() << "Test requires exactly 8 ranks";
  }
  // relayPipelineTiles() returns 1 whenever nGroups != 1, so the two-group
  // cases above cannot reach the pipelined flat schedule. This one is
  // single-group, at a size the depth selector pipelines -- asserted, so a
  // resize cannot quietly turn this into a third test of the non-pipelined
  // path.
  const size_t count = 8ULL * 1024 * 1024;
  size_t counts[1] = {count};
  ASSERT_EQ(
      rcclx::relay::selectReduceScatterRoute(
          kActive, 8 - kActive, 1, counts, sizeof(float)),
      rcclx::relay::ReduceScatterRoute::FlatOffload);
  ASSERT_GT(
      rcclx::relay::relayPipelineTiles(
          1,
          rcclx::relay::relayShapeFanout(kActive, 8 - kActive),
          count,
          sizeof(float)),
      1);

  // Group 0's active ranks are {0, 1, 2, 3}; ranks 4-7 are its helpers.
  const int myActiveIndex = this->globalRank % kActive;
  const bool owner = this->globalRank < kActive;
  const size_t inputCount = static_cast<size_t>(kActive) * count;

  void* sendMem = nullptr;
  void* recvMem = nullptr;
  HIPCHECK_TEST(hipMalloc(&sendMem, inputCount * sizeof(float)));
  HIPCHECK_TEST(hipMalloc(&recvMem, count * sizeof(float)));
  HIPCHECK_TEST(hipMemset(recvMem, 0, count * sizeof(float)));
  if (owner) {
    std::vector<float> host(inputCount);
    for (int j = 0; j < kActive; j++) {
      std::fill(
          host.begin() + static_cast<ptrdiff_t>(j * count),
          host.begin() + static_cast<ptrdiff_t>((j + 1) * count),
          blockValue(myActiveIndex, j));
    }
    HIPCHECK_TEST(hipMemcpy(
        sendMem,
        host.data(),
        inputCount * sizeof(float),
        hipMemcpyHostToDevice));
  } else {
    HIPCHECK_TEST(hipMemset(sendMem, 0, inputCount * sizeof(float)));
  }

  const void* sendPtrs[1] = {sendMem};
  void* recvPtrs[1] = {owner ? recvMem : sendMem};
  barrierSyncOn(nullptr);

  TwoGroupFourActiveRanks layout;
  rcclx::relay::lpResetCounters();
  ASSERT_EQ(
      callReduceScatterCompat(
          sendPtrs,
          recvPtrs,
          counts,
          ncclFloat32,
          ncclSum,
          this->commFor(kActive),
          this->stream,
          layout.allActiveRanks,
          kActive,
          1,
          /*lowPrecision=*/1),
      ncclSuccess);
  HIPCHECK_TEST(hipStreamSynchronize(this->stream));

  if (owner) {
    const float want = expectedOwnerValue(myActiveIndex);
    std::vector<float> got(count);
    HIPCHECK_TEST(hipMemcpy(
        got.data(), recvMem, count * sizeof(float), hipMemcpyDeviceToHost));
    size_t reported = 0;
    for (size_t i = 0; i < count && reported < 8; i++) {
      if (got[i] != want) {
        reported++;
        ADD_FAILURE() << "R" << this->globalRank << ": element " << i
                      << ": got " << got[i] << ", want " << want;
      }
    }
  }
  EXPECT_GT(rcclx::relay::lpEngageCount(), 0u)
      << "low precision never engaged, so this case proved nothing";
  EXPECT_EQ(rcclx::relay::lpDeclineCount(), 0u);

  HIPCHECK_TEST(hipFree(sendMem));
  HIPCHECK_TEST(hipFree(recvMem));
}

int main(int argc, char* argv[]) {
  // The low-precision size thresholds that SHIP are a tuning policy measured
  // per shape, and they decline most shapes outright (see lpMinBytes()). This
  // suite covers the MECHANISM -- that the wire format is correct wherever it
  // runs -- so it must not be coupled to that policy: a retune would otherwise
  // silently turn these cases into no-ops that still pass. Set before any
  // communicator exists, because NCCL_PARAM caches on first read.
  setenv("NCCL_SHARDED_RELAY_LP_MIN_KB", "1", /*overwrite=*/1);
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
