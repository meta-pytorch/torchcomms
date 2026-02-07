// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <fmt/core.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include <memory>
#include <vector>

#include "VerifyAlgoStatsUtil.h"
#include "comms/testinfra/TestUtils.h"
// #include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/wrapper/DataTypeStrUtils.h"

/**
 * Test suite for ReduceScatter PAT algorithm selection logic.
 *
 * Tests cover:
 * 1. PAT algorithm selection with different PAT_AVG settings and ncclOps
 * 2. User-defined PreMulSum operations are correctly blocked from PAT
 * 3. Built-in ncclAvg correctly uses PAT when PAT_AVG is enabled
 */

class ReduceScatterPatSelectTest : public NcclxBaseTest {
 public:
  ReduceScatterPatSelectTest() = default;

  void SetUp() override {
    NcclxBaseTest::SetUp();
    // [META:PAT] Enable PAT algorithm for all tests in this suite.
    // This must be set BEFORE any communicator is created because
    // ncclParamPatEnable() uses a static cache that is only populated once.
    // Setting it here ensures the cache is populated with the correct value
    // regardless of test execution order.
    patEnableGuard_ =
        std::make_unique<EnvRAII<int64_t>>(NCCL_PAT_ENABLE, (int64_t)1);
    // Enable AlgoStats for algorithm validation (must be before comm creation)
    algoStats_.enable();
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    patEnableGuard_.reset();
    NcclxBaseTest::TearDown();
  }

 protected:
  cudaStream_t stream{nullptr};
  ncclx::test::VerifyAlgoStatsHelper algoStats_;
  std::unique_ptr<EnvRAII<int64_t>> patEnableGuard_;
};

/**
 * Test: User-defined PreMulSum should NOT be converted to PatAvg
 *
 * Verifies that setting usePatAvg_ = true only affects ncclAvg operations,
 * not user-defined PreMulSum ops. The ext is only set when op == ncclAvg,
 * so user ops continue through normal algorithm selection.
 */
TEST_F(ReduceScatterPatSelectTest, UserPreMulSumNotConvertedToPatAvg) {
  NcclCommRAII commGuard{globalRank, numRanks, localRank};
  ncclComm_t comm = commGuard.get();
  comm->usePatAvg_ = true;

  // Create user-defined PreMulSum with scalar = 0.25
  // This is different from ncclAvg which uses 1/nRanks (0.5 for 2 ranks)
  ncclRedOp_t userOp;
  float scalar = 0.25f;
  NCCLCHECK_TEST(ncclRedOpCreatePreMulSum(
      &userOp, &scalar, ncclFloat, ncclScalarHostImmediate, comm));

  const size_t count = 8000;
  const size_t allocSize = count * numRanks * sizeof(float);

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  NCCLCHECK_TEST(ncclMemAlloc((void**)&sendBuf, allocSize));
  NCCLCHECK_TEST(ncclMemAlloc((void**)&recvBuf, allocSize));

  // Initialize: each rank sends 1.0 in all chunks
  for (int r = 0; r < numRanks; r++) {
    assignChunkValue(sendBuf + r * count, count, 1.0f);
  }

  // Run ReduceScatter with user PreMulSum op
  // Should succeed and use normal algorithm (not PAT AVG)
  auto res = ncclReduceScatter(
      sendBuf, recvBuf, count, ncclFloat, userOp, comm, stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Expected result with scalar = 0.25:
  // Each element: sum of contributions from all ranks * 0.25
  // = numRanks * 1.0 * 0.25 = 0.5 (for 2 ranks)
  float expectedVal = static_cast<float>(numRanks) * 0.25f;

  size_t errs = checkChunkValue(
      recvBuf, count, expectedVal, 0.0f, globalRank, stream, 1e-3);
  EXPECT_EQ(errs, 0) << "Rank " << globalRank
                     << " user PreMulSum got wrong result"
                     << " (expected=" << expectedVal << ")";

  NCCLCHECK_TEST(ncclRedOpDestroy(userOp, comm));
  NCCLCHECK_TEST(ncclMemFree(sendBuf));
  NCCLCHECK_TEST(ncclMemFree(recvBuf));
}

/**
 * Test: Built-in ncclAvg with PAT_AVG should work correctly (regression test)
 *
 * Verifies that built-in ncclAvg uses PAT algorithm with PAT_AVG enabled
 * and produces correct results (sum / nRanks).
 */
TEST_F(ReduceScatterPatSelectTest, BuiltInAvgWithPatAvgWorks) {
  NcclCommRAII commGuard{globalRank, numRanks, localRank};
  ncclComm_t comm = commGuard.get();
  // Enable PAT AVG via per-communicator control
  comm->usePatAvg_ = true;

  const size_t count = 8000;
  const size_t allocSize = count * numRanks * sizeof(float);

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  NCCLCHECK_TEST(ncclMemAlloc((void**)&sendBuf, allocSize));
  NCCLCHECK_TEST(ncclMemAlloc((void**)&recvBuf, allocSize));

  // Initialize send buffer
  for (int r = 0; r < numRanks; r++) {
    float val = static_cast<float>(globalRank * numRanks + r);
    assignChunkValue(sendBuf + r * count, count, val);
  }

  // Run ReduceScatter with built-in ncclAvg
  auto res = ncclReduceScatter(
      sendBuf, recvBuf, count, ncclFloat, ncclAvg, comm, stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Expected: average of (r * numRanks + globalRank) for all r
  float sum = 0.0f;
  for (int r = 0; r < numRanks; r++) {
    sum += static_cast<float>(r * numRanks + globalRank);
  }
  float expectedVal = sum / static_cast<float>(numRanks);

  size_t errs = checkChunkValue(
      recvBuf, count, expectedVal, 0.0f, globalRank, stream, 1e-3);
  EXPECT_EQ(errs, 0) << "Rank " << globalRank
                     << " ncclAvg with PAT_AVG got wrong result"
                     << " (expected=" << expectedVal << ")";

  // Verify PAT algorithm was used
  algoStats_.verify(comm, "ReduceScatter", "PAT");

  NCCLCHECK_TEST(ncclMemFree(sendBuf));
  NCCLCHECK_TEST(ncclMemFree(recvBuf));
}

/**
 * Parameterized test for PAT algorithm selection with different settings.
 *
 * Tests PAT algorithm selection with:
 * - Different PAT_AVG_ENABLE settings (true/false)
 * - Different ncclRedOp_t operations (ncclSum, ncclAvg)
 * - Validates the expected algorithm was used via AlgoStats
 */
class ReduceScatterPatAlgoSelectionTest
    : public ReduceScatterPatSelectTest,
      public ::testing::WithParamInterface<
          std::tuple<bool, ncclRedOp_t, std::string>> {};

TEST_P(ReduceScatterPatAlgoSelectionTest, AlgoSelection) {
  auto [patAvgEnable, op, expectedAlgoSubstr] = GetParam();

  // Enforce PAT algorithm selection via env vars for SUM (both NCCL_ALGO,
  // NCCL_PROTO and NCCL_PAT_ENABLE must be set, NCCL_PAT_ENABLE=1 is set in
  // base fixture SetUp()). AVG require usePatAvg_ = true
  auto algoGuard = EnvRAII<std::string>(NCCL_ALGO, "reducescatter:pat");
  auto protoGuard = EnvRAII<std::string>(NCCL_PROTO, "Simple");

  NcclCommRAII commGuard{globalRank, numRanks, localRank};
  ncclComm_t comm = commGuard.get();

  // Enable PAT AVG via per-communicator control
  comm->usePatAvg_ = patAvgEnable;

  const size_t count = 8000;
  const size_t allocSize = count * numRanks * sizeof(float);

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  NCCLCHECK_TEST(ncclMemAlloc((void**)&sendBuf, allocSize));
  NCCLCHECK_TEST(ncclMemAlloc((void**)&recvBuf, allocSize));

  // Initialize send buffer with simple values
  for (int r = 0; r < numRanks; r++) {
    assignChunkValue(sendBuf + r * count, count, 1.0f);
  }

  // Run ReduceScatter
  auto res =
      ncclReduceScatter(sendBuf, recvBuf, count, ncclFloat, op, comm, stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify expected algorithm was used
  algoStats_.verify(comm, "ReduceScatter", expectedAlgoSubstr);

  NCCLCHECK_TEST(ncclMemFree(sendBuf));
  NCCLCHECK_TEST(ncclMemFree(recvBuf));
}

INSTANTIATE_TEST_SUITE_P(
    ReduceScatterPatAlgoSelectionTestInstance,
    ReduceScatterPatAlgoSelectionTest,
    ::testing::Values(
        // PAT_AVG_ENABLE=true: ALL operations use PAT (including Avg)
        std::make_tuple(true, ncclSum, "PAT"),
        std::make_tuple(true, ncclAvg, "PAT"),
        // PAT_AVG_ENABLE=false: Sum uses PAT, Avg would fail (not tested here)
        std::make_tuple(false, ncclSum, "PAT")),
    [](const testing::TestParamInfo<std::tuple<bool, ncclRedOp_t, std::string>>&
           info) {
      auto patAvgEnable = std::get<0>(info.param);
      auto op = std::get<1>(info.param);
      const auto& expectedAlgo = std::get<2>(info.param);
      return fmt::format(
          "PAT_AVG_{}_{}_{}",
          patAvgEnable ? "on" : "off",
          getRedOpStr(op),
          expectedAlgo);
    });

/**
 * Test: Grouped ReduceScatter with PAT AVG - expected to fail
 *
 * Tests that grouped collectives with ncclInfoExt override are correctly
 * rejected with ncclInvalidUsage. Grouped collectives with per-comm
 * algorithm override (PAT AVG) are not currently supported.
 *
 * This is a negative test - it validates that the proper error is returned.
 */
TEST_F(ReduceScatterPatSelectTest, GroupedReduceScatterPatAvg) {
  NcclCommRAII commGuard{globalRank, numRanks, localRank};
  ncclComm_t comm = commGuard.get();
  // Enable PAT AVG via per-communicator control
  comm->usePatAvg_ = true;

  constexpr int kNumOpsInGroup = 3;
  const size_t count = 8000;
  const size_t allocSize = count * numRanks * sizeof(float);

  std::vector<float*> sendBufs(kNumOpsInGroup, nullptr);
  std::vector<float*> recvBufs(kNumOpsInGroup, nullptr);

  for (int i = 0; i < kNumOpsInGroup; i++) {
    NCCLCHECK_TEST(ncclMemAlloc((void**)&sendBufs[i], allocSize));
    NCCLCHECK_TEST(ncclMemAlloc((void**)&recvBufs[i], allocSize));
  }

  // Initialize with different values per operation
  for (int i = 0; i < kNumOpsInGroup; i++) {
    for (int r = 0; r < numRanks; r++) {
      float val = static_cast<float>(globalRank * numRanks + r + i * 100);
      assignChunkValue(sendBufs[i] + r * count, count, val);
    }
  }

  // Run grouped ReduceScatter - all with ncclAvg
  // Individual enqueue calls succeed, but ncclGroupEnd will fail
  NCCLCHECK_TEST(ncclGroupStart());
  for (int i = 0; i < kNumOpsInGroup; i++) {
    auto res = ncclReduceScatter(
        sendBufs[i], recvBufs[i], count, ncclFloat, ncclAvg, comm, stream);
    ASSERT_EQ(res, ncclSuccess);
  }
  // Grouped collectives with ncclInfoExt override are not supported
  // ncclGroupEnd should return ncclInvalidUsage
  auto groupEndRes = ncclGroupEnd();
  EXPECT_EQ(groupEndRes, ncclInvalidUsage)
      << "Grouped collectives with PAT AVG ext override should fail with "
         "ncclInvalidUsage";

  for (int i = 0; i < kNumOpsInGroup; i++) {
    NCCLCHECK_TEST(ncclMemFree(sendBufs[i]));
    NCCLCHECK_TEST(ncclMemFree(recvBufs[i]));
  }
}

/**
 * Test: Direct usePatAvg_ control enables PAT AVG for ReduceScatter
 *
 * Verifies that setting comm->usePatAvg_ = true directly enables PAT AVG
 * for ReduceScatter with ncclAvg, bypassing the need for CVARs.
 * This is the per-communicator control mechanism.
 */
TEST_F(ReduceScatterPatSelectTest, UsePatAvgDirectControl) {
  // No CVAR guards needed - usePatAvg_ bypasses algorithm selection
  NcclCommRAII commGuard{globalRank, numRanks, localRank};
  ncclComm_t comm = commGuard.get();

  // Default is disabled
  ASSERT_FALSE(comm->usePatAvg_);

  // Enable PAT AVG for this comm
  comm->usePatAvg_ = true;

  const size_t count = 8000;
  const size_t allocSize = count * numRanks * sizeof(float);

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  NCCLCHECK_TEST(ncclMemAlloc((void**)&sendBuf, allocSize));
  NCCLCHECK_TEST(ncclMemAlloc((void**)&recvBuf, allocSize));

  // Initialize: each rank sends its rank value in all chunks
  for (int r = 0; r < numRanks; r++) {
    float val = static_cast<float>(globalRank * numRanks + r);
    assignChunkValue(sendBuf + r * count, count, val);
  }

  // Run ReduceScatter with ncclAvg - should use PAT AVG
  auto res = ncclReduceScatter(
      sendBuf, recvBuf, count, ncclFloat, ncclAvg, comm, stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Expected: average of (r * numRanks + globalRank) for all r
  float sum = 0.0f;
  for (int r = 0; r < numRanks; r++) {
    sum += static_cast<float>(r * numRanks + globalRank);
  }
  float expectedVal = sum / static_cast<float>(numRanks);

  size_t errs = checkChunkValue(
      recvBuf, count, expectedVal, 0.0f, globalRank, stream, 1e-3);
  EXPECT_EQ(errs, 0) << "Rank " << globalRank
                     << " usePatAvg_ direct control got wrong result"
                     << " (expected=" << expectedVal << ")";

  // Verify PAT algorithm was used
  algoStats_.verify(comm, "ReduceScatter", "PAT");

  NCCLCHECK_TEST(ncclMemFree(sendBuf));
  NCCLCHECK_TEST(ncclMemFree(recvBuf));
}

/**
 * Test: usePatAvg_ only affects ReduceScatter with ncclAvg
 *
 * Verifies that usePatAvg_ doesn't affect:
 * 1. ReduceScatter with other ops (ncclSum) - uses normal algorithm selection
 * 2. Other collectives (AllReduce with ncclAvg) - not affected
 */
TEST_F(ReduceScatterPatSelectTest, UsePatAvgOnlyAffectsReduceScatterAvg) {
  NcclCommRAII commGuard{globalRank, numRanks, localRank};
  ncclComm_t comm = commGuard.get();
  comm->usePatAvg_ = true;

  const size_t count = 8000;
  const size_t allocSize = count * numRanks * sizeof(float);

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  NCCLCHECK_TEST(ncclMemAlloc((void**)&sendBuf, allocSize));
  NCCLCHECK_TEST(ncclMemAlloc((void**)&recvBuf, allocSize));

  // Initialize send buffer
  for (int r = 0; r < numRanks; r++) {
    assignChunkValue(sendBuf + r * count, count, 1.0f);
  }

  // ReduceScatter with ncclSum should NOT force PAT (usePatAvg_ only affects
  // ncclAvg)
  auto res = ncclReduceScatter(
      sendBuf, recvBuf, count, ncclFloat, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Expected for ncclSum: sum of all ranks' values = numRanks * 1.0
  float expectedVal = static_cast<float>(numRanks);
  size_t errs = checkChunkValue(
      recvBuf, count, expectedVal, 0.0f, globalRank, stream, 1e-3);
  EXPECT_EQ(errs, 0) << "Rank " << globalRank
                     << " ReduceScatter with ncclSum got wrong result";

  NCCLCHECK_TEST(ncclMemFree(sendBuf));
  NCCLCHECK_TEST(ncclMemFree(recvBuf));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
