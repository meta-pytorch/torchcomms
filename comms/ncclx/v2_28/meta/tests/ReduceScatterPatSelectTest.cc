// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <cuda_fp16.h>
#include <fmt/core.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "VerifyAlgoStatsUtil.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/hints/GlobalHints.h" // @manual
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
    // Reset global hint to avoid affecting subsequent tests
    ncclx::resetGlobalHint(
        std::string(ncclx::HintKeys::kCommAlgoReduceScatter));
    NcclxBaseTest::TearDown();
  }

 protected:
  // Common run() helper that encapsulates the typical ReduceScatter
  // test flow: create comm -> set usePatAvg_ -> alloc buffers ->
  // init send buffer -> run ncclReduceScatter -> sync -> check result
  // -> verify algo -> free buffers.
  //
  // Template parameter T: C++ data type (float, __half, int32_t, etc.)
  //
  // sendValFn(numRanks, globalRank, chunkIdx) returns the value to
  // fill into each chunk of the send buffer.
  // expectedValFn(numRanks, globalRank) returns the expected result;
  // pass nullptr to skip result checking.
  template <typename T>
  void run(
      ncclDataType_t dtype,
      ncclRedOp_t op,
      bool usePatAvg,
      const std::function<
          T(int /*numRanks*/, int /*globalRank*/, int /*chunkIdx*/)>& sendValFn,
      const std::function<T(int /*numRanks*/, int /*globalRank*/)>&
          expectedValFn,
      std::optional<std::string> expectedAlgo = std::nullopt,
      std::optional<std::string> unexpectedAlgo = std::nullopt,
      std::optional<double> tolerance = std::nullopt) {
    NcclCommRAII commGuard{globalRank, numRanks, localRank};
    ncclComm_t comm = commGuard.get();
    comm->usePatAvg_ = usePatAvg;

    const size_t count = 8000;
    const size_t allocSize = count * numRanks * sizeof(T);

    T* sendBuf = nullptr;
    T* recvBuf = nullptr;
    NCCLCHECK_TEST(ncclMemAlloc((void**)&sendBuf, allocSize));
    NCCLCHECK_TEST(ncclMemAlloc((void**)&recvBuf, allocSize));

    for (int r = 0; r < numRanks; r++) {
      assignChunkValue(
          sendBuf + r * count, count, sendValFn(numRanks, globalRank, r));
    }

    auto res =
        ncclReduceScatter(sendBuf, recvBuf, count, dtype, op, comm, stream);
    ASSERT_EQ(res, ncclSuccess);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    if (expectedValFn) {
      T expected = expectedValFn(numRanks, globalRank);
      T zero{0};
      size_t errs = checkChunkValue(
          recvBuf, count, expected, zero, globalRank, stream, tolerance);
      EXPECT_EQ(errs, 0) << "Rank " << globalRank
                         << " got wrong result (expected="
                         << static_cast<double>(expected) << ")";
    }

    if (expectedAlgo.has_value()) {
      algoStats_.verify(comm, "ReduceScatter", expectedAlgo.value());
    }
    if (unexpectedAlgo.has_value()) {
      algoStats_.verifyNot(comm, "ReduceScatter", unexpectedAlgo.value());
    }

    NCCLCHECK_TEST(ncclMemFree(sendBuf));
    NCCLCHECK_TEST(ncclMemFree(recvBuf));
  }

  cudaStream_t stream{nullptr};
  ncclx::test::VerifyAlgoStatsHelper algoStats_;
  std::unique_ptr<EnvRAII<int64_t>> patEnableGuard_;
};

/**
 * Test: User-defined PreMulSum should NOT be converted to PatAvg
 *
 * Verifies that enabling PAT AVG only affects ncclAvg operations,
 * not user-defined PreMulSum ops. The ext is only set when op == ncclAvg,
 * so user ops continue through normal algorithm selection.
 */
TEST_F(ReduceScatterPatSelectTest, UserPreMulSumNotConvertedToPatAvg) {
  // Enable PAT AVG via global hint before comm creation
  ASSERT_EQ(
      ncclx::setGlobalHint(
          std::string(ncclx::HintKeys::kCommAlgoReduceScatter), "avg:patavg"),
      ncclSuccess);

  NcclCommRAII commGuard{globalRank, numRanks, localRank};
  ncclComm_t comm = commGuard.get();
  ASSERT_TRUE(comm->usePatAvg_);

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
  // Enable PAT AVG via global hint before comm creation
  ASSERT_EQ(
      ncclx::setGlobalHint(
          std::string(ncclx::HintKeys::kCommAlgoReduceScatter), "avg:patavg"),
      ncclSuccess);

  NcclCommRAII commGuard{globalRank, numRanks, localRank};
  ncclComm_t comm = commGuard.get();
  ASSERT_TRUE(comm->usePatAvg_);

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
  // base fixture SetUp()). AVG requires PAT AVG CVAR or hint.
  auto algoGuard = EnvRAII<std::string>(NCCL_ALGO, "reducescatter:pat");
  auto protoGuard = EnvRAII<std::string>(NCCL_PROTO, "Simple");

  // Enable PAT AVG via CVAR before comm creation
  auto patAvgGuard = EnvRAII(NCCL_REDUCESCATTER_PAT_AVG_ENABLE, patAvgEnable);

  NcclCommRAII commGuard{globalRank, numRanks, localRank};
  ncclComm_t comm = commGuard.get();
  ASSERT_EQ(comm->usePatAvg_, patAvgEnable);

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
  // Enable PAT AVG via global hint before comm creation
  ASSERT_EQ(
      ncclx::setGlobalHint(
          std::string(ncclx::HintKeys::kCommAlgoReduceScatter), "avg:patavg"),
      ncclSuccess);

  NcclCommRAII commGuard{globalRank, numRanks, localRank};
  ncclComm_t comm = commGuard.get();
  ASSERT_TRUE(comm->usePatAvg_);

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
 * Test: CVAR control enables PAT AVG for ReduceScatter
 *
 * Verifies that setting NCCL_REDUCESCATTER_PAT_AVG_ENABLE=true enables PAT AVG
 * for ReduceScatter with ncclAvg at comm creation time.
 */
TEST_F(ReduceScatterPatSelectTest, UsePatAvgCvarControl) {
  // Enable PAT AVG via CVAR before comm creation
  auto patAvgGuard = EnvRAII(NCCL_REDUCESCATTER_PAT_AVG_ENABLE, true);

  NcclCommRAII commGuard{globalRank, numRanks, localRank};
  ncclComm_t comm = commGuard.get();

  // Verify CVAR enabled usePatAvg_
  ASSERT_TRUE(comm->usePatAvg_);

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
                     << " CVAR control got wrong result"
                     << " (expected=" << expectedVal << ")";

  // Verify PAT algorithm was used
  algoStats_.verify(comm, "ReduceScatter", "PAT");

  NCCLCHECK_TEST(ncclMemFree(sendBuf));
  NCCLCHECK_TEST(ncclMemFree(recvBuf));
}

/**
 * Test: PAT AVG only affects ReduceScatter with ncclAvg
 *
 * Verifies that PAT AVG doesn't affect:
 * 1. ReduceScatter with other ops (ncclSum) - uses normal algorithm selection
 * 2. Other collectives (AllReduce with ncclAvg) - not affected
 */
TEST_F(ReduceScatterPatSelectTest, UsePatAvgOnlyAffectsReduceScatterAvg) {
  // Enable PAT AVG via global hint before comm creation
  ASSERT_EQ(
      ncclx::setGlobalHint(
          std::string(ncclx::HintKeys::kCommAlgoReduceScatter), "avg:patavg"),
      ncclSuccess);

  NcclCommRAII commGuard{globalRank, numRanks, localRank};
  ncclComm_t comm = commGuard.get();
  ASSERT_TRUE(comm->usePatAvg_);

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

  // ReduceScatter with ncclSum should NOT force PAT (PAT AVG only affects
  // ncclAvg)
  auto res = ncclReduceScatter(
      sendBuf, recvBuf, count, ncclFloat, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Expected: sum of 1.0 from each rank = numRanks
  float expectedVal = static_cast<float>(numRanks);
  size_t errs = checkChunkValue(
      recvBuf, count, expectedVal, 0.0f, globalRank, stream, 1e-3);
  EXPECT_EQ(errs, 0) << "Rank " << globalRank
                     << " ncclSum with PAT_AVG got wrong result"
                     << " (expected=" << expectedVal << ")";

  NCCLCHECK_TEST(ncclMemFree(sendBuf));
  NCCLCHECK_TEST(ncclMemFree(recvBuf));
}

/**
 * Test: Unsupported dtype (fp16) falls back from PAT to normal algorithm
 *
 * When usePatAvg_ is true but dtype is fp16, isPatAvgSupportedType() returns
 * false, so the PAT AVG override is not applied and normal algorithm selection
 * is used instead. Verifies:
 * 1. The operation succeeds (no crash)
 * 2. The result is correct (standard ncclAvg via PreMulSum path)
 * 3. The algorithm used is NOT PAT
 */
TEST_F(ReduceScatterPatSelectTest, UnsupportedDtypeFallsBackFromPat) {
  using Traits = DataTypeTraits<__half>;
  constexpr bool kUsePatAvg = true;
  const std::string kUnexpectAlgo = "PAT";
  run<__half>(
      ncclFloat16,
      ncclAvg,
      kUsePatAvg,
      [](int nRanks, int rank, int chunk) -> __half {
        using T = DataTypeTraits<__half>;
        float val = static_cast<float>(rank * nRanks + chunk);
        return T::toDevice(val);
      },
      [](int nRanks, int rank) -> __half {
        using T = DataTypeTraits<__half>;
        float sum = 0.0f;
        for (int r = 0; r < nRanks; r++) {
          sum += static_cast<float>(r * nRanks + rank);
        }
        return T::toDevice(sum / static_cast<float>(nRanks));
      },
      std::nullopt,
      kUnexpectAlgo,
      Traits::tolerance());
}

/**
 * Test: Signed integer (ncclInt32) average with PAT AVG works correctly
 *
 * Verifies that FuncPatSumPostDiv correctly handles signed integer division
 * via the isSigned flag. Each rank sends negative values, and the post-division
 * must reinterpret the accumulated unsigned sum as signed before dividing.
 *
 * For N ranks: each rank r sends -(r+1) in every element.
 * Expected average = sum(-(r+1) for r in 0..N-1) / N
 *   e.g. 2 ranks: (-1 + -2) / 2 = -3/2 = -1 (truncation toward zero)
 *   e.g. 8 ranks: (-1 + -2 + ... + -8) / 8 = -36/8 = -4 (exact)
 */
TEST_F(ReduceScatterPatSelectTest, SignedIntAvgWithPatSumPostDiv) {
  constexpr bool kUsePatAvg = true;
  const std::string kExpectAlgo = "PAT";
  run<int32_t>(
      ncclInt32,
      ncclAvg,
      kUsePatAvg,
      [](int /*nRanks*/, int rank, int /*chunk*/) -> int32_t {
        return -(static_cast<int32_t>(rank) + 1);
      },
      [](int nRanks, int /*rank*/) -> int32_t {
        int32_t sum = 0;
        for (int r = 0; r < nRanks; r++) {
          sum += -(static_cast<int32_t>(r) + 1);
        }
        return sum / static_cast<int32_t>(nRanks);
      },
      kExpectAlgo);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
