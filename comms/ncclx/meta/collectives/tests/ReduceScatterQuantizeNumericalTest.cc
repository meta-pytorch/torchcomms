// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Numerical correctness tests for ncclReduceScatterQuantize.
// These tests verify properties of stochastic rounding that distinguish
// RSQ from native BF16 reduce-scatter: variance reduction via averaging
// and unbiasedness. Each test explicitly runs both RSQ and BF16 RS and
// asserts that BF16 RS cannot pass the same checks.
//
// Basic functionality and API validation tests are in
// ReduceScatterQuantizeTest.cc. Numerical benchmarks are in
// benchmarks/ReduceScatterQuantizeNumericalBench.cc.

#include <comm.h>
#include <cuda_bf16.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <cmath>
#include <cstddef>
#include <vector>
#include "comms/ncclx/meta/collectives/tests/ReduceScatterQuantizeTestUtils.h"

// Generate a deterministic seed from trial index and rank.
// Uses coprime multipliers to avoid correlation between trials and ranks.
static uint64_t trialSeed(int trial, int rank) {
  return static_cast<uint64_t>(trial) * 9973 +
      static_cast<uint64_t>(rank) * 31;
}

// Compute a test value for a given (index, rank, chunk) triple.
// base must not be exactly BF16-representable so that every element
// exercises stochastic rounding. Perturbation scales are fractions of the
// BF16 ULP at the base magnitude, ensuring values span multiple BF16
// intervals across elements.
static float
varianceTestValue(float base, float ulp, size_t i, int rank, int chunk) {
  return base + static_cast<float>(i) * (ulp / 16.0f) +
      static_cast<float>(rank) * ulp +
      static_cast<float>(chunk) * (ulp * 2.0f);
}

// ---------------------------------------------------------------------------
// MultiTrialVarianceReduction: prove that averaging multiple RSQ runs
// (with different seeds) reduces error, while BF16 RS (deterministic) cannot
// benefit from averaging.
//
// This test exploits the fundamental property that stochastic rounding errors
// are independent across seeds. With K trials, the MAE should shrink by
// ~1/sqrt(K). If RSQ is replaced by native BF16 RS, all trials produce
// identical output, so averaging changes nothing and the assertion fails.
// ---------------------------------------------------------------------------
TEST_F(ReduceScatterQuantizeTest, MultiTrialVarianceReduction) {
  const size_t count = 8192;
  const int numTrials = 16;

  // Base value between BF16 representable values 1.328125 and 1.3359375,
  // chosen so every element exercises stochastic rounding.
  const float kBase = 1.33f;
  const float kBaseUlp = bf16Ulp(kBase);

  // With 16 trials, theoretical MAE reduction is 1/sqrt(16) = 0.25.
  // Threshold 0.6 gives 2.4x safety margin against statistical variance.
  const double kVarianceReductionThreshold = 0.6;

  const size_t sendSize = count * numRanks * sizeof(float);
  const size_t recvSize = count * sizeof(float);
  const size_t sendSizeBf16 = count * numRanks * sizeof(__nv_bfloat16);
  const size_t recvSizeBf16 = count * sizeof(__nv_bfloat16);

  float *sendBuf = nullptr, *recvBuf = nullptr, *recvRS = nullptr;
  __nv_bfloat16 *sendBufBf16 = nullptr, *recvBufBf16 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, recvSize));
  CUDACHECK_TEST(cudaMalloc(&recvRS, recvSize));
  CUDACHECK_TEST(cudaMalloc(&sendBufBf16, sendSizeBf16));
  CUDACHECK_TEST(cudaMalloc(&recvBufBf16, recvSizeBf16));

  uint64_t* seedBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));

  std::vector<float> hostSend(count * numRanks);
  std::vector<__nv_bfloat16> hostSendBf16(count * numRanks);
  for (int c = 0; c < numRanks; c++) {
    for (size_t i = 0; i < count; i++) {
      float val = varianceTestValue(kBase, kBaseUlp, i, globalRank, c);
      hostSend[c * count + i] = val;
      hostSendBf16[c * count + i] = __float2bfloat16(val);
    }
  }
  CUDACHECK_TEST(
      cudaMemcpy(sendBuf, hostSend.data(), sendSize, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      sendBufBf16, hostSendBf16.data(), sendSizeBf16, cudaMemcpyHostToDevice));

  // FP32 reduce-scatter as near-exact reference.
  CUDACHECK_TEST(cudaMemset(recvRS, 0, recvSize));
  auto res = ncclReduceScatter(
      sendBuf, recvRS, count, ncclFloat32, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> hostRS(count);
  CUDACHECK_TEST(
      cudaMemcpy(hostRS.data(), recvRS, recvSize, cudaMemcpyDeviceToHost));

  // ---- RSQ path: single trial + multi-trial averaging ----

  uint64_t seed = trialSeed(0, globalRank);
  CUDACHECK_TEST(
      cudaMemcpy(seedBuf, &seed, sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(recvBuf, 0, recvSize));
  res = ncclReduceScatterQuantize(
      sendBuf,
      recvBuf,
      count,
      ncclFloat32,
      ncclBfloat16,
      ncclSum,
      seedBuf,
      comm,
      stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> hostSingle(count);
  CUDACHECK_TEST(
      cudaMemcpy(hostSingle.data(), recvBuf, recvSize, cudaMemcpyDeviceToHost));

  double singleTrialMAE = 0.0;
  for (size_t i = 0; i < count; i++) {
    singleTrialMAE += std::abs(static_cast<double>(hostSingle[i]) - hostRS[i]);
  }
  singleTrialMAE /= count;

  // Run numTrials RSQ invocations with different seeds and accumulate in FP64.
  std::vector<double> accumulated(count, 0.0);
  for (int t = 0; t < numTrials; t++) {
    seed = trialSeed(t, globalRank);
    CUDACHECK_TEST(
        cudaMemcpy(seedBuf, &seed, sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(recvBuf, 0, recvSize));

    res = ncclReduceScatterQuantize(
        sendBuf,
        recvBuf,
        count,
        ncclFloat32,
        ncclBfloat16,
        ncclSum,
        seedBuf,
        comm,
        stream);
    ASSERT_EQ(res, ncclSuccess);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<float> hostTrial(count);
    CUDACHECK_TEST(cudaMemcpy(
        hostTrial.data(), recvBuf, recvSize, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; i++) {
      accumulated[i] += static_cast<double>(hostTrial[i]);
    }
  }

  double averagedMAE = 0.0;
  for (size_t i = 0; i < count; i++) {
    double avg = accumulated[i] / numTrials;
    averagedMAE += std::abs(avg - static_cast<double>(hostRS[i]));
  }
  averagedMAE /= count;

  double rsqRatio = singleTrialMAE > 0 ? averagedMAE / singleTrialMAE : 0.0;
  printf(
      "Rank %d: MultiTrialVarianceReduction (RSQ): singleMAE=%.10f, "
      "averagedMAE=%.10f, ratio=%.4f\n",
      globalRank,
      singleTrialMAE,
      averagedMAE,
      rsqRatio);

  EXPECT_LT(averagedMAE, singleTrialMAE * kVarianceReductionThreshold)
      << "Rank " << globalRank << ": averaging " << numTrials
      << " RSQ trials did not reduce MAE. ratio=" << rsqRatio;

  // ---- BF16 RS path: verify that native BF16 RS cannot pass this test ----
  // BF16 RS is deterministic — all trials produce identical output, so
  // averaging cannot reduce error.

  CUDACHECK_TEST(cudaMemset(recvBufBf16, 0, recvSizeBf16));
  res = ncclReduceScatter(
      sendBufBf16, recvBufBf16, count, ncclBfloat16, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> hostSingleBf16(count);
  CUDACHECK_TEST(cudaMemcpy(
      hostSingleBf16.data(),
      recvBufBf16,
      recvSizeBf16,
      cudaMemcpyDeviceToHost));

  double bf16SingleMAE = 0.0;
  for (size_t i = 0; i < count; i++) {
    bf16SingleMAE += std::abs(
        static_cast<double>(__bfloat162float(hostSingleBf16[i])) - hostRS[i]);
  }
  bf16SingleMAE /= count;

  // Run BF16 RS numTrials times and accumulate (all trials are identical).
  std::vector<double> bf16Accumulated(count, 0.0);
  for (int t = 0; t < numTrials; t++) {
    CUDACHECK_TEST(cudaMemset(recvBufBf16, 0, recvSizeBf16));
    res = ncclReduceScatter(
        sendBufBf16, recvBufBf16, count, ncclBfloat16, ncclSum, comm, stream);
    ASSERT_EQ(res, ncclSuccess);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> hostTrialBf16(count);
    CUDACHECK_TEST(cudaMemcpy(
        hostTrialBf16.data(),
        recvBufBf16,
        recvSizeBf16,
        cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; i++) {
      bf16Accumulated[i] +=
          static_cast<double>(__bfloat162float(hostTrialBf16[i]));
    }
  }

  double bf16AveragedMAE = 0.0;
  for (size_t i = 0; i < count; i++) {
    double avg = bf16Accumulated[i] / numTrials;
    bf16AveragedMAE += std::abs(avg - static_cast<double>(hostRS[i]));
  }
  bf16AveragedMAE /= count;

  double bf16Ratio = bf16SingleMAE > 0 ? bf16AveragedMAE / bf16SingleMAE : 1.0;

  printf(
      "Rank %d: MultiTrialVarianceReduction (BF16): singleMAE=%.10f, "
      "averagedMAE=%.10f, ratio=%.4f\n",
      globalRank,
      bf16SingleMAE,
      bf16AveragedMAE,
      bf16Ratio);

  // BF16 RS is deterministic: averaging identical outputs does not reduce
  // error. Assert it does NOT pass the variance reduction threshold.
  EXPECT_GE(bf16AveragedMAE, bf16SingleMAE * kVarianceReductionThreshold)
      << "Rank " << globalRank
      << ": BF16 RS unexpectedly passed the variance reduction test. "
      << "ratio=" << bf16Ratio
      << ". BF16 is deterministic so averaging should not help.";

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
  CUDACHECK_TEST(cudaFree(recvRS));
  CUDACHECK_TEST(cudaFree(sendBufBf16));
  CUDACHECK_TEST(cudaFree(recvBufBf16));
  CUDACHECK_TEST(cudaFree(seedBuf));
}

// ---------------------------------------------------------------------------
// SystematicBiasDetection: construct input where BF16 RNE creates
// measurable one-sided rounding bias, while stochastic rounding is unbiased.
//
// Every input value sits at a fixed fractional position (kFracPosition)
// between two consecutive BF16 values. Since kFracPosition < 0.5, RNE
// rounds ALL of them DOWN, creating a systematic negative bias of
// ~kFracPosition ULPs at the output magnitude.
// SR rounds each randomly → unbiased (E[SR(x)] = x).
//
// If RSQ is replaced by BF16 RS, the mean signed error is ~-kFracPosition
// ULPs, exceeding the kBiasThresholdUlps threshold → assertion fails.
// ---------------------------------------------------------------------------
TEST_F(ReduceScatterQuantizeTest, SystematicBiasDetection) {
  const size_t count = 16384;

  // Base value: exactly BF16-representable so we can precisely control
  // the fractional position within the BF16 interval.
  const float kBase = 1.0f;
  const float kUlp = bf16Ulp(kBase);

  // Fractional position within the BF16 interval [kBase, kBase + kUlp).
  // Must be < 0.5 so RNE always rounds down. 0.25 gives a clear signal
  // while leaving room for perturbations.
  const float kFracPosition = 0.25f;
  const float fracOffset = kFracPosition * kUlp;

  // Perturbation budget: total perturbation across all ranks and indices
  // must stay below (0.5 - kFracPosition) * kUlp to keep all values on
  // the same side of the BF16 midpoint.
  const float perturbBudget = (0.5f - kFracPosition) * kUlp;
  const float rankPerturb = perturbBudget * 0.04f / numRanks;
  const float indexPerturb = perturbBudget * 0.8f / count;

  // Max acceptable bias in ULPs for unbiased SR. Must be between 0 and
  // kFracPosition so that RSQ passes but BF16 (with ~kFracPosition ULP
  // bias) fails. 0.15 is midway, giving margin on both sides.
  const double kBiasThresholdUlps = 0.15;

  const size_t sendSize = count * numRanks * sizeof(float);
  const size_t recvSize = count * sizeof(float);
  const size_t sendSizeBf16 = count * numRanks * sizeof(__nv_bfloat16);
  const size_t recvSizeBf16 = count * sizeof(__nv_bfloat16);

  float *sendBuf = nullptr, *recvBuf = nullptr;
  __nv_bfloat16 *sendBufBf16 = nullptr, *recvBufBf16 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, recvSize));
  CUDACHECK_TEST(cudaMalloc(&sendBufBf16, sendSizeBf16));
  CUDACHECK_TEST(cudaMalloc(&recvBufBf16, recvSizeBf16));

  uint64_t* seedBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));
  uint64_t seed = trialSeed(0, globalRank);
  CUDACHECK_TEST(
      cudaMemcpy(seedBuf, &seed, sizeof(uint64_t), cudaMemcpyHostToDevice));

  std::vector<float> hostSend(count * numRanks);
  std::vector<__nv_bfloat16> hostSendBf16(count * numRanks);
  for (int c = 0; c < numRanks; c++) {
    for (size_t i = 0; i < count; i++) {
      float val = kBase + fracOffset +
          static_cast<float>(globalRank) * rankPerturb +
          static_cast<float>(i) * indexPerturb;
      hostSend[c * count + i] = val;
      hostSendBf16[c * count + i] = __float2bfloat16(val);
    }
  }
  CUDACHECK_TEST(
      cudaMemcpy(sendBuf, hostSend.data(), sendSize, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      sendBufBf16, hostSendBf16.data(), sendSizeBf16, cudaMemcpyHostToDevice));

  // Compute FP64 expected sum for this rank's output chunk.
  std::vector<double> expected(count);
  double totalAbsExpected = 0.0;
  for (size_t i = 0; i < count; i++) {
    double exp = 0.0;
    for (int r = 0; r < numRanks; r++) {
      exp += static_cast<double>(kBase) + static_cast<double>(fracOffset) +
          static_cast<double>(r) * static_cast<double>(rankPerturb) +
          static_cast<double>(i) * static_cast<double>(indexPerturb);
    }
    expected[i] = exp;
    totalAbsExpected += std::abs(exp);
  }
  double meanAbsExpected = totalAbsExpected / count;
  float outputUlp = bf16Ulp(static_cast<float>(meanAbsExpected));

  // ---- RSQ path: stochastic rounding should be unbiased ----
  CUDACHECK_TEST(cudaMemset(recvBuf, 0, recvSize));
  auto res = ncclReduceScatterQuantize(
      sendBuf,
      recvBuf,
      count,
      ncclFloat32,
      ncclBfloat16,
      ncclSum,
      seedBuf,
      comm,
      stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> hostRecv(count);
  CUDACHECK_TEST(
      cudaMemcpy(hostRecv.data(), recvBuf, recvSize, cudaMemcpyDeviceToHost));

  double rsqTotalSignedErr = 0.0;
  for (size_t i = 0; i < count; i++) {
    rsqTotalSignedErr += static_cast<double>(hostRecv[i]) - expected[i];
  }
  double rsqMeanSignedErr = rsqTotalSignedErr / count;

  printf(
      "Rank %d: SystematicBiasDetection (RSQ): meanSignedErr=%.10f, "
      "outputUlp=%.6f, biasInUlps=%.4f\n",
      globalRank,
      rsqMeanSignedErr,
      static_cast<double>(outputUlp),
      rsqMeanSignedErr / outputUlp);

  EXPECT_LT(std::abs(rsqMeanSignedErr), kBiasThresholdUlps * outputUlp)
      << "Rank " << globalRank
      << ": RSQ stochastic rounding shows systematic bias. "
      << "Mean signed error = " << rsqMeanSignedErr << " ("
      << rsqMeanSignedErr / outputUlp << " ULPs), "
      << "threshold = " << kBiasThresholdUlps << " ULPs.";

  // ---- BF16 RS path: verify that native BF16 RS has systematic bias ----
  CUDACHECK_TEST(cudaMemset(recvBufBf16, 0, recvSizeBf16));
  res = ncclReduceScatter(
      sendBufBf16, recvBufBf16, count, ncclBfloat16, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> hostRecvBf16(count);
  CUDACHECK_TEST(cudaMemcpy(
      hostRecvBf16.data(), recvBufBf16, recvSizeBf16, cudaMemcpyDeviceToHost));

  double bf16TotalSignedErr = 0.0;
  for (size_t i = 0; i < count; i++) {
    bf16TotalSignedErr +=
        static_cast<double>(__bfloat162float(hostRecvBf16[i])) - expected[i];
  }
  double bf16MeanSignedErr = bf16TotalSignedErr / count;

  printf(
      "Rank %d: SystematicBiasDetection (BF16): meanSignedErr=%.10f, "
      "outputUlp=%.6f, biasInUlps=%.4f\n",
      globalRank,
      bf16MeanSignedErr,
      static_cast<double>(outputUlp),
      bf16MeanSignedErr / outputUlp);

  EXPECT_GE(std::abs(bf16MeanSignedErr), kBiasThresholdUlps * outputUlp)
      << "Rank " << globalRank
      << ": BF16 RS unexpectedly passed the bias test. "
      << "Mean signed error = " << bf16MeanSignedErr << " ("
      << bf16MeanSignedErr / outputUlp << " ULPs), "
      << "expected >= " << kBiasThresholdUlps
      << " ULPs due to systematic RNE rounding bias.";

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
  CUDACHECK_TEST(cudaFree(sendBufBf16));
  CUDACHECK_TEST(cudaFree(recvBufBf16));
  CUDACHECK_TEST(cudaFree(seedBuf));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
