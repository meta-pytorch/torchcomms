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
  return static_cast<uint64_t>(trial) * 9973 + static_cast<uint64_t>(rank) * 31;
}

// ---------------------------------------------------------------------------
// MultiTrialVarianceReduction: prove that averaging multiple RSQ runs
// (with different seeds) reduces error, while BF16 RS (deterministic) cannot
// benefit from averaging.
//
// This test exploits the fundamental property that stochastic rounding errors
// are independent across seeds. The deterministic (RNE) rounding error provides
// a stable baseline. With K SR trials averaged, the MAE should drop well below
// this baseline. BF16 RS is deterministic, so averaging produces no
// improvement.
// ---------------------------------------------------------------------------
TEST_F(ReduceScatterQuantizeTest, MultiTrialVarianceReduction) {
  const size_t count = 65536;
  const int numTrials = 10000;

  // Base value between BF16 representable values 1.328125 and 1.3359375,
  // chosen so every element exercises stochastic rounding.
  const float kBase = 1.33f;
  const float kBaseUlp = bf16Ulp(kBase);

  // Theoretical MAE reduction is 1/sqrt(numTrials).
  // Add 1.3x safety margin against statistical variance.
  const double kVarianceReductionThreshold = 1 / sqrt(numTrials) * 1.3;

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

  std::vector<float> hostFp32Ref(count);
  CUDACHECK_TEST(
      cudaMemcpy(hostFp32Ref.data(), recvRS, recvSize, cudaMemcpyDeviceToHost));

  // ---- BF16 RS: deterministic (RNE) rounding as stable baseline ----
  // RNE error is constant across runs, providing a reproducible baseline
  // to measure SR improvement against.
  CUDACHECK_TEST(cudaMemset(recvBufBf16, 0, recvSizeBf16));
  res = ncclReduceScatter(
      sendBufBf16, recvBufBf16, count, ncclBfloat16, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> hostBf16Rne(count);
  CUDACHECK_TEST(cudaMemcpy(
      hostBf16Rne.data(), recvBufBf16, recvSizeBf16, cudaMemcpyDeviceToHost));

  double rsBf16RneMAE = 0.0;
  for (size_t i = 0; i < count; i++) {
    rsBf16RneMAE += std::abs(
        static_cast<double>(__bfloat162float(hostBf16Rne[i])) - hostFp32Ref[i]);
  }
  rsBf16RneMAE /= count;

  // ---- RSQ path: multi-trial averaging should reduce error below
  // the deterministic RNE baseline ----
  std::vector<double> accumulated(count, 0.0);
  std::vector<double> accumulatedSq(count, 0.0);
  for (int t = 0; t < numTrials; t++) {
    uint64_t seed = trialSeed(t, globalRank);
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
      double val = static_cast<double>(hostTrial[i]);
      accumulated[i] += val;
      accumulatedSq[i] += val * val;
    }
  }

  double averagedMAE = 0.0;
  for (size_t i = 0; i < count; i++) {
    double avg = accumulated[i] / numTrials;
    averagedMAE += std::abs(avg - static_cast<double>(hostFp32Ref[i]));
  }
  averagedMAE /= count;

  double rsqRatio = rsBf16RneMAE > 0 ? averagedMAE / rsBf16RneMAE : 0.0;
  printf(
      "Rank %d: MultiTrialVarianceReduction (RSQ): rsBf16RneMAE=%.10f, "
      "averagedMAE=%.10f, ratio=%.4f\n",
      globalRank,
      rsBf16RneMAE,
      averagedMAE,
      rsqRatio);

  EXPECT_LT(averagedMAE, rsBf16RneMAE * kVarianceReductionThreshold)
      << "Rank " << globalRank << ": averaging " << numTrials
      << " RSQ trials did not reduce MAE below deterministic baseline. "
      << "ratio=" << rsqRatio;

  // Per-element unbiasedness check: the sample mean should be consistent
  // with the FP32 reference within a confidence interval. Uses 6-sigma
  // threshold giving P(false positive) ≈ 2e-9 per element, ≈~10⁻⁴ expected
  // across 65536 elements. Increasing numTrials tightens the interval
  // (detects smaller biases) without increasing false positive rate.
  const double kZScore = 6.0;
  size_t numOutliers = 0;
  for (size_t i = 0; i < count; i++) {
    double mean = accumulated[i] / numTrials;
    double meanSq = accumulatedSq[i] / numTrials;
    double variance = meanSq - mean * mean;
    double stderr = std::sqrt(std::max(variance, 0.0) / numTrials);
    double avgErr = std::abs(mean - static_cast<double>(hostFp32Ref[i]));
    if (avgErr > kZScore * stderr) {
      if (numOutliers < 10) {
        printf(
            "Rank %d: per-element unbiasedness check failed at index %zu: "
            "avgErr=%.10e, %.1f-sigma threshold=%.10e, stderr=%.10e\n",
            globalRank,
            i,
            avgErr,
            kZScore,
            kZScore * stderr,
            stderr);
      }
      numOutliers++;
    }
  }

  EXPECT_EQ(numOutliers, 0)
      << "Rank " << globalRank << ": Per-element unbiasedness check failed. "
      << numOutliers << " / " << count << " elements exceeded " << kZScore
      << "-sigma confidence interval over " << numTrials << " trials.";

  // ---- BF16 RS negative control: deterministic RS cannot benefit from
  // averaging since all trials produce identical output ----
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
    bf16AveragedMAE += std::abs(avg - static_cast<double>(hostFp32Ref[i]));
  }
  bf16AveragedMAE /= count;

  double bf16Ratio = rsBf16RneMAE > 0 ? bf16AveragedMAE / rsBf16RneMAE : 1.0;

  printf(
      "Rank %d: MultiTrialVarianceReduction (BF16): rsBf16RneMAE=%.10f, "
      "averagedMAE=%.10f, ratio=%.4f\n",
      globalRank,
      rsBf16RneMAE,
      bf16AveragedMAE,
      bf16Ratio);

  // BF16 RS is deterministic: averaging identical outputs does not reduce
  // error. Assert it does NOT pass the variance reduction threshold.
  EXPECT_GE(bf16AveragedMAE, rsBf16RneMAE * kVarianceReductionThreshold)
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
