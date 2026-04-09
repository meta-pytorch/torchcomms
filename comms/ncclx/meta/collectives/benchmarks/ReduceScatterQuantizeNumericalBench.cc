// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Numerical benchmarks for ncclReduceScatterQuantize.
// These benchmarks compare RS(FP32), RS(BF16), and RS-Quantized
// (FP32→BF16→FP32) across different data patterns and report error statistics.
// They are informational (print-only, no pass/fail assertions) and meant to be
// read by a human inspecting the printed tables.
//
// Asserting correctness tests are in tests/ReduceScatterQuantizeTest.cc
// and tests/ReduceScatterQuantizeNumericalTest.cc.

#include <comm.h>
#include <cuda_bf16.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <cmath>
#include <cstddef>
#include <vector>
#include "comms/ncclx/meta/collectives/tests/ReduceScatterQuantizeTestUtils.h"

// Collect per-element error statistics for a result vector vs a reference.
struct ErrorStats {
  double maxAbs{0.0};
  double meanAbs{0.0};
  double rmsError{0.0};
  double meanSigned{0.0};
  double maxUlps{0.0};
  double meanUlps{0.0};

  void compute(
      const std::vector<float>& result,
      const std::vector<double>& reference,
      size_t count) {
    double sumAbs = 0, sumSigned = 0, sumSq = 0, sumUlps = 0;
    for (size_t i = 0; i < count; i++) {
      double diff = static_cast<double>(result[i]) - reference[i];
      double ad = std::abs(diff);
      sumAbs += ad;
      sumSigned += diff;
      sumSq += diff * diff;
      float ulp = bf16Ulp(static_cast<float>(std::abs(reference[i])));
      double ulps = ad / ulp;
      sumUlps += ulps;
      maxAbs = std::max(maxAbs, ad);
      maxUlps = std::max(maxUlps, ulps);
    }
    meanAbs = sumAbs / count;
    meanSigned = sumSigned / count;
    rmsError = std::sqrt(sumSq / count);
    meanUlps = sumUlps / count;
  }
};

// ---------------------------------------------------------------------------
// BenchCancellationError: construct input so that the exact sum is 0 for
// every element (large negative on rank 0 + small positives on other ranks).
// Measure how far RS(FP32), RS(BF16), and RSQ deviate from zero.
// ---------------------------------------------------------------------------
TEST_F(ReduceScatterQuantizeTest, BenchCancellationError) {
  const size_t count = 4096;

  // Choose a "hard" value that is not exactly representable in BF16.
  const float smallPos = 1.33f;
  const float largeNeg = -smallPos * static_cast<float>(numRanks - 1);

  const size_t sendSize = count * numRanks * sizeof(float);
  const size_t recvSize = count * sizeof(float);
  const size_t sendSizeBf16 = count * numRanks * sizeof(__nv_bfloat16);
  const size_t recvSizeBf16 = count * sizeof(__nv_bfloat16);

  float *sendBuf = nullptr, *recvRS = nullptr, *recvRSQ = nullptr;
  __nv_bfloat16 *sendBufBf16 = nullptr, *recvBufBf16 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvRS, recvSize));
  CUDACHECK_TEST(cudaMalloc(&recvRSQ, recvSize));
  CUDACHECK_TEST(cudaMalloc(&sendBufBf16, sendSizeBf16));
  CUDACHECK_TEST(cudaMalloc(&recvBufBf16, recvSizeBf16));

  uint64_t* seedBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));
  uint64_t seed = 42;
  CUDACHECK_TEST(
      cudaMemcpy(seedBuf, &seed, sizeof(uint64_t), cudaMemcpyHostToDevice));

  std::vector<float> hostSend(count * numRanks);
  std::vector<__nv_bfloat16> hostSendBf16(count * numRanks);
  for (int c = 0; c < numRanks; c++) {
    for (size_t i = 0; i < count; i++) {
      float val = (globalRank == 0) ? largeNeg : smallPos;
      hostSend[c * count + i] = val;
      hostSendBf16[c * count + i] = __float2bfloat16(val);
    }
  }
  CUDACHECK_TEST(
      cudaMemcpy(sendBuf, hostSend.data(), sendSize, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      sendBufBf16, hostSendBf16.data(), sendSizeBf16, cudaMemcpyHostToDevice));

  CUDACHECK_TEST(cudaMemset(recvRS, 0, recvSize));
  auto res = ncclReduceScatter(
      sendBuf, recvRS, count, ncclFloat32, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);

  CUDACHECK_TEST(cudaMemset(recvBufBf16, 0, recvSizeBf16));
  res = ncclReduceScatter(
      sendBufBf16, recvBufBf16, count, ncclBfloat16, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);

  CUDACHECK_TEST(cudaMemset(recvRSQ, 0, recvSize));
  res = ncclReduceScatterQuantize(
      sendBuf,
      recvRSQ,
      count,
      ncclFloat32,
      ncclBfloat16,
      ncclSum,
      seedBuf,
      comm,
      stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> hostRS(count), hostRSQ(count);
  std::vector<__nv_bfloat16> hostRecvBf16(count);
  CUDACHECK_TEST(
      cudaMemcpy(hostRS.data(), recvRS, recvSize, cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(
      cudaMemcpy(hostRSQ.data(), recvRSQ, recvSize, cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      hostRecvBf16.data(), recvBufBf16, recvSizeBf16, cudaMemcpyDeviceToHost));

  std::vector<float> hostRSBf16(count);
  for (size_t i = 0; i < count; i++) {
    hostRSBf16[i] = __bfloat162float(hostRecvBf16[i]);
  }

  std::vector<double> reference(count, 0.0);
  ErrorStats statsRS, statsRSBf16, statsRSQ;
  statsRS.compute(hostRS, reference, count);
  statsRSBf16.compute(hostRSBf16, reference, count);
  statsRSQ.compute(hostRSQ, reference, count);

  if (globalRank == 0) {
    printf(
        "\n=== Cancellation Error Benchmark (count=%zu, "
        "numRanks=%d, PAT steps=%d) ===\n",
        count,
        numRanks,
        patSteps(numRanks));
    printf(
        "  %-25s %15s %15s %15s\n", "", "RS(FP32)", "RS(BF16)", "RS-Quantized");
    printf(
        "  %-25s %15.8f %15.8f %15.8f\n",
        "max |error|",
        statsRS.maxAbs,
        statsRSBf16.maxAbs,
        statsRSQ.maxAbs);
    printf(
        "  %-25s %15.8f %15.8f %15.8f\n",
        "mean |error|",
        statsRS.meanAbs,
        statsRSBf16.meanAbs,
        statsRSQ.meanAbs);
    printf(
        "  %-25s %15.8f %15.8f %15.8f\n",
        "RMS error",
        statsRS.rmsError,
        statsRSBf16.rmsError,
        statsRSQ.rmsError);
    printf(
        "  %-25s %+15.8f %+15.8f %+15.8f\n",
        "mean signed error",
        statsRS.meanSigned,
        statsRSBf16.meanSigned,
        statsRSQ.meanSigned);
    printf(
        "  %-25s %15.2f %15.2f %15.2f\n",
        "max error (BF16 ULPs)",
        statsRS.maxUlps,
        statsRSBf16.maxUlps,
        statsRSQ.maxUlps);
    printf(
        "  %-25s %15.2f %15.2f %15.2f\n",
        "mean error (BF16 ULPs)",
        statsRS.meanUlps,
        statsRSBf16.meanUlps,
        statsRSQ.meanUlps);
    printf("\n");
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvRS));
  CUDACHECK_TEST(cudaFree(recvRSQ));
  CUDACHECK_TEST(cudaFree(sendBufBf16));
  CUDACHECK_TEST(cudaFree(recvBufBf16));
  CUDACHECK_TEST(cudaFree(seedBuf));
}

// ---------------------------------------------------------------------------
// BenchSummationAccuracy: gradient-like values from a deterministic sin
// sequence. Compare BF16 RS and RSQ error against FP32 RS reference.
// ---------------------------------------------------------------------------
TEST_F(ReduceScatterQuantizeTest, BenchSummationAccuracy) {
  const size_t count = 8192;
  const size_t sendSize = count * numRanks * sizeof(float);
  const size_t recvSize = count * sizeof(float);
  const size_t sendSizeBf16 = count * numRanks * sizeof(__nv_bfloat16);
  const size_t recvSizeBf16 = count * sizeof(__nv_bfloat16);

  float *sendBuf = nullptr, *recvRS = nullptr, *recvRSQ = nullptr;
  __nv_bfloat16 *sendBufBf16 = nullptr, *recvBufBf16 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvRS, recvSize));
  CUDACHECK_TEST(cudaMalloc(&recvRSQ, recvSize));
  CUDACHECK_TEST(cudaMalloc(&sendBufBf16, sendSizeBf16));
  CUDACHECK_TEST(cudaMalloc(&recvBufBf16, recvSizeBf16));

  uint64_t* seedBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));
  uint64_t seed = 7;
  CUDACHECK_TEST(
      cudaMemcpy(seedBuf, &seed, sizeof(uint64_t), cudaMemcpyHostToDevice));

  std::vector<float> hostSend(count * numRanks);
  std::vector<__nv_bfloat16> hostSendBf16(count * numRanks);
  for (int c = 0; c < numRanks; c++) {
    for (size_t i = 0; i < count; i++) {
      float val =
          std::sin(static_cast<float>(globalRank * 1000 + c * count + i)) *
          0.01f;
      hostSend[c * count + i] = val;
      hostSendBf16[c * count + i] = __float2bfloat16(val);
    }
  }
  CUDACHECK_TEST(
      cudaMemcpy(sendBuf, hostSend.data(), sendSize, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      sendBufBf16, hostSendBf16.data(), sendSizeBf16, cudaMemcpyHostToDevice));

  CUDACHECK_TEST(cudaMemset(recvRS, 0, recvSize));
  auto res = ncclReduceScatter(
      sendBuf, recvRS, count, ncclFloat32, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);

  CUDACHECK_TEST(cudaMemset(recvBufBf16, 0, recvSizeBf16));
  res = ncclReduceScatter(
      sendBufBf16, recvBufBf16, count, ncclBfloat16, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);

  CUDACHECK_TEST(cudaMemset(recvRSQ, 0, recvSize));
  res = ncclReduceScatterQuantize(
      sendBuf,
      recvRSQ,
      count,
      ncclFloat32,
      ncclBfloat16,
      ncclSum,
      seedBuf,
      comm,
      stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> hostRS(count), hostRSQ(count);
  std::vector<__nv_bfloat16> hostRecvBf16(count);
  CUDACHECK_TEST(
      cudaMemcpy(hostRS.data(), recvRS, recvSize, cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(
      cudaMemcpy(hostRSQ.data(), recvRSQ, recvSize, cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      hostRecvBf16.data(), recvBufBf16, recvSizeBf16, cudaMemcpyDeviceToHost));

  std::vector<float> hostRSBf16(count);
  for (size_t i = 0; i < count; i++) {
    hostRSBf16[i] = __bfloat162float(hostRecvBf16[i]);
  }

  std::vector<double> refRS(count);
  for (size_t i = 0; i < count; i++) {
    refRS[i] = static_cast<double>(hostRS[i]);
  }

  ErrorStats statsRSBf16, statsRSQ;
  statsRSBf16.compute(hostRSBf16, refRS, count);
  statsRSQ.compute(hostRSQ, refRS, count);

  if (globalRank == 0) {
    printf(
        "\n=== Summation Accuracy Benchmark (count=%zu, numRanks=%d) ===\n",
        count,
        numRanks);
    printf("  Reference: FP32 ReduceScatter (PAT)\n");
    printf("  %-30s %15s %15s\n", "", "RS(BF16)", "RS-Quantized");
    printf(
        "  %-30s %15.10f %15.10f\n",
        "max |error|",
        statsRSBf16.maxAbs,
        statsRSQ.maxAbs);
    printf(
        "  %-30s %15.10f %15.10f\n",
        "mean |error|",
        statsRSBf16.meanAbs,
        statsRSQ.meanAbs);
    printf(
        "  %-30s %15.10f %15.10f\n",
        "RMS error",
        statsRSBf16.rmsError,
        statsRSQ.rmsError);
    printf(
        "  %-30s %+15.10f %+15.10f\n",
        "mean signed error",
        statsRSBf16.meanSigned,
        statsRSQ.meanSigned);
    printf(
        "  %-30s %15.4f %15.4f\n",
        "max error (BF16 ULPs)",
        statsRSBf16.maxUlps,
        statsRSQ.maxUlps);
    printf(
        "  %-30s %15.4f %15.4f\n",
        "mean error (BF16 ULPs)",
        statsRSBf16.meanUlps,
        statsRSQ.meanUlps);
    printf("\n");
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvRS));
  CUDACHECK_TEST(cudaFree(recvRSQ));
  CUDACHECK_TEST(cudaFree(sendBufBf16));
  CUDACHECK_TEST(cudaFree(recvBufBf16));
  CUDACHECK_TEST(cudaFree(seedBuf));
}

// ---------------------------------------------------------------------------
// BenchSRBiasConvergence: run multiple RSQ trials with different seeds and
// average the results. If SR is truly unbiased, the averaged error should
// converge toward the FP32 RS result. Reports single-trial vs averaged
// error statistics.
// ---------------------------------------------------------------------------
TEST_F(ReduceScatterQuantizeTest, BenchSRBiasConvergence) {
  const size_t count = 4096;
  const int numTrials = 16;
  const size_t sendSize = count * numRanks * sizeof(float);
  const size_t recvSize = count * sizeof(float);

  float *sendBuf = nullptr, *recvRS = nullptr, *recvRSQ = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvRS, recvSize));
  CUDACHECK_TEST(cudaMalloc(&recvRSQ, recvSize));

  uint64_t* seedBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));

  const float kBase = 1.33f;
  const float kUlp = bf16Ulp(kBase);
  std::vector<float> hostSend(count * numRanks);
  for (int c = 0; c < numRanks; c++) {
    for (size_t i = 0; i < count; i++) {
      hostSend[c * count + i] = kBase + static_cast<float>(i) * (kUlp / 64.0f) +
          static_cast<float>(globalRank) * (kUlp / 8.0f);
    }
  }
  CUDACHECK_TEST(
      cudaMemcpy(sendBuf, hostSend.data(), sendSize, cudaMemcpyHostToDevice));

  CUDACHECK_TEST(cudaMemset(recvRS, 0, recvSize));
  auto res = ncclReduceScatter(
      sendBuf, recvRS, count, ncclFloat32, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> hostRS(count);
  CUDACHECK_TEST(
      cudaMemcpy(hostRS.data(), recvRS, recvSize, cudaMemcpyDeviceToHost));

  std::vector<double> refRS(count);
  for (size_t i = 0; i < count; i++) {
    refRS[i] = static_cast<double>(hostRS[i]);
  }

  std::vector<double> accumulated(count, 0.0);
  for (int t = 0; t < numTrials; t++) {
    uint64_t seed = static_cast<uint64_t>(t * 9973 + globalRank * 31);
    CUDACHECK_TEST(
        cudaMemcpy(seedBuf, &seed, sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(recvRSQ, 0, recvSize));

    res = ncclReduceScatterQuantize(
        sendBuf,
        recvRSQ,
        count,
        ncclFloat32,
        ncclBfloat16,
        ncclSum,
        seedBuf,
        comm,
        stream);
    ASSERT_EQ(res, ncclSuccess);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<float> hostRSQ(count);
    CUDACHECK_TEST(
        cudaMemcpy(hostRSQ.data(), recvRSQ, recvSize, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; i++) {
      accumulated[i] += static_cast<double>(hostRSQ[i]);
    }
  }

  std::vector<float> avgResult(count);
  for (size_t i = 0; i < count; i++) {
    avgResult[i] = static_cast<float>(accumulated[i] / numTrials);
  }

  std::vector<float> lastTrial(count);
  CUDACHECK_TEST(
      cudaMemcpy(lastTrial.data(), recvRSQ, recvSize, cudaMemcpyDeviceToHost));

  ErrorStats statsSingle, statsAvg;
  statsSingle.compute(lastTrial, refRS, count);
  statsAvg.compute(avgResult, refRS, count);

  if (globalRank == 0) {
    printf(
        "\n=== SR Bias Convergence Benchmark (count=%zu, "
        "numTrials=%d, numRanks=%d) ===\n",
        count,
        numTrials,
        numRanks);
    printf("  Reference: FP32 ReduceScatter (PAT)\n");
    printf("  %-30s %15s %15s\n", "", "SingleTrial", "Avg-of-Trials");
    printf(
        "  %-30s %15.10f %15.10f\n",
        "max |error|",
        statsSingle.maxAbs,
        statsAvg.maxAbs);
    printf(
        "  %-30s %15.10f %15.10f\n",
        "mean |error|",
        statsSingle.meanAbs,
        statsAvg.meanAbs);
    printf(
        "  %-30s %15.10f %15.10f\n",
        "RMS error",
        statsSingle.rmsError,
        statsAvg.rmsError);
    printf(
        "  %-30s %+15.10f %+15.10f\n",
        "mean signed error",
        statsSingle.meanSigned,
        statsAvg.meanSigned);
    printf(
        "  %-30s %15.4f %15.4f\n",
        "max error (BF16 ULPs)",
        statsSingle.maxUlps,
        statsAvg.maxUlps);
    printf(
        "  %-30s %15.4f %15.4f\n",
        "mean error (BF16 ULPs)",
        statsSingle.meanUlps,
        statsAvg.meanUlps);
    printf("\n");
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvRS));
  CUDACHECK_TEST(cudaFree(recvRSQ));
  CUDACHECK_TEST(cudaFree(seedBuf));
}

// ---------------------------------------------------------------------------
// BenchDynamicRangeSweep: evaluate RS(BF16) and RSQ error across several
// magnitudes (1e-3 to 1e+3) to show how quantization error scales with
// value magnitude. Uses FP32 RS as reference.
// ---------------------------------------------------------------------------
TEST_F(ReduceScatterQuantizeTest, BenchDynamicRangeSweep) {
  const size_t count = 2048;
  const size_t sendSize = count * numRanks * sizeof(float);
  const size_t recvSize = count * sizeof(float);
  const size_t sendSizeBf16 = count * numRanks * sizeof(__nv_bfloat16);
  const size_t recvSizeBf16 = count * sizeof(__nv_bfloat16);

  float *sendBuf = nullptr, *recvRS = nullptr, *recvRSQ = nullptr;
  __nv_bfloat16 *sendBufBf16 = nullptr, *recvBufBf16 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvRS, recvSize));
  CUDACHECK_TEST(cudaMalloc(&recvRSQ, recvSize));
  CUDACHECK_TEST(cudaMalloc(&sendBufBf16, sendSizeBf16));
  CUDACHECK_TEST(cudaMalloc(&recvBufBf16, recvSizeBf16));

  uint64_t* seedBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));
  uint64_t seed = 123;
  CUDACHECK_TEST(
      cudaMemcpy(seedBuf, &seed, sizeof(uint64_t), cudaMemcpyHostToDevice));

  if (globalRank == 0) {
    printf(
        "\n=== Dynamic Range Sweep (count=%zu, numRanks=%d) ===\n",
        count,
        numRanks);
    printf("  Reference: FP32 ReduceScatter (PAT)\n");
    printf(
        "  %-12s %15s %15s %15s %15s\n",
        "Magnitude",
        "BF16 mean|err|",
        "RSQ mean|err|",
        "BF16 meanULP",
        "RSQ meanULP");
  }

  const float kBase = 1.33f;
  const float kBaseUlp = bf16Ulp(kBase);
  std::vector<float> magnitudes = {1e-3f, 1e-2f, 1e-1f, 1.0f, 1e1f, 1e2f, 1e3f};

  for (float mag : magnitudes) {
    std::vector<float> hostSend(count * numRanks);
    std::vector<__nv_bfloat16> hostSendBf16(count * numRanks);
    for (int c = 0; c < numRanks; c++) {
      for (size_t i = 0; i < count; i++) {
        float val = mag * (kBase + static_cast<float>(i % 128) * kBaseUlp) +
            static_cast<float>(globalRank) * mag * 0.01f;
        hostSend[c * count + i] = val;
        hostSendBf16[c * count + i] = __float2bfloat16(val);
      }
    }
    CUDACHECK_TEST(
        cudaMemcpy(sendBuf, hostSend.data(), sendSize, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        sendBufBf16,
        hostSendBf16.data(),
        sendSizeBf16,
        cudaMemcpyHostToDevice));

    CUDACHECK_TEST(cudaMemset(recvRS, 0, recvSize));
    auto res = ncclReduceScatter(
        sendBuf, recvRS, count, ncclFloat32, ncclSum, comm, stream);
    ASSERT_EQ(res, ncclSuccess);

    CUDACHECK_TEST(cudaMemset(recvBufBf16, 0, recvSizeBf16));
    res = ncclReduceScatter(
        sendBufBf16, recvBufBf16, count, ncclBfloat16, ncclSum, comm, stream);
    ASSERT_EQ(res, ncclSuccess);

    CUDACHECK_TEST(cudaMemset(recvRSQ, 0, recvSize));
    res = ncclReduceScatterQuantize(
        sendBuf,
        recvRSQ,
        count,
        ncclFloat32,
        ncclBfloat16,
        ncclSum,
        seedBuf,
        comm,
        stream);
    ASSERT_EQ(res, ncclSuccess);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<float> hostRS(count), hostRSQ(count);
    std::vector<__nv_bfloat16> hostRecvBf16(count);
    CUDACHECK_TEST(
        cudaMemcpy(hostRS.data(), recvRS, recvSize, cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(
        cudaMemcpy(hostRSQ.data(), recvRSQ, recvSize, cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaMemcpy(
        hostRecvBf16.data(),
        recvBufBf16,
        recvSizeBf16,
        cudaMemcpyDeviceToHost));

    std::vector<float> hostRSBf16(count);
    for (size_t i = 0; i < count; i++) {
      hostRSBf16[i] = __bfloat162float(hostRecvBf16[i]);
    }

    std::vector<double> refRS(count);
    for (size_t i = 0; i < count; i++) {
      refRS[i] = static_cast<double>(hostRS[i]);
    }

    ErrorStats statsRSBf16, statsRSQ;
    statsRSBf16.compute(hostRSBf16, refRS, count);
    statsRSQ.compute(hostRSQ, refRS, count);

    if (globalRank == 0) {
      printf(
          "  %-12.0e %15.10f %15.10f %15.4f %15.4f\n",
          static_cast<double>(mag),
          statsRSBf16.meanAbs,
          statsRSQ.meanAbs,
          statsRSBf16.meanUlps,
          statsRSQ.meanUlps);
    }
  }
  if (globalRank == 0) {
    printf("\n");
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvRS));
  CUDACHECK_TEST(cudaFree(recvRSQ));
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
