// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <nccl.h>
#include <stdlib.h>
#include <cmath>
#include <cstddef>
#include <optional>

#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"

// Compute the number of PAT reduction steps = log2(numRanks).
static int patSteps(int numRanks) {
  int steps = 0;
  int n = numRanks;
  while (n > 1) {
    n >>= 1;
    steps++;
  }
  return steps;
}

// BF16 ULP (unit in the last place) at a given magnitude.
// BF16 has 7 mantissa bits, so ULP = 2^(exponent - 8).
constexpr float bf16Ulp(float value) {
  float absVal = std::max(std::abs(value), 1e-30f);
  int exponent;
  std::frexp(absVal, &exponent);
  return std::ldexp(1.0f, exponent - 8);
}

// Compute a test value for a given (index, rank, chunk) triple.
// base must not be exactly BF16-representable so that every element
// exercises stochastic rounding. Perturbation scales are fractions of the
// BF16 ULP at the base magnitude, ensuring values span multiple BF16
// intervals across elements.
static float
varianceTestValue(float base, float ulp, size_t i, int rank, int chunk) {
  return base + static_cast<float>(i) * (ulp / 16.0f) +
      static_cast<float>(rank) * ulp + static_cast<float>(chunk) * (ulp * 2.0f);
}

// Compute the expected sum of varianceTestValue across all ranks for a given
// element index and chunk (output rank). This is the analytical result of
// reduce-scatter with ncclSum.
static float varianceTestExpectedSum(
    float base,
    float ulp,
    size_t i,
    int chunk,
    int numRanks) {
  return static_cast<float>(numRanks) * base +
      static_cast<float>(i) * (ulp / 16.0f) * static_cast<float>(numRanks) +
      ulp * static_cast<float>(numRanks) * static_cast<float>(numRanks - 1) /
      2.0f +
      static_cast<float>(chunk) * (ulp * 2.0f) * static_cast<float>(numRanks);
}

// Test fixture for ReduceScatterQuantize tests.
// Sets NCCL_PAT_ENABLE=1 and NCCL_ALGO=PAT, creates an NCCL communicator
// and a CUDA stream.
class ReduceScatterQuantizeTest : public NcclxBaseTestFixture {
 public:
  ReduceScatterQuantizeTest() = default;
  void SetUp() override {
    NcclxBaseTestFixture::SetUp({
        {"NCCL_PAT_ENABLE", "1"},
        {"NCCL_ALGO", "PAT"},
    });
    commRAII_.emplace(globalRank, numRanks, localRank, bootstrap_.get());
    comm = commRAII_->get();
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    commRAII_.reset();
    NcclxBaseTestFixture::TearDown();
  }

 protected:
  std::optional<ncclx::test::NcclCommRAII> commRAII_;
  cudaStream_t stream;
};
