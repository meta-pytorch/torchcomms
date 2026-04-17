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
static float bf16Ulp(float value) {
  float absVal = std::max(std::abs(value), 1e-30f);
  int exponent;
  std::frexp(absVal, &exponent);
  return std::ldexp(1.0f, exponent - 8);
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
