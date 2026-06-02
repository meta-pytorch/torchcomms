// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

namespace comms::pipes::test {

struct AllReduceTestConfig {
  std::size_t total_elements;
  int num_blocks;
  std::string name;
};

class AllReduceTestBase : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void fill_input(float* input_d, std::size_t total_elements) {
    std::vector<float> h_input(total_elements);
    for (std::size_t i = 0; i < total_elements; i++) {
      h_input[i] = static_cast<float>((globalRank * 7 + i) % 100) * 0.01f;
    }
    CUDACHECK_TEST(cudaMemcpy(
        input_d,
        h_input.data(),
        total_elements * sizeof(float),
        cudaMemcpyHostToDevice));
  }

  void verify_allreduce(const float* output_d, std::size_t total_elements) {
    std::vector<float> h_output(total_elements);
    CUDACHECK_TEST(cudaMemcpy(
        h_output.data(),
        output_d,
        total_elements * sizeof(float),
        cudaMemcpyDeviceToHost));

    int errors = 0;
    for (std::size_t i = 0; i < total_elements; i++) {
      float expected = 0.0f;
      for (int rank = 0; rank < worldSize; rank++) {
        expected += static_cast<float>((rank * 7 + i) % 100) * 0.01f;
      }
      float actual = h_output[i];
      if (std::abs(actual - expected) > 1e-2f) {
        if (errors < 10) {
          EXPECT_NEAR(actual, expected, 1e-2f)
              << "Mismatch at offset=" << i << " (my_rank=" << globalRank
              << ")";
        }
        errors++;
      }
    }
    EXPECT_EQ(errors, 0) << "Total mismatches: " << errors << " out of "
                         << total_elements
                         << " elements (my_rank=" << globalRank << ")";
  }
};

} // namespace comms::pipes::test
