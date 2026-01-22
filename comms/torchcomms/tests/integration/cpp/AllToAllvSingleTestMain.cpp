// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllToAllvSingleTest.hpp"

#include <gtest/gtest.h>
#include <vector>

TEST_F(AllToAllvSingleTest, AllTests) {
  // Define enum for size patterns
  enum class SizePattern { Uniform, Variable, ZeroSizes };

  // Helper function to convert enum to string
  auto getPatternName = [](SizePattern pattern) -> std::string {
    switch (pattern) {
      case SizePattern::Uniform:
        return "Uniform";
      case SizePattern::Variable:
        return "Variable";
      case SizePattern::ZeroSizes:
        return "ZeroSizes";
      default:
        return "Unknown";
    }
  };

  // Define different counts to test
  std::vector<uint64_t> counts = {4, 1024, 1024 * 1024};

  // Define size patterns to test (excluding AllZero which is handled
  // separately)
  std::vector<SizePattern> size_patterns = {
      SizePattern::Uniform, SizePattern::Variable, SizePattern::ZeroSizes};

  // Define datatypes to test
  std::vector<at::ScalarType> dtypes = {at::kFloat, at::kInt, at::kChar};

  // Nested loops for all parameter combinations (counts x patterns x dtypes)
  for (uint64_t count : counts) {
    for (SizePattern pattern : size_patterns) {
      for (at::ScalarType dtype : dtypes) {
        // Create size vectors based on pattern and count
        std::vector<uint64_t> input_sizes, output_sizes;

        switch (pattern) {
          case SizePattern::Uniform:
            // Test with uniform sizes
            input_sizes = std::vector<uint64_t>(num_ranks_, count);
            output_sizes = std::vector<uint64_t>(num_ranks_, count);
            break;
          case SizePattern::Variable:
            // Test with variable sizes - create a symmetric communication
            // pattern Each rank i sends (i+1)*count elements to each other
            // rank j So rank j should expect to receive (i+1)*count elements
            // from rank i
            for (int i = 0; i < num_ranks_; i++) {
              // This rank sends (rank_+1)*count elements to rank i
              input_sizes.push_back((rank_ + 1) * count);
              // This rank receives (i+1)*count elements from rank i
              output_sizes.push_back((i + 1) * count);
            }
            break;
          case SizePattern::ZeroSizes:
            // Test with some zero sizes - ensure symmetric pattern
            for (int i = 0; i < num_ranks_; i++) {
              // Create a pattern where some communications have zero size
              // If this rank sends 0 to rank i, then rank i sends 0 to this
              // rank
              uint64_t size = (rank_ + i) % 3 == 0 ? 0 : count;
              input_sizes.push_back(size);

              // This rank receives from rank i what rank i sends to this rank
              uint64_t recv_size = (i + rank_) % 3 == 0 ? 0 : count;
              output_sizes.push_back(recv_size);
            }
            break;
          default:
            TORCH_INTERNAL_ASSERT(false, "Unexpected SizePattern enum value");
        }

        // Create a descriptive test name for better test output
        std::string testName = getPatternName(pattern) + "_" +
            std::to_string(count) + "_" + getDtypeName(dtype);

        SCOPED_TRACE("Running tests with parameters: " + testName);

        // Run all test functions with clear tracing, passing parameters
        // directly
        SCOPED_TRACE("Running testSyncAllToAllvSingle");
        testSyncAllToAllvSingle(input_sizes, output_sizes, dtype);

        SCOPED_TRACE("Running testSyncAllToAllvSingleNoWork");
        testSyncAllToAllvSingleNoWork(input_sizes, output_sizes, dtype);

        SCOPED_TRACE("Running testAsyncAllToAllvSingle");
        testAsyncAllToAllvSingle(input_sizes, output_sizes, dtype);

        SCOPED_TRACE("Running testAsyncAllToAllvSingleEarlyReset");
        testAsyncAllToAllvSingleEarlyReset(input_sizes, output_sizes, dtype);

        SCOPED_TRACE("Running testAllToAllvSingleInputDeleted");
        testAllToAllvSingleInputDeleted(input_sizes, output_sizes, dtype);

        // Run CUDA Graph tests
        SCOPED_TRACE("Running testGraphAllToAllvSingle");
        testGraphAllToAllvSingle(input_sizes, output_sizes, dtype);

        SCOPED_TRACE("Running testGraphAllToAllvSingleInputDeleted");
        testGraphAllToAllvSingleInputDeleted(input_sizes, output_sizes, dtype);

        SCOPED_TRACE("Running testSyncAllToAllvSingleMultiDimTensor");
        testSyncAllToAllvSingleMultiDimTensor(input_sizes, output_sizes, dtype);
      }
    }
  }

  // Handle AllZero separately, as it is independent of the count and datatype
  for (at::ScalarType dtype : dtypes) {
    // Test with all zero sizes
    std::vector<uint64_t> input_sizes = std::vector<uint64_t>(num_ranks_, 0);
    std::vector<uint64_t> output_sizes = std::vector<uint64_t>(num_ranks_, 0);

    // Create a descriptive test name for better test output
    std::string testName = "AllZero_" + getDtypeName(dtype);

    SCOPED_TRACE("Running tests with parameters: " + testName);

    // Run all test functions with clear tracing, passing parameters directly
    SCOPED_TRACE("Running testSyncAllToAllvSingle");
    testSyncAllToAllvSingle(input_sizes, output_sizes, dtype);

    SCOPED_TRACE("Running testSyncAllToAllvSingleNoWork");
    testSyncAllToAllvSingleNoWork(input_sizes, output_sizes, dtype);

    SCOPED_TRACE("Running testAsyncAllToAllvSingle");
    testAsyncAllToAllvSingle(input_sizes, output_sizes, dtype);

    SCOPED_TRACE("Running testAsyncAllToAllvSingleEarlyReset");
    testAsyncAllToAllvSingleEarlyReset(input_sizes, output_sizes, dtype);

    SCOPED_TRACE("Running testAllToAllvSingleInputDeleted");
    testAllToAllvSingleInputDeleted(input_sizes, output_sizes, dtype);

    SCOPED_TRACE("Running testGraphAllToAllvSingle");
    testGraphAllToAllvSingle(input_sizes, output_sizes, dtype);

    SCOPED_TRACE("Running testGraphAllToAllvSingleInputDeleted");
    testGraphAllToAllvSingleInputDeleted(input_sizes, output_sizes, dtype);

    SCOPED_TRACE("Running testSyncAllToAllvSingleMultiDimTensor");
    testSyncAllToAllvSingleMultiDimTensor(input_sizes, output_sizes, dtype);
  }

  // Test asymmetric communication: some ranks have all zero inputs but non-zero
  // outputs
  for (uint64_t count : counts) {
    for (at::ScalarType dtype : dtypes) {
      // Create an asymmetric pattern where even ranks send data but odd ranks
      // don't However, odd ranks receive data from even ranks
      std::vector<uint64_t> input_sizes, output_sizes;

      for (int i = 0; i < num_ranks_; i++) {
        if (rank_ % 2 == 0) {
          // Even ranks send data to all ranks
          input_sizes.push_back(count);
        } else {
          // Odd ranks don't send any data
          input_sizes.push_back(0);
        }

        if (i % 2 == 0) {
          // All ranks receive data from even ranks
          output_sizes.push_back(count);
        } else {
          // All ranks don't receive from odd ranks (since they don't send)
          output_sizes.push_back(0);
        }
      }

      // Create a descriptive test name for better test output
      std::string testName = "AsymmetricZeroInput_" + std::to_string(count) +
          "_" + getDtypeName(dtype);

      SCOPED_TRACE("Running tests with parameters: " + testName);

      // Run all test functions with clear tracing, passing parameters directly
      SCOPED_TRACE("Running testSyncAllToAllvSingle");
      testSyncAllToAllvSingle(input_sizes, output_sizes, dtype);

      SCOPED_TRACE("Running testSyncAllToAllvSingleNoWork");
      testSyncAllToAllvSingleNoWork(input_sizes, output_sizes, dtype);

      SCOPED_TRACE("Running testAsyncAllToAllvSingle");
      testAsyncAllToAllvSingle(input_sizes, output_sizes, dtype);

      SCOPED_TRACE("Running testAsyncAllToAllvSingleEarlyReset");
      testAsyncAllToAllvSingleEarlyReset(input_sizes, output_sizes, dtype);

      SCOPED_TRACE("Running testAllToAllvSingleInputDeleted");
      testAllToAllvSingleInputDeleted(input_sizes, output_sizes, dtype);

      // Run CUDA Graph tests
      SCOPED_TRACE("Running testGraphAllToAllvSingle");
      testGraphAllToAllvSingle(input_sizes, output_sizes, dtype);

      SCOPED_TRACE("Running testGraphAllToAllvSingleInputDeleted");
      testGraphAllToAllvSingleInputDeleted(input_sizes, output_sizes, dtype);

      SCOPED_TRACE("Running testSyncAllToAllvSingleMultiDimTensor");
      testSyncAllToAllvSingleMultiDimTensor(input_sizes, output_sizes, dtype);
    }
  }
}

// This main function is provided by gtest
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
