// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllToAllSingleTest.hpp"

#include <gtest/gtest.h>
#include <vector>

TEST_F(AllToAllSingleTest, AllTests) {
  // Define parameter values directly in the test
  std::vector<int> counts = {0, 4, 1024, 1024 * 1024};
  std::vector<at::ScalarType> dtypes = {at::kFloat, at::kInt, at::kChar};

  // Nested loops for all parameter combinations
  for (int count : counts) {
    for (at::ScalarType dtype : dtypes) {
      // Create a descriptive test name for better test output
      std::string testName =
          "Count_" + std::to_string(count) + "_" + getDtypeName(dtype);

      SCOPED_TRACE("Running tests with parameters: " + testName);

      // Run all test functions with clear tracing, passing parameters directly
      SCOPED_TRACE("Running testSyncAllToAllSingle");
      testSyncAllToAllSingle(count, dtype);

      SCOPED_TRACE("Running testSyncAllToAllSingleNoWork");
      testSyncAllToAllSingleNoWork(count, dtype);

      SCOPED_TRACE("Running testAsyncAllToAllSingle");
      testAsyncAllToAllSingle(count, dtype);

      SCOPED_TRACE("Running testAsyncAllToAllSingleEarlyReset");
      testAsyncAllToAllSingleEarlyReset(count, dtype);

      SCOPED_TRACE("Running testAllToAllSingleInputDeleted");
      testAllToAllSingleInputDeleted(count, dtype);

      // Run CUDA Graph tests
      SCOPED_TRACE("Running testGraphAllToAllSingle");
      testGraphAllToAllSingle(count, dtype);

      SCOPED_TRACE("Running testGraphAllToAllSingleInputDeleted");
      testGraphAllToAllSingleInputDeleted(count, dtype);
    }
  }
}

// This main function is provided by gtest
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
