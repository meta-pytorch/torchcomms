// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllToAllTest.hpp"

#include <gtest/gtest.h>
#include <vector>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

TEST_F(AllToAllTest, AllTests) {
  // Define the parameter combinations
  std::vector<int> counts = {0, 4, 1024, 1024 * 1024};
  std::vector<at::ScalarType> dtypes = {at::kFloat, at::kInt, at::kChar};

  // Loop over all parameter combinations
  for (int count : counts) {
    for (at::ScalarType dtype : dtypes) {
      // Create a descriptive test name for better test output
      std::string testName =
          "Count_" + std::to_string(count) + "_" + getDtypeName(dtype);

      SCOPED_TRACE("Running tests with parameters: " + testName);

      // Run all test functions with the current parameters
      SCOPED_TRACE("Running testSyncAllToAll");
      testSyncAllToAll(count, dtype);

      SCOPED_TRACE("Running testSyncAllToAllNoWork");
      testSyncAllToAllNoWork(count, dtype);

      SCOPED_TRACE("Running testAsyncAllToAll");
      testAsyncAllToAll(count, dtype);

      SCOPED_TRACE("Running testAsyncAllToAllEarlyReset");
      testAsyncAllToAllEarlyReset(count, dtype);

      SCOPED_TRACE("Running testAllToAllInputDeleted");
      testAllToAllInputDeleted(count, dtype);

      SCOPED_TRACE("Running testGraphAllToAll");
      testGraphAllToAll(count, dtype);

      SCOPED_TRACE("Running testGraphAllToAllInputDeleted");
      testGraphAllToAllInputDeleted(count, dtype);
    }
  }
}

// This main function is provided by gtest
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
