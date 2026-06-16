// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <optional>
#include <string>
#include <vector>
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "nccl.h" // @manual

namespace ncclx::test {

// Helper class for algorithm statistics verification in tests.
//
// Usage:
//   class MyTest : public NcclxBaseTestFixture {
//    protected:
//     VerifyAlgoStatsHelper algoStats_;
//
//     void SetUp() override {
//       NcclxBaseTestFixture::SetUp();
//       algoStats_.enable();  // Must be called before comm creation
//     }
//   };
//
//   TEST_F(MyTest, Foo) {
//     // ... run collective ...
//     algoStats_.verify(comm, "ReduceScatter", "PAT");
//   }
class VerifyAlgoStatsHelper {
 public:
  VerifyAlgoStatsHelper() = default;

  // Enable AlgoStats tracing. Must be called after Cvars are initialized
  // (e.g., via initEnv()) and before NCCL comm creation.
  void enable();

  // Dump all algorithm statistics to stderr.
  // @param comm The NCCL communicator to query stats from
  // @param collective The collective name (e.g., "ReduceScatter", "AllReduce")
  void dump(ncclComm_t comm, const std::string& collective) const;

  // Verify that the expected algorithm was used via AlgoStats.
  // @param comm The NCCL communicator to query stats from
  // @param collective The collective name (e.g., "ReduceScatter", "AllReduce")
  // @param expectedAlgoSubstr Substring to match in algorithm name (e.g.,
  // "PAT", "Ring")
  void verify(
      ncclComm_t comm,
      const std::string& collective,
      const std::string& expectedAlgoSubstr) const;

  // Verify that the given algorithm was NOT used via AlgoStats.
  // @param comm The NCCL communicator to query stats from
  // @param collective The collective name (e.g., "ReduceScatter", "AllReduce")
  // @param unexpectedAlgoSubstr Substring that must NOT appear in any algorithm
  // name with nonzero call count
  void verifyNot(
      ncclComm_t comm,
      const std::string& collective,
      const std::string& unexpectedAlgoSubstr) const;

  // Verify that EVERY recorded algorithm (callCount > 0) for the collective
  // contains `expectedSubstr`, and that there is at least one such record.
  // @param comm The NCCL communicator to query stats from
  // @param collective The collective name (e.g., "ReduceScatter", "AllReduce")
  // @param expectedSubstr Substring every recorded algorithm name must contain
  void verifyExact(
      ncclComm_t comm,
      const std::string& collective,
      const std::string& expectedSubstr) const;

  // Compares the full algo-stats maps (algorithm names AND call counts) between
  // two comms for complete equality.
  // @param expected The reference comm (e.g. the no-tuner default)
  // @param actual The comm under test (e.g. with a non-matching rule)
  // @param collective The collective name (e.g., "ReduceScatter", "AllReduce")
  void verifyEqual(
      ncclComm_t expected,
      ncclComm_t actual,
      const std::string& collective) const;

 private:
  std::optional<EnvRAII<std::vector<std::string>>> colltraceGuard_;
};

} // namespace ncclx::test
