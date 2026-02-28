// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <sys/types.h>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace torch::comms::test {

struct RankExpectation {
  // Exit expectations — set both for "either is OK"
  std::optional<int> exitCode;
  std::optional<int> signal;

  // Log expectations (matched against captured stderr)
  std::vector<std::string> logMustContain;
  std::vector<std::string> logMustNotContain;
};

// Utility class providing execution mode dispatch and child process
// orchestration for timeout tests. Does not inherit from gtest — used via
// composition.
class TimeoutTestHelper {
 public:
  enum class ExecMode {
    kEager,
    kMultiGraphSequential,
    kMultiGraphConcurrent,
  };

  static std::string execModeName(ExecMode mode);

  void exec(const ExecMode mode, const std::vector<std::function<void()>>& ops);

  // Fork child processes with redirected stderr, run childBody in each,
  // then verify exit status and log expectations.
  void launch(
      const std::string& testName,
      const int numChildren,
      const std::function<void(int rank)>& childBody,
      const std::vector<RankExpectation>& expectations,
      const int childWaitTimeoutSecs = 60);
};

} // namespace torch::comms::test
