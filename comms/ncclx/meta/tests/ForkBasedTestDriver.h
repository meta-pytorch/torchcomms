// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <torch/csrc/distributed/c10d/TCPStore.hpp> // @manual=//caffe2:libtorch

namespace ncclx::test {

// Fork-based test driver for multi-process NCCLX tests that need to inspect
// worker exit codes (e.g., watchdog crash tests). The test driver process
// re-execs the test binary as worker subprocesses, each scoped to the current
// test case via --gtest_filter.
//
// Usage:
//   class MyTest : public ForkBasedTestDriver, public ::testing::Test {
//    public:
//     void SetUp() override {
//       ForkBasedTestDriver::SetUp(Config{.numRanks = 4});
//     }
//   };
//
//   TEST_F(MyTest, Foo) {
//     if (isTestDriverProcess()) {
//       // Inspect exit codes via state_
//       return;
//     }
//     // Worker logic
//     auto rank = getRank();
//   }
class ForkBasedTestDriver {
 public:
  struct Config {
    int numRanks{1};
    bool shouldExitOnFailure{true};
    std::vector<std::string> env; // "KEY=VALUE" format
  };

  struct WorkerState {
    std::shared_ptr<c10d::TCPStore> store;
  };

  struct TestDriverState {
    // Exit codes for each worker process.
    // Signal-terminated workers are reported as 128 + signal number.
    std::vector<int> workerExitCodes;
  };

  void SetUp(const Config& config);
  void TearDown();

  static bool isTestDriverProcess();
  static int getRank();
  static int getWorldSize();
  static std::shared_ptr<c10d::TCPStore> getStore();

  // KV convenience helpers (wrapping TCPStore set/get)
  static void setKey(const std::string& key, const std::string& value);
  static std::string waitForKey(const std::string& key);
  static std::optional<std::string> waitForKey(
      const std::string& key,
      std::chrono::milliseconds timeout);

  const TestDriverState& getTestDriverState();

  // Sets CUDA device for the given rank and returns the device ID
  static int getCudaDeviceId(int rank);

  using State = std::variant<WorkerState, TestDriverState>;
  State state_; // Default-constructed as WorkerState.
};

} // namespace ncclx::test
