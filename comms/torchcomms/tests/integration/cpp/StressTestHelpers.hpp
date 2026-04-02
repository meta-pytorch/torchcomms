// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Shared helpers for stress device API tests.
// Provides configuration from environment variables, message size sweeps,
// scope parameterization, and verification utilities.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "comms/torchcomms/device/TorchCommDeviceWindow.hpp"

namespace torchcomms::device::test {

// Configuration parsed from environment variables.
struct StressTestConfig {
  // Number of iterations for soak-style tests (default: 50).
  int num_iterations{50};

  // Message sizes in bytes to sweep (default: 4B, 1KB, 1MB, 16MB).
  std::vector<size_t> msg_sizes{4, 1024, 1048576, 16777216};

  // Scopes to test (default: all three).
  std::vector<CoopScope> scopes{
      CoopScope::THREAD,
      CoopScope::WARP,
      CoopScope::BLOCK};

  // Number of windows for multi-window tests (default: 4).
  int window_count{4};

  // Number of communicators for multi-comm tests (default: 2).
  int comm_count{2};

  // Number of create/destroy cycles for lifecycle tests (default: 50).
  int lifecycle_cycles{50};

  // Verbose logging per iteration.
  bool verbose{false};
};

// Parse configuration from environment variables:
//   NUM_ITERATIONS, STRESS_MSG_SIZES (comma-separated bytes),
//   STRESS_SCOPES (comma-separated: thread,warp,block),
//   STRESS_WINDOW_COUNT, STRESS_COMM_COUNT,
//   STRESS_LIFECYCLE_CYCLES, STRESS_VERBOSE
StressTestConfig parseStressTestConfig();

// Check if stress tests should run (RUN_DEVICE_STRESS_TEST env var).
// Returns true if the env var is set to "1" or "true".
bool shouldRunStressTest();

// Map CoopScope to the number of threads to launch.
int threadsForScope(CoopScope scope);

// Human-readable scope name.
const char* scopeName(CoopScope scope);

// Format a byte count for logging (e.g., "1.00 KB", "16.00 MB").
std::string formatBytes(size_t bytes);

// Fill pattern value for verification: encodes rank and iteration into data.
// Pattern: (rank + 1) * 1000.0f + iteration
// This uniquely identifies which rank and iteration produced the data.
inline float fillPatternValue(int rank, int iteration) {
  return static_cast<float>((rank + 1) * 1000 + iteration);
}

// Verify that a host-side float buffer matches the expected fill pattern.
// Returns true if all elements match, false otherwise.
// On mismatch, sets out_first_bad_index and out_got / out_expected.
bool verifyFillPattern(
    const float* data,
    size_t count,
    int expected_rank,
    int expected_iteration,
    size_t* out_first_bad_index = nullptr,
    float* out_got = nullptr,
    float* out_expected = nullptr);

} // namespace torchcomms::device::test
