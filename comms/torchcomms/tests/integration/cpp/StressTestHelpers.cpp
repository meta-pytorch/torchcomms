// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "StressTestHelpers.hpp"

#include <folly/String.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace torchcomms::device::test {

namespace {

// Parse a comma-separated list of integers from an env var.
std::vector<size_t> parseSizeList(const char* env_val) {
  std::vector<size_t> result;
  std::vector<folly::StringPiece> parts;
  folly::split(',', env_val, parts);
  for (const auto& part : parts) {
    if (!part.empty()) {
      result.push_back(folly::to<size_t>(part));
    }
  }
  return result;
}

// Parse a comma-separated list of scope names.
std::vector<CoopScope> parseScopeList(const char* env_val) {
  std::vector<CoopScope> result;
  std::string s(env_val);
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  std::vector<folly::StringPiece> parts;
  folly::split(',', s, parts);
  for (const auto& part : parts) {
    if (part == "thread") {
      result.push_back(CoopScope::THREAD);
    } else if (part == "warp") {
      result.push_back(CoopScope::WARP);
    } else if (part == "block") {
      result.push_back(CoopScope::BLOCK);
    }
  }
  return result;
}

bool envBool(const char* name) {
  const char* val = std::getenv(name);
  if (!val) {
    return false;
  }
  std::string s(val);
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  return s == "1" || s == "true" || s == "yes" || s == "y" || s == "t";
}

int envInt(const char* name, int default_val) {
  const char* val = std::getenv(name);
  if (!val) {
    return default_val;
  }
  return std::atoi(val);
}

} // namespace

StressTestConfig parseStressTestConfig() {
  StressTestConfig config;

  config.num_iterations = envInt("NUM_ITERATIONS", config.num_iterations);
  config.window_count = envInt("STRESS_WINDOW_COUNT", config.window_count);
  config.comm_count = envInt("STRESS_COMM_COUNT", config.comm_count);
  config.lifecycle_cycles =
      envInt("STRESS_LIFECYCLE_CYCLES", config.lifecycle_cycles);
  config.verbose = envBool("STRESS_VERBOSE");

  const char* sizes_env = std::getenv("STRESS_MSG_SIZES");
  if (sizes_env) {
    auto parsed = parseSizeList(sizes_env);
    if (!parsed.empty()) {
      config.msg_sizes = std::move(parsed);
    }
  }

  const char* scopes_env = std::getenv("STRESS_SCOPES");
  if (scopes_env) {
    auto parsed = parseScopeList(scopes_env);
    if (!parsed.empty()) {
      config.scopes = std::move(parsed);
    }
  }

  return config;
}

bool shouldRunStressTest() {
  return envBool("RUN_DEVICE_STRESS_TEST");
}

int threadsForScope(CoopScope scope) {
  switch (scope) {
    case CoopScope::THREAD:
      return 1;
    case CoopScope::WARP:
      return 32;
    case CoopScope::BLOCK:
      return 256;
  }
  return 1;
}

const char* scopeName(CoopScope scope) {
  switch (scope) {
    case CoopScope::THREAD:
      return "thread";
    case CoopScope::WARP:
      return "warp";
    case CoopScope::BLOCK:
      return "block";
  }
  return "unknown";
}

std::string formatBytes(size_t bytes) {
  const char* units[] = {"B", "KB", "MB", "GB"};
  int unit_idx = 0;
  double val = static_cast<double>(bytes);
  while (val >= 1024.0 && unit_idx < 3) {
    val /= 1024.0;
    unit_idx++;
  }
  std::ostringstream oss;
  if (unit_idx == 0) {
    oss << bytes << " B";
  } else {
    oss << std::fixed;
    oss.precision(2);
    oss << val << " " << units[unit_idx];
  }
  return oss.str();
}

bool verifyFillPattern(
    const float* data,
    size_t count,
    int expected_rank,
    int expected_iteration,
    size_t* out_first_bad_index,
    float* out_got,
    float* out_expected) {
  float expected_val = fillPatternValue(expected_rank, expected_iteration);
  for (size_t i = 0; i < count; i++) {
    if (std::abs(data[i] - expected_val) > 1e-3f) {
      if (out_first_bad_index) {
        *out_first_bad_index = i;
      }
      if (out_got) {
        *out_got = data[i];
      }
      if (out_expected) {
        *out_expected = expected_val;
      }
      return false;
    }
  }
  return true;
}

} // namespace torchcomms::device::test
