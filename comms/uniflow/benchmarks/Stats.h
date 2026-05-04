// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <vector>

namespace uniflow::benchmark {

struct Stats {
  double min{0};
  double max{0};
  double avg{0};
  double p50{0};
  double p99{0};

  /// Compute statistics from a vector of samples.
  /// Returns zero-initialized Stats if samples is empty.
  static Stats compute(std::vector<double> samples);
};

} // namespace uniflow::benchmark
