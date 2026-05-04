// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/benchmarks/Stats.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace uniflow::benchmark {

Stats Stats::compute(std::vector<double> samples) {
  if (samples.empty()) {
    return {};
  }

  std::sort(samples.begin(), samples.end());

  size_t n = samples.size();
  double sum = std::accumulate(samples.begin(), samples.end(), 0.0);

  Stats s;
  s.min = samples.front();
  s.max = samples.back();
  s.avg = sum / static_cast<double>(n);
  s.p50 = samples[n / 2];

  size_t p99Idx = std::min(
      static_cast<size_t>(std::ceil(static_cast<double>(n) * 0.99)) - 1, n - 1);
  s.p99 = samples[p99Idx];

  return s;
}

} // namespace uniflow::benchmark
