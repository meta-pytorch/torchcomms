// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/benchmarks/BenchmarkRunner.h"

#include "comms/uniflow/logging/Logger.h"

namespace uniflow::benchmark {

void BenchmarkRunner::registerBenchmark(std::unique_ptr<Benchmark> bench) {
  UNIFLOW_LOG_INFO("Registered benchmark: {}", bench->name());
  benchmarks_.push_back(std::move(bench));
}

std::vector<std::string> BenchmarkRunner::listBenchmarks() const {
  std::vector<std::string> names;
  names.reserve(benchmarks_.size());
  for (const auto& b : benchmarks_) {
    names.push_back(b->name());
  }
  return names;
}

std::vector<BenchmarkResult> BenchmarkRunner::runAll(
    const BenchmarkConfig& config,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap) {
  std::vector<BenchmarkResult> allResults;

  for (auto& bench : benchmarks_) {
    UNIFLOW_LOG_INFO("Running benchmark: {}", bench->name());
    auto results = bench->run(config, peers, bootstrap);
    allResults.insert(
        allResults.end(),
        std::make_move_iterator(results.begin()),
        std::make_move_iterator(results.end()));
  }

  return allResults;
}

std::vector<BenchmarkResult> BenchmarkRunner::runByName(
    const std::string& name,
    const BenchmarkConfig& config,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap) {
  for (auto& bench : benchmarks_) {
    if (bench->name() == name) {
      UNIFLOW_LOG_INFO("Running benchmark: {}", bench->name());
      return bench->run(config, peers, bootstrap);
    }
  }
  UNIFLOW_LOG_WARN("Benchmark not found: {}", name);
  return {};
}

std::vector<size_t> generateSizes(size_t minSize, size_t maxSize) {
  std::vector<size_t> sizes;
  for (size_t s = minSize; s <= maxSize;) {
    sizes.push_back(s);
    if (s > maxSize / 2) {
      break; // next multiply would overflow or exceed maxSize
    }
    s *= 2;
  }
  return sizes;
}

} // namespace uniflow::benchmark
