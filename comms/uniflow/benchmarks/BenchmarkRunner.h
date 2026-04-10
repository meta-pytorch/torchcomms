// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "comms/uniflow/benchmarks/BenchmarkResult.h"
#include "comms/uniflow/benchmarks/Bootstrap.h"
#include "comms/uniflow/benchmarks/Rendezvous.h"

namespace uniflow::benchmark {

struct BenchmarkConfig {
  size_t minSize{1};
  size_t maxSize{1UL << 30};
  int iterations{100};
  int warmupIterations{10};
  int loopCount{1};
  bool bidirectional{false};
  std::string direction{"both"};
  std::vector<int> numStreams{1, 2, 4, 8};
};

class Benchmark {
 public:
  virtual ~Benchmark() = default;
  virtual std::string name() const = 0;
  virtual std::vector<BenchmarkResult> run(
      const BenchmarkConfig& config,
      std::vector<PeerConnection>& peers,
      const BootstrapConfig& bootstrap) = 0;
};

/// Registry and runner for benchmarks.
class BenchmarkRunner {
 public:
  void registerBenchmark(std::unique_ptr<Benchmark> bench);
  std::vector<std::string> listBenchmarks() const;

  std::vector<BenchmarkResult> runAll(
      const BenchmarkConfig& config,
      std::vector<PeerConnection>& peers,
      const BootstrapConfig& bootstrap);

  /// Returns empty vector if name not found.
  std::vector<BenchmarkResult> runByName(
      const std::string& name,
      const BenchmarkConfig& config,
      std::vector<PeerConnection>& peers,
      const BootstrapConfig& bootstrap);

 private:
  std::vector<std::unique_ptr<Benchmark>> benchmarks_;
};

/// Generate message sizes as powers of 2 from minSize to maxSize (inclusive).
std::vector<size_t> generateSizes(size_t minSize, size_t maxSize);

} // namespace uniflow::benchmark
