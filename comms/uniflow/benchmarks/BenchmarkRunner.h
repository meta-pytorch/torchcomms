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
  int batchSize{1};
  int txDepth{1};
  int numNics{0}; // 0 = use all topology-selected NICs
  size_t chunkSize{512 * 1024};
  int cudaDevice{-1};
  /*
   * Multiple GPU device indices for single-process multi-GPU runs. When
   * non-empty, the benchmark drives one transport per device concurrently and
   * reports aggregate bandwidth. Empty falls back to the single cudaDevice.
   */
  std::vector<int> cudaDevices;
  /*
   * Optional explicit NIC assignment per GPU (one inner list per cudaDevices
   * entry). When set, each GPU uses its own NICs instead of topology selection,
   * which avoids NICs being double-booked across adjacent GPUs.
   */
  std::vector<std::vector<std::string>> gpuNicGroups;
  bool bidirectional{false};
  bool dataDirect{false}; // Register GPU memory over the mlx5 Data Direct path.
  std::string direction{"both"};
  std::vector<int> numStreams{1, 2, 4, 8};
  std::string topology{"fanout"}; // "fanout", "fanin"
  int pipelineDepth{2}; // send/recv staging pipeline depth
  size_t slabSize{0}; // 0 = use chunk-size
  int slabNum{0}; // 0 = use pipeline-depth
  /*
   * Run a correctness sweep before the timed loop. A bandwidth number proves
   * nothing about the bytes: a transport that moved garbage, or nothing, still
   * reports excellent throughput.
   */
  bool verify{true};
  /*
   * Cross-rank measurement barrier. A multi-GPU run launches N independent
   * rank-pairs; nothing lines their timed loops up, so summing their
   * bandwidths sums windows that do not coincide. When barrierDir is set all N
   * rendezvous after warmup, immediately before the clock starts.
   */
  std::string barrierDir;
  int barrierRanks{0};
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

class BenchmarkRunner {
 public:
  void registerBenchmark(std::unique_ptr<Benchmark> bench);
  std::vector<std::string> listBenchmarks() const;

  std::vector<BenchmarkResult> runAll(
      const BenchmarkConfig& config,
      std::vector<PeerConnection>& peers,
      const BootstrapConfig& bootstrap);

  std::vector<BenchmarkResult> runByName(
      const std::string& name,
      const BenchmarkConfig& config,
      std::vector<PeerConnection>& peers,
      const BootstrapConfig& bootstrap);

 private:
  std::vector<std::unique_ptr<Benchmark>> benchmarks_;
};

std::vector<size_t> generateSizes(size_t minSize, size_t maxSize);

} // namespace uniflow::benchmark
