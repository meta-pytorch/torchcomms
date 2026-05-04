// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/benchmarks/bench/ConnectionSetupBenchmark.h"

#include <chrono>

#include "comms/uniflow/benchmarks/Rendezvous.h"
#include "comms/uniflow/benchmarks/Stats.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/logging/Logger.h"
#include "comms/uniflow/transport/nvlink/NVLinkTransport.h"

namespace uniflow::benchmark {

std::vector<BenchmarkResult> ConnectionSetupBenchmark::run(
    const BenchmarkConfig& config,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap) {
  if (peers.empty()) {
    UNIFLOW_LOG_WARN("ConnectionSetupBenchmark: no peers, skipping");
    return {};
  }

  CudaApi cudaApi;
  auto setDevStatus = cudaApi.setDevice(bootstrap.localRank);
  if (!setDevStatus) {
    UNIFLOW_LOG_ERROR(
        "ConnectionSetupBenchmark: cudaSetDevice({}) failed: {}",
        bootstrap.localRank,
        setDevStatus.error().toString());
    return {};
  }

  const int totalIterations = config.warmupIterations + config.iterations;
  std::vector<double> latenciesUs;
  latenciesUs.reserve(config.iterations);

  for (int iter = 0; iter < totalIterations; ++iter) {
    auto barrierStatus = barrier(peers, bootstrap);
    if (!barrierStatus) {
      UNIFLOW_LOG_ERROR(
          "ConnectionSetupBenchmark: barrier failed: {}",
          barrierStatus.error().toString());
      return {};
    }

    auto start = std::chrono::steady_clock::now();

    ScopedEventBaseThread evbThread("bench-evb");
    NVLinkTransportFactory factory(
        bootstrap.localRank, evbThread.getEventBase());

    auto localTopology = factory.getTopology();
    auto remoteTopologyResult =
        exchangeMetadata(*peers[0].ctrl, localTopology, bootstrap.isRank0());
    if (!remoteTopologyResult) {
      UNIFLOW_LOG_ERROR(
          "ConnectionSetupBenchmark: topology exchange failed: {}",
          remoteTopologyResult.error().toString());
      return {};
    }
    auto remoteTopology = std::move(remoteTopologyResult).value();

    auto transportResult = factory.createTransport(remoteTopology);
    if (!transportResult) {
      UNIFLOW_LOG_ERROR(
          "ConnectionSetupBenchmark: createTransport failed: {}",
          transportResult.error().toString());
      return {};
    }
    auto transport = std::move(transportResult).value();

    auto localInfo = transport->bind();
    auto remoteInfoResult =
        exchangeMetadata(*peers[0].ctrl, localInfo, bootstrap.isRank0());
    if (!remoteInfoResult) {
      UNIFLOW_LOG_ERROR(
          "ConnectionSetupBenchmark: transport info exchange failed: {}",
          remoteInfoResult.error().toString());
      return {};
    }
    auto remoteInfo = std::move(remoteInfoResult).value();

    auto connectStatus = transport->connect(remoteInfo);
    if (!connectStatus) {
      UNIFLOW_LOG_ERROR(
          "ConnectionSetupBenchmark: connect failed: {}",
          connectStatus.error().toString());
      return {};
    }

    auto end = std::chrono::steady_clock::now();
    double elapsedUs =
        std::chrono::duration<double, std::micro>(end - start).count();

    transport->shutdown();

    // Skip warmup iterations.
    if (iter >= config.warmupIterations) {
      latenciesUs.push_back(elapsedUs);
    }

    UNIFLOW_LOG_DEBUG(
        "ConnectionSetupBenchmark: iter {} elapsed {:.1f} us{}",
        iter,
        elapsedUs,
        iter < config.warmupIterations ? " (warmup)" : "");
  }

  auto stats = Stats::compute(std::move(latenciesUs));

  BenchmarkResult result;
  result.benchmarkName = name();
  result.transport = "nvlink";
  result.direction = "n/a";
  result.messageSize = 0;
  result.iterations = config.iterations;
  result.latency = stats;

  UNIFLOW_LOG_INFO(
      "ConnectionSetupBenchmark: avg={:.1f}us p50={:.1f}us p99={:.1f}us",
      stats.avg,
      stats.p50,
      stats.p99);

  return {result};
}

} // namespace uniflow::benchmark
