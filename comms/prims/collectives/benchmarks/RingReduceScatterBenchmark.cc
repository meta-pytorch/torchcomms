// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <glog/logging.h>
#include <nccl.h>

#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <vector>

#include "comms/prims/benchmarks/BenchmarkMacros.h"
#include "comms/prims/collectives/RingReduceScatterLauncher.h"
#include "comms/prims/collectives/RingUtils.h"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"
#include "comms/prims/transport/ibrc/MultipeerIbrcTransport.h"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;

namespace comms::prims::benchmark {

namespace {

struct RingReduceScatterBenchmarkConfig {
  std::size_t chunk_elements;
  int num_blocks;
  int num_rings;
  std::size_t data_buffer_size;
  int pipeline_depth;
  int num_qps;
  bool use_ibrc{false};
  std::string name;
};

struct RingReduceScatterBenchmarkResult {
  std::string test_name;
  std::size_t chunk_elements{};
  std::size_t total_bytes{};
  int num_rings{};
  float baseline_bandwidth{};
  float candidate_bandwidth{};
  float baseline_latency{};
  float candidate_latency{};
  float speedup{};
};

class RingReduceScatterBenchmarkFixture
    : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    CUDA_CHECK_VOID(cudaSetDevice(localRank));

    setenv("NCCL_P2P_DISABLE", "1", 1);
    CUDA_CHECK_VOID(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    CUDA_CHECK_VOID(cudaStreamDestroy(stream_));
    BenchmarkTestFixture::TearDown();
  }

  ncclUniqueId get_nccl_id() {
    ncclUniqueId id;
    if (globalRank == 0) {
      ncclResult_t res = ncclGetUniqueId(&id);
      if (res != ncclSuccess) {
        XLOGF(ERR, "ncclGetUniqueId failed: {}", ncclGetErrorString(res));
        std::abort();
      }
    }
    std::vector<ncclUniqueId> all_ids(worldSize);
    all_ids[globalRank] = id;
    auto result =
        bootstrap
            ->allGather(
                all_ids.data(), sizeof(ncclUniqueId), globalRank, worldSize)
            .get();
    if (result != 0) {
      XLOG(ERR) << "Bootstrap allGather for NCCL ID failed";
      std::abort();
    }
    return all_ids[0];
  }

  float run_nccl_benchmark(
      const RingReduceScatterBenchmarkConfig& config,
      float& latency_us) {
    const std::size_t total_elements = config.chunk_elements * worldSize;
    const std::size_t total_bytes = total_elements * sizeof(float);

    DeviceBuffer send_buf(total_bytes);
    DeviceBuffer recv_buf(config.chunk_elements * sizeof(float));
    CUDA_CHECK(cudaMemset(send_buf.get(), 0, total_bytes));
    CUDA_CHECK(
        cudaMemset(recv_buf.get(), 0, config.chunk_elements * sizeof(float)));

    ncclComm_t nccl_comm{};
    NCCL_CHECK(
        ncclCommInitRank(&nccl_comm, worldSize, get_nccl_id(), globalRank));
    auto ncclCommGuard = folly::makeGuard([&] {
      const ncclResult_t res = ncclCommDestroy(nccl_comm);
      if (res != ncclSuccess) {
        XLOGF(ERR, "ncclCommDestroy failed: {}", ncclGetErrorString(res));
      }
    });

    CudaEvent start, stop;
    const int n_warmup = 5;
    const int n_iter = 100;

    bootstrap->barrierAll();
    for (int i = 0; i < n_warmup; i++) {
      NCCL_CHECK(ncclReduceScatter(
          send_buf.get(),
          recv_buf.get(),
          config.chunk_elements,
          ncclFloat,
          ncclSum,
          nccl_comm,
          stream_));
    }

    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < n_iter; i++) {
      NCCL_CHECK(ncclReduceScatter(
          send_buf.get(),
          recv_buf.get(),
          config.chunk_elements,
          ncclFloat,
          ncclSum,
          nccl_comm,
          stream_));
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start.get(), stop.get()));
    float avg_ms = total_ms / n_iter;
    latency_us = avg_ms * 1000.0f;

    float bw = (total_bytes / (1e9f)) / (avg_ms / 1e3f);
    bootstrap->barrierAll();
    return bw;
  }

  float run_ring_benchmark(
      const RingReduceScatterBenchmarkConfig& config,
      float& latency_us) {
    const std::size_t total_elements = config.chunk_elements * worldSize;
    const std::size_t total_bytes = total_elements * sizeof(float);

    DeviceBuffer send_buf(total_bytes);
    DeviceBuffer recv_buf(config.chunk_elements * sizeof(float));
    CUDA_CHECK(cudaMemset(send_buf.get(), 0, total_bytes));
    CUDA_CHECK(
        cudaMemset(recv_buf.get(), 0, config.chunk_elements * sizeof(float)));

    const int maxGroups = config.num_blocks * config.num_rings;
    constexpr int kMaxEagerExchangeQpsPerPeerPerNic = 128;
    MultipeerIbgdaTransportConfig transport_config{
        .cudaDevice = localRank,
        .dataBufferSize = config.data_buffer_size,
        .maxGroups = maxGroups,
        .sendRecv =
            MultipeerIbgdaTransportConfig::SendRecvConfig{
                .pipelineDepth = config.pipeline_depth,
            },
        .qpsPerBlockPerNic = config.num_qps,
        .ibLazyConnect =
            maxGroups * config.num_qps > kMaxEagerExchangeQpsPerPeerPerNic,
    };

    if (config.use_ibrc) {
      MultipeerIbrcTransport transport(
          globalRank, worldSize, bootstrap, transport_config);
      return run_ring_benchmark_with_transport(
          config, send_buf, recv_buf, total_bytes, latency_us, transport);
    }
    MultipeerIbgdaTransport transport(
        globalRank, worldSize, bootstrap, transport_config);
    return run_ring_benchmark_with_transport(
        config, send_buf, recv_buf, total_bytes, latency_us, transport);
  }

  template <typename Transport>
  float run_ring_benchmark_with_transport(
      const RingReduceScatterBenchmarkConfig& config,
      DeviceBuffer& send_buf,
      DeviceBuffer& recv_buf,
      std::size_t total_bytes,
      float& latency_us,
      Transport& transport) {
    transport.exchange();

    auto rings_opt =
        make_standard_rings(worldSize, globalRank, config.num_rings);
    if (!rings_opt) {
      XLOGF(
          ERR,
          "Cannot construct {} distinct rings for {} ranks",
          config.num_rings,
          worldSize);
      latency_us = 0.0f;
      return 0.0f;
    }
    auto& rings = *rings_opt;

    RingReduceScatterLaunchParams launch_params{};
    launch_params.my_rank = globalRank;
    launch_params.num_ranks = worldSize;
    launch_params.chunk_elements = config.chunk_elements;
    launch_params.input = static_cast<const float*>(send_buf.get());
    launch_params.output = static_cast<float*>(recv_buf.get());
    launch_params.num_blocks = config.num_blocks * config.num_rings;
    launch_params.num_rings = config.num_rings;

    for (int r = 0; r < config.num_rings; r++) {
      auto& rp = launch_params.rings[r];
      rp.prev_rank = rings[r].prev_rank;
      rp.next_rank = rings[r].next_rank;
      transport.queuePeerForMaterialization(rp.prev_rank);
      transport.queuePeerForMaterialization(rp.next_rank);
    }
    transport.connectPeers();
    for (int r = 0; r < config.num_rings; r++) {
      auto& rp = launch_params.rings[r];
      rp.prev =
          P2pIbTransportDevice(transport.getP2pTransportDevice(rp.prev_rank));
      rp.next =
          P2pIbTransportDevice(transport.getP2pTransportDevice(rp.next_rank));
    }

    CudaEvent start, stop;
    const int n_warmup = 5;
    const int n_iter = 100;

    bootstrap->barrierAll();
    for (int i = 0; i < n_warmup; i++) {
      launch_ring_reduce_scatter(launch_params);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    bootstrap->barrierAll();

    CUDA_CHECK(cudaEventRecord(start.get()));
    for (int i = 0; i < n_iter; i++) {
      launch_ring_reduce_scatter(launch_params);
    }
    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start.get(), stop.get()));
    float avg_ms = total_ms / n_iter;
    latency_us = avg_ms * 1000.0f;

    float bw = (total_bytes / (1e9f)) / (avg_ms / 1e3f);
    bootstrap->barrierAll();
    return bw;
  }

  void print_results(
      const std::vector<RingReduceScatterBenchmarkResult>& results,
      const char* baselineLabel,
      const char* candidateLabel) {
    if (globalRank != 0) {
      return;
    }

    auto fmt_bytes = [](std::size_t bytes) -> std::string {
      if (bytes >= 1024UL * 1024 * 1024) {
        return std::to_string(bytes / (1024UL * 1024 * 1024)) + "GB";
      }
      if (bytes >= 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + "MB";
      }
      if (bytes >= 1024) {
        return std::to_string(bytes / 1024) + "KB";
      }
      return std::to_string(bytes) + "B";
    };

    auto fmt_float = [](float value, int precision) -> std::string {
      if (value <= 0.0f) {
        return "n/a";
      }
      std::stringstream value_ss;
      value_ss << std::fixed << std::setprecision(precision) << value;
      return value_ss.str();
    };

    std::stringstream ss;
    ss << "\n";
    ss << "================================================================================\n";
    ss << "             Ring ReduceScatter Benchmark (" << candidateLabel
       << " vs " << baselineLabel << ")\n";
    ss << "================================================================================\n";
    ss << std::left << std::setw(14) << "Test" << std::right << std::setw(10)
       << "Size" << std::right << std::setw(6) << "Rings" << std::right
       << std::setw(10) << baselineLabel << std::right << std::setw(10)
       << candidateLabel << std::right << std::setw(10) << "Speedup"
       << std::right << std::setw(12) << "Base Lat" << std::right
       << std::setw(12) << "Cand Lat\n";
    ss << std::left << std::setw(14) << "" << std::right << std::setw(10) << ""
       << std::right << std::setw(6) << "" << std::right << std::setw(10)
       << "(GB/s)" << std::right << std::setw(10) << "(GB/s)" << std::right
       << std::setw(10) << "" << std::right << std::setw(12) << "(us)"
       << std::right << std::setw(12) << "(us)\n";
    ss << "--------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      ss << std::left << std::setw(14) << r.test_name << std::right
         << std::setw(10) << fmt_bytes(r.total_bytes) << std::right
         << std::setw(6) << r.num_rings << std::right << std::setw(10)
         << fmt_float(r.baseline_bandwidth, 2) << std::right << std::setw(10)
         << fmt_float(r.candidate_bandwidth, 2) << std::right << std::setw(10)
         << (r.speedup > 0.0f ? fmt_float(r.speedup, 2) + "x" : "n/a")
         << std::right << std::setw(12) << fmt_float(r.baseline_latency, 1)
         << std::right << std::setw(12) << fmt_float(r.candidate_latency, 1)
         << "\n";
    }

    ss << "================================================================================\n";
    ss << worldSize << " ranks, chunk_elements = per-rank output elements\n";
    ss << "================================================================================\n";

    XLOG(INFO) << ss.str();
  }

  std::vector<RingReduceScatterBenchmarkConfig> make_benchmark_configs(
      bool useIbrc) {
    // Matched config for a fair IBRC-vs-IBGDA comparison: same QP count and
    // staging buffer size for both backends.
    std::size_t kDataBufferSize = 32UL * 1024 * 1024;
    int kNumQps = 4;

    auto configs = std::vector<RingReduceScatterBenchmarkConfig>{
        {.chunk_elements = 64 * 1024,
         .num_blocks = 8,
         .num_rings = 1,
         .data_buffer_size = kDataBufferSize,
         .pipeline_depth = 2,
         .num_qps = kNumQps,
         .name = "256K_8B"},
        {.chunk_elements = 256 * 1024,
         .num_blocks = 16,
         .num_rings = 1,
         .data_buffer_size = kDataBufferSize,
         .pipeline_depth = 2,
         .num_qps = kNumQps,
         .name = "1M_16B"},
        {.chunk_elements = 1024 * 1024,
         .num_blocks = 16,
         .num_rings = 1,
         .data_buffer_size = kDataBufferSize,
         .pipeline_depth = 2,
         .num_qps = kNumQps,
         .name = "4M_16B"},
        {.chunk_elements = 4 * 1024 * 1024,
         .num_blocks = 16,
         .num_rings = 1,
         .data_buffer_size = kDataBufferSize,
         .pipeline_depth = 2,
         .num_qps = kNumQps,
         .name = "16M_16B"},
        {.chunk_elements = 16 * 1024 * 1024,
         .num_blocks = 16,
         .num_rings = 1,
         .data_buffer_size = kDataBufferSize,
         .pipeline_depth = 2,
         .num_qps = kNumQps,
         .name = "64M_16B"},
        {.chunk_elements = 32 * 1024 * 1024,
         .num_blocks = 16,
         .num_rings = 1,
         .data_buffer_size = kDataBufferSize,
         .pipeline_depth = 2,
         .num_qps = kNumQps,
         .name = "128M_16B"},
        {.chunk_elements = 64 * 1024 * 1024,
         .num_blocks = 32,
         .num_rings = 1,
         .data_buffer_size = kDataBufferSize,
         .pipeline_depth = 2,
         .num_qps = kNumQps,
         .name = "256M_32B"},
        {.chunk_elements = 1024 * 1024,
         .num_blocks = 16,
         .num_rings = 2,
         .data_buffer_size = kDataBufferSize,
         .pipeline_depth = 2,
         .num_qps = kNumQps,
         .name = "4M_16B_2R"},
        {.chunk_elements = 4 * 1024 * 1024,
         .num_blocks = 16,
         .num_rings = 2,
         .data_buffer_size = kDataBufferSize,
         .pipeline_depth = 2,
         .num_qps = kNumQps,
         .name = "16M_16B_2R"},
        {.chunk_elements = 16 * 1024 * 1024,
         .num_blocks = 16,
         .num_rings = 2,
         .data_buffer_size = kDataBufferSize,
         .pipeline_depth = 2,
         .num_qps = kNumQps,
         .name = "64M_16B_2R"},
        {.chunk_elements = 64 * 1024 * 1024,
         .num_blocks = 32,
         .num_rings = 2,
         .data_buffer_size = kDataBufferSize,
         .pipeline_depth = 2,
         .num_qps = kNumQps,
         .name = "256M_32B_2R"},
    };
    for (auto& config : configs) {
      config.use_ibrc = useIbrc;
    }
    return configs;
  }

  RingReduceScatterBenchmarkResult make_benchmark_result(
      const RingReduceScatterBenchmarkConfig& config,
      float baselineBw,
      float baselineLatencyUs,
      float candidateBw,
      float candidateLatencyUs) {
    return {
        .test_name = config.name,
        .chunk_elements = config.chunk_elements,
        .total_bytes = config.chunk_elements *
            static_cast<std::size_t>(worldSize) * sizeof(float),
        .num_rings = config.num_rings,
        .baseline_bandwidth = baselineBw,
        .candidate_bandwidth = candidateBw,
        .baseline_latency = baselineLatencyUs,
        .candidate_latency = candidateLatencyUs,
        .speedup = (baselineBw > 0) ? candidateBw / baselineBw : 0.0f,
    };
  }

  void run_ibgda_vs_nccl_benchmark_suite() {
    if (globalRank == 0) {
      XLOG(INFO) << "\n=== Ring ReduceScatter (IBGDA) vs NCCL Comparison ===\n";
    }

    std::vector<RingReduceScatterBenchmarkResult> results;
    for (const auto& config : make_benchmark_configs(false)) {
      float ncclLatencyUs = 0.0f;
      float ncclBw = run_nccl_benchmark(config, ncclLatencyUs);
      float ringLatencyUs = 0.0f;
      float ringBw = run_ring_benchmark(config, ringLatencyUs);
      if (globalRank == 0) {
        results.push_back(make_benchmark_result(
            config, ncclBw, ncclLatencyUs, ringBw, ringLatencyUs));
      }
      bootstrap->barrierAll();
    }

    print_results(results, "NCCL", "IBGDA");
  }

  void run_ibrc_vs_ibgda_benchmark_suite() {
    if (globalRank == 0) {
      XLOG(INFO) << "\n=== Ring ReduceScatter (IBRC) vs IBGDA Comparison ===\n";
    }

    std::vector<RingReduceScatterBenchmarkResult> results;
    auto ibgdaConfigs = make_benchmark_configs(false);
    auto ibrcConfigs = make_benchmark_configs(true);
    CHECK_EQ(ibgdaConfigs.size(), ibrcConfigs.size());
    for (std::size_t i = 0; i < ibrcConfigs.size(); ++i) {
      const auto& ibgdaConfig = ibgdaConfigs.at(i);
      const auto& ibrcConfig = ibrcConfigs.at(i);

      float ibgdaLatencyUs = 0.0f;
      float ibgdaBw = run_ring_benchmark(ibgdaConfig, ibgdaLatencyUs);

      float ibrcLatencyUs = 0.0f;
      float ibrcBw = run_ring_benchmark(ibrcConfig, ibrcLatencyUs);

      if (globalRank == 0) {
        results.push_back(make_benchmark_result(
            ibrcConfig, ibgdaBw, ibgdaLatencyUs, ibrcBw, ibrcLatencyUs));
      }
      bootstrap->barrierAll();
    }

    print_results(results, "IBGDA", "IBRC");
  }

  cudaStream_t stream_{};
};

TEST_F(RingReduceScatterBenchmarkFixture, IbgdaVsNccl) {
  run_ibgda_vs_nccl_benchmark_suite();
}

TEST_F(RingReduceScatterBenchmarkFixture, IbrcVsIbgda) {
  run_ibrc_vs_ibgda_benchmark_suite();
}

} // namespace

} // namespace comms::prims::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::BenchmarkEnvironment());
  return RUN_ALL_TESTS();
}
