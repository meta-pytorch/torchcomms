// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>

#include <iomanip>
#include <sstream>
#include <vector>

#include "comms/utils/cvars/nccl_cvars.h"

#include "comms/pipes/MultipeerIbgdaTransport.h"
#include "comms/pipes/benchmarks/BenchmarkMacros.h"
#include "comms/pipes/collectives/RingAllReduceLauncher.h"
#include "comms/pipes/collectives/RingUtils.h"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

namespace {

struct RingAllReduceBenchmarkConfig {
  std::size_t total_elements;
  int num_blocks;
  int num_rings;
  std::size_t data_buffer_size;
  int pipeline_depth;
  int num_qps;
  bool enable_bidir_ag{false};
  bool use_reverse_ring{false};
  std::size_t signaling_data_size{0};
  std::size_t ib_window_bytes{0};
  bool skip_reduction{false};
  std::string name;
};

struct RingAllReduceBenchmarkResult {
  std::string test_name;
  std::size_t total_elements{};
  std::size_t total_bytes{};
  int num_rings{};
  float nccl_bandwidth{};
  float nccl_ring_bandwidth{};
  float ctran_bandwidth{};
  float ring_bandwidth{};
  float nccl_latency{};
  float nccl_ring_latency{};
  float ctran_latency{};
  float ring_latency{};
  float speedup_vs_nccl{};
  float speedup_vs_nccl_ring{};
  float speedup_vs_ctran{};
};

class RingAllReduceBenchmarkFixture : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    CUDA_CHECK_VOID(cudaSetDevice(localRank));

    setenv("NCCL_P2P_DISABLE", "1", 1);
    setenv("NCCL_CTRAN_ENABLE", "1", 1);
    setenv("NCCL_COMM_STATE_DEBUG_TOPO", "nolocal", 1);
    setenv("NCCL_IGNORE_TOPO_LOAD_FAILURE", "true", 1);
    NCCL_CHECK_VOID(
        ncclCommInitRank(&comm_, worldSize, get_nccl_id(), globalRank));

    setenv("NCCL_ALGO", "Ring", 1);
    setenv("NCCL_PROTO", "Simple", 1);
    NCCL_CHECK_VOID(
        ncclCommInitRank(&ring_comm_, worldSize, get_nccl_id(), globalRank));
    unsetenv("NCCL_ALGO");
    unsetenv("NCCL_PROTO");

    CUDA_CHECK_VOID(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    NCCL_CHECK_VOID(ncclCommDestroy(comm_));
    NCCL_CHECK_VOID(ncclCommDestroy(ring_comm_));
    CUDA_CHECK_VOID(cudaStreamDestroy(stream_));
    BenchmarkTestFixture::TearDown();
  }

  ncclUniqueId get_nccl_id() {
    ncclUniqueId id{};
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
      const RingAllReduceBenchmarkConfig& config,
      float& latency_us) {
    NCCL_CHECK(set_allreduce_algo("orig"));
    return run_ncclx_benchmark(config, latency_us, comm_);
  }

  float run_nccl_ring_benchmark(
      const RingAllReduceBenchmarkConfig& config,
      float& latency_us) {
    return run_ncclx_benchmark(config, latency_us, ring_comm_);
  }

  float run_ctran_benchmark(
      const RingAllReduceBenchmarkConfig& config,
      float& latency_us) {
    NCCL_CHECK(set_allreduce_algo("ctring"));
    float bw = run_ncclx_benchmark(config, latency_us, comm_);
    NCCL_CHECK(set_allreduce_algo("orig"));
    return bw;
  }

  ncclResult_t set_allreduce_algo(const std::string& algo) {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclx::Hints hints({{"allreduceAlgo", algo}});
    config.hints = &hints;
    return ncclx::commSetConfig(comm_, &config);
  }

  float run_ncclx_benchmark(
      const RingAllReduceBenchmarkConfig& config,
      float& latency_us,
      ncclComm_t nccl_comm = nullptr) {
    if (nccl_comm == nullptr) {
      nccl_comm = comm_;
    }
    const std::size_t total_bytes = config.total_elements * sizeof(float);

    DeviceBuffer send_buf(total_bytes);
    DeviceBuffer recv_buf(total_bytes);
    CUDA_CHECK(cudaMemset(send_buf.get(), 0, total_bytes));
    CUDA_CHECK(cudaMemset(recv_buf.get(), 0, total_bytes));

    CudaEvent start, stop;
    const int n_warmup = 5;
    const int n_iter = 100;

    bootstrap->barrierAll();
    for (int i = 0; i < n_warmup; i++) {
      NCCL_CHECK(ncclAllReduce(
          send_buf.get(),
          recv_buf.get(),
          config.total_elements,
          ncclFloat,
          ncclSum,
          nccl_comm,
          stream_));
    }

    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < n_iter; i++) {
      NCCL_CHECK(ncclAllReduce(
          send_buf.get(),
          recv_buf.get(),
          config.total_elements,
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
      const RingAllReduceBenchmarkConfig& config,
      float& latency_us) {
    const std::size_t total_bytes = config.total_elements * sizeof(float);

    DeviceBuffer send_buf(total_bytes);
    DeviceBuffer recv_buf(total_bytes);
    CUDA_CHECK(cudaMemset(send_buf.get(), 0, total_bytes));
    CUDA_CHECK(cudaMemset(recv_buf.get(), 0, total_bytes));

    std::unique_ptr<MultipeerIbgdaTransport> transport;
    try {
      MultipeerIbgdaTransportConfig transport_config{
          .cudaDevice = localRank,
          .dataBufferSize = config.data_buffer_size,
          .sendRecv =
              MultipeerIbgdaTransportConfig::SendRecvConfig{
                  .maxGroups = config.num_blocks * config.num_rings,
                  .pipelineDepth = config.pipeline_depth,
              },
          .numQpsPerPeerPerNic = config.num_qps,
      };
      transport = std::make_unique<MultipeerIbgdaTransport>(
          globalRank, worldSize, bootstrap, transport_config);
      transport->exchange();
    } catch (const std::exception& e) {
      XLOGF(ERR, "IBGDA transport not available: {}", e.what());
      latency_us = 0.0f;
      return 0.0f;
    }

    if (config.use_reverse_ring && config.num_rings != 2) {
      XLOGF(
          ERR,
          "use_reverse_ring requires num_rings=2, got {}",
          config.num_rings);
      latency_us = 0.0f;
      return 0.0f;
    }

    std::optional<std::vector<RingNeighbors>> rings_opt;
    if (config.use_reverse_ring) {
      rings_opt = make_bidir_rings(worldSize, globalRank);
    } else {
      rings_opt = make_standard_rings(worldSize, globalRank, config.num_rings);
    }
    if (!rings_opt) {
      XLOGF(
          ERR,
          "Cannot construct {} rings for {} ranks",
          config.num_rings,
          worldSize);
      latency_us = 0.0f;
      return 0.0f;
    }
    auto& rings = *rings_opt;
    if (rings.size() < static_cast<std::size_t>(config.num_rings)) {
      XLOGF(
          ERR,
          "Constructed {} rings, expected {}",
          rings.size(),
          config.num_rings);
      latency_us = 0.0f;
      return 0.0f;
    }

    RingAllReduceLaunchParams launch_params{};
    launch_params.my_rank = globalRank;
    launch_params.num_ranks = worldSize;
    launch_params.count = config.total_elements;
    launch_params.input = static_cast<const float*>(send_buf.get());
    launch_params.output = static_cast<float*>(recv_buf.get());
    launch_params.num_blocks = config.num_blocks * config.num_rings;
    launch_params.num_rings = config.num_rings;
    launch_params.stream = stream_;
    launch_params.enable_bidir_ag = config.enable_bidir_ag;
    launch_params.signaling_data_size = config.signaling_data_size;
    launch_params.ib_window_bytes = config.ib_window_bytes;
    launch_params.skip_reduction = config.skip_reduction;

    for (int r = 0; r < config.num_rings; r++) {
      auto& rp = launch_params.rings[r];
      rp.prev_rank = rings[r].prev_rank;
      rp.next_rank = rings[r].next_rank;
      rp.prev = transport->getP2pTransportDevice(rings[r].prev_rank);
      rp.next = transport->getP2pTransportDevice(rings[r].next_rank);
    }

    CudaEvent start, stop;
    const int n_warmup = 5;
    const int n_iter = 100;

    bootstrap->barrierAll();
    for (int i = 0; i < n_warmup; i++) {
      launch_ring_allreduce(launch_params);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    bootstrap->barrierAll();

    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < n_iter; i++) {
      launch_ring_allreduce(launch_params);
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

  void print_results(const std::vector<RingAllReduceBenchmarkResult>& results) {
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

    std::stringstream ss;
    ss << "\n";
    ss << "============================================================================================================================\n";
    ss << "         NCCL (auto) vs NCCL (Ring) vs Ctran (ctring) vs Pipes Ring AllReduce Benchmark\n";
    ss << "============================================================================================================================\n";
    ss << std::left << std::setw(16) << "Test" << std::right << std::setw(8)
       << "Size" << std::setw(4) << "R" << std::setw(9) << "NCCL"
       << std::setw(10) << "NCCLRing" << std::setw(9) << "Ctran" << std::setw(9)
       << "Pipes" << std::setw(10) << "vs NCCL" << std::setw(10) << "vs Ring"
       << std::setw(10) << "vs Ctran" << "\n";
    ss << std::left << std::setw(16) << "" << std::right << std::setw(8) << ""
       << std::setw(4) << "" << std::setw(9) << "(GB/s)" << std::setw(10)
       << "(GB/s)" << std::setw(9) << "(GB/s)" << std::setw(9) << "(GB/s)"
       << "\n";
    ss << "----------------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      ss << std::left << std::setw(16) << r.test_name << std::right
         << std::setw(8) << fmt_bytes(r.total_bytes) << std::setw(4)
         << r.num_rings << std::setw(9) << std::fixed << std::setprecision(2)
         << r.nccl_bandwidth << std::setw(10) << std::fixed
         << std::setprecision(2) << r.nccl_ring_bandwidth << std::setw(9)
         << std::fixed << std::setprecision(2) << r.ctran_bandwidth
         << std::setw(9) << std::fixed << std::setprecision(2)
         << r.ring_bandwidth << std::setw(9) << std::fixed
         << std::setprecision(2) << r.speedup_vs_nccl << "x" << std::setw(9)
         << std::fixed << std::setprecision(2) << r.speedup_vs_nccl_ring << "x"
         << std::setw(9) << std::fixed << std::setprecision(2)
         << r.speedup_vs_ctran << "x\n";
    }

    ss << "============================================================================================================================\n";
    ss << worldSize
       << " ranks | NCCL=orig(auto) | NCCLRing=Ring/Simple | Ctran=ctring(GPE) | Pipes=IBGDA(device-native)\n";
    ss << "============================================================================================================================\n";

    XLOG(INFO) << ss.str();
  }

  ncclComm_t comm_{};
  ncclComm_t ring_comm_{};
  cudaStream_t stream_{};
};

TEST_F(RingAllReduceBenchmarkFixture, VsNccl) {
  if (globalRank == 0) {
    XLOG(INFO) << "\n=== Ring AllReduce (Pipes IBGDA) vs NCCL Comparison ===\n";
  }

  std::size_t kDataBufferSize = 32UL * 1024 * 1024;
  int kNumQps = 4;

  std::vector<RingAllReduceBenchmarkConfig> configs = {
      // === Medium-size optimization: block count sweep at 4MB/16MB/32MB ===
      // 4MB baselines
      {.total_elements = 1024 * 1024,
       .num_blocks = 16,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "4MB_16B"},
      {.total_elements = 1024 * 1024,
       .num_blocks = 8,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "4MB_8B"},
      {.total_elements = 1024 * 1024,
       .num_blocks = 4,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "4MB_4B"},
      {.total_elements = 1024 * 1024,
       .num_blocks = 2,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "4MB_2B"},
      // 16MB baselines
      {.total_elements = 4 * 1024 * 1024,
       .num_blocks = 16,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "16MB_16B"},
      {.total_elements = 4 * 1024 * 1024,
       .num_blocks = 8,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "16MB_8B"},
      {.total_elements = 4 * 1024 * 1024,
       .num_blocks = 4,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "16MB_4B"},
      {.total_elements = 4 * 1024 * 1024,
       .num_blocks = 2,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "16MB_2B"},
      // 32MB baselines
      {.total_elements = 8 * 1024 * 1024,
       .num_blocks = 16,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "32MB_16B"},
      {.total_elements = 8 * 1024 * 1024,
       .num_blocks = 8,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "32MB_8B"},
      {.total_elements = 8 * 1024 * 1024,
       .num_blocks = 4,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "32MB_4B"},
      // 64MB + 512MB reference points
      {.total_elements = 16 * 1024 * 1024,
       .num_blocks = 16,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "64MB_16B"},
      {.total_elements = 128 * 1024 * 1024,
       .num_blocks = 16,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "512MB_16B"},
  };

  std::vector<RingAllReduceBenchmarkResult> results;

  for (const auto& config : configs) {
    float nccl_lat = 0.0f;
    float nccl_bw = run_nccl_benchmark(config, nccl_lat);

    float nccl_ring_lat = 0.0f;
    float nccl_ring_bw = run_nccl_ring_benchmark(config, nccl_ring_lat);

    float ctran_lat = 0.0f;
    float ctran_bw = run_ctran_benchmark(config, ctran_lat);

    float ring_lat = 0.0f;
    float ring_bw = run_ring_benchmark(config, ring_lat);

    if (globalRank == 0) {
      results.push_back({
          .test_name = config.name,
          .total_elements = config.total_elements,
          .total_bytes = config.total_elements * sizeof(float),
          .num_rings = config.num_rings,
          .nccl_bandwidth = nccl_bw,
          .nccl_ring_bandwidth = nccl_ring_bw,
          .ctran_bandwidth = ctran_bw,
          .ring_bandwidth = ring_bw,
          .nccl_latency = nccl_lat,
          .nccl_ring_latency = nccl_ring_lat,
          .ctran_latency = ctran_lat,
          .ring_latency = ring_lat,
          .speedup_vs_nccl = (nccl_bw > 0) ? ring_bw / nccl_bw : 0.0f,
          .speedup_vs_nccl_ring =
              (nccl_ring_bw > 0) ? ring_bw / nccl_ring_bw : 0.0f,
          .speedup_vs_ctran = (ctran_bw > 0) ? ring_bw / ctran_bw : 0.0f,
      });
    }
    bootstrap->barrierAll();
  }

  print_results(results);
}

} // namespace

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::BenchmarkEnvironment());
  return RUN_ALL_TESTS();
}
