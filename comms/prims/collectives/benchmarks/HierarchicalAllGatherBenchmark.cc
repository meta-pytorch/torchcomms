// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#include "comms/prims/benchmarks/BenchmarkMacros.h"
#include "comms/prims/bootstrap/NvlBootstrapAdapter.h"
#include "comms/prims/collectives/AllGatherLauncher.h"
#include "comms/prims/collectives/RingUtils.h"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"
#include "comms/prims/transport/nvl/MultiPeerNvlTransport.h"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;

namespace comms::prims::benchmark {

namespace {

constexpr int kBenchmarkNumBlocks = 8;
constexpr int kBenchmarkIbLinks = 4;
constexpr std::size_t kDataBufferSize = 32UL * 1024 * 1024;
constexpr int kPipelineDepth = 2;
constexpr std::size_t kDefaultIbSignalBytes = 1024 * 1024;

std::string format_size(std::size_t bytes) {
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
}

std::vector<std::size_t> benchmark_sizes() {
  return {
      8 * 1024,
      16 * 1024,
      32 * 1024,
      64 * 1024,
      128 * 1024,
      256 * 1024,
      512 * 1024,
      1024 * 1024,
      2 * 1024 * 1024,
      4 * 1024 * 1024,
      8 * 1024 * 1024,
      16 * 1024 * 1024,
      32 * 1024 * 1024,
      64 * 1024 * 1024,
      128 * 1024 * 1024,
      256 * 1024 * 1024,
      512UL * 1024 * 1024,
      1024UL * 1024 * 1024,
  };
}

std::vector<std::size_t> configured_benchmark_sizes() {
  const char* only_bytes = std::getenv("HIER_AG_BENCH_ONLY_BYTES");
  if (only_bytes != nullptr) {
    return {std::strtoull(only_bytes, nullptr, 10)};
  }
  return benchmark_sizes();
}

int benchmark_num_blocks() {
  const char* num_blocks = std::getenv("HIER_AG_BENCH_NUM_BLOCKS");
  if (num_blocks != nullptr) {
    return static_cast<int>(std::strtol(num_blocks, nullptr, 10));
  }
  return kBenchmarkNumBlocks;
}

int benchmark_nvl_size(int world_size) {
  const char* nvl_size = std::getenv("HIER_AG_BENCH_NVL_SIZE");
  if (nvl_size != nullptr) {
    return static_cast<int>(std::strtol(nvl_size, nullptr, 10));
  }
  return world_size % 4 == 0 ? 4 : world_size;
}

int benchmark_ib_links() {
  const char* ib_links = std::getenv("HIER_AG_BENCH_IB_LINKS");
  if (ib_links != nullptr) {
    return static_cast<int>(std::strtol(ib_links, nullptr, 10));
  }
  return kBenchmarkIbLinks;
}

std::size_t benchmark_data_buffer_size() {
  const char* data_buffer_size = std::getenv("HIER_AG_BENCH_DATA_BUFFER_SIZE");
  if (data_buffer_size != nullptr) {
    return std::strtoull(data_buffer_size, nullptr, 10);
  }
  return kDataBufferSize;
}

int benchmark_pipeline_depth() {
  const char* pipeline_depth = std::getenv("HIER_AG_BENCH_PIPELINE_DEPTH");
  if (pipeline_depth != nullptr) {
    return static_cast<int>(std::strtol(pipeline_depth, nullptr, 10));
  }
  return kPipelineDepth;
}

float benchmark_timeout_ms() {
  const char* timeout_ms = std::getenv("HIER_AG_BENCH_TIMEOUT_MS");
  if (timeout_ms != nullptr) {
    return std::strtof(timeout_ms, nullptr);
  }
  return 30000.0f;
}

std::string benchmark_ib_hca() {
  const char* ib_hca = std::getenv("HIER_AG_BENCH_IB_HCA");
  if (ib_hca != nullptr) {
    return ib_hca;
  }
  return "";
}

bool env_flag_enabled(const char* name, bool default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }
  return std::strtol(value, nullptr, 10) != 0;
}

bool benchmark_nccl_disable_p2p() {
  return env_flag_enabled("HIER_AG_BENCH_NCCL_P2P_DISABLE", true);
}

bool benchmark_nccl_disable_shm() {
  return env_flag_enabled("HIER_AG_BENCH_NCCL_SHM_DISABLE", true);
}

bool benchmark_skip_nccl() {
  return env_flag_enabled("HIER_AG_BENCH_SKIP_NCCL", false);
}

std::size_t benchmark_nvl_signal_bytes() {
  const char* signal_bytes = std::getenv("HIER_AG_BENCH_NVL_SIGNAL_BYTES");
  if (signal_bytes != nullptr) {
    return std::strtoull(signal_bytes, nullptr, 10);
  }
  return 0;
}

std::size_t benchmark_ib_signal_bytes(
    std::size_t data_buffer_size,
    int num_blocks) {
  const char* signal_bytes = std::getenv("HIER_AG_BENCH_IB_SIGNAL_BYTES");
  if (signal_bytes != nullptr) {
    return std::strtoull(signal_bytes, nullptr, 10);
  }
  const std::size_t per_block_slot =
      (data_buffer_size / static_cast<std::size_t>(num_blocks)) & ~15ULL;
  return std::min(per_block_slot, kDefaultIbSignalBytes);
}

struct HierarchicalAllGatherBenchmarkConfig {
  std::size_t total_bytes;
  std::size_t sendcount;
  std::size_t ib_signaling_data_size;
  std::size_t nvl_signaling_data_size;
  int num_blocks;
  int ib_links;
  int ib_size;
  int nvl_size;
  std::string ib_hca;
  std::size_t data_buffer_size;
  int pipeline_depth;
  int num_qps;
  std::string name;
};

struct HierarchicalAllGatherBenchmarkResult {
  std::string test_name;
  std::size_t sendcount{};
  std::size_t total_bytes{};
  int num_blocks{};
  int ib_links{};
  int ib_nics{};
  std::size_t ib_signaling_data_size{};
  std::size_t data_buffer_size{};
  int pipeline_depth{};
  int ib_size{};
  int nvl_size{};
  float nccl_bandwidth{};
  float hierarchical_bandwidth{};
  float nccl_latency{};
  float hierarchical_latency{};
  float speedup{};
  bool nccl_p2p_disabled{};
  bool nccl_shm_disabled{};
};

class HierarchicalAllGatherBenchmarkFixture
    : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    CUDA_CHECK_VOID(cudaSetDevice(localRank));

    setenv("NCCL_ALGO", "Ring", 1);
    setenv("NCCL_PROTO", "Simple", 1);
    setenv("NCCL_NVLS_ENABLE", "0", 1);
    setenv("NCCL_P2P_DISABLE", benchmark_nccl_disable_p2p() ? "1" : "0", 1);
    setenv("NCCL_SHM_DISABLE", benchmark_nccl_disable_shm() ? "1" : "0", 1);
    const std::string ib_hca = benchmark_ib_hca();
    if (!ib_hca.empty()) {
      setenv("NCCL_IB_HCA", ib_hca.c_str(), 1);
    }
    const std::string num_blocks = std::to_string(benchmark_num_blocks());
    setenv("NCCL_MIN_NCHANNELS", num_blocks.c_str(), 1);
    setenv("NCCL_MAX_NCHANNELS", num_blocks.c_str(), 1);
    CUDA_CHECK_VOID(cudaStreamCreate(&stream_));
    if (!benchmark_skip_nccl()) {
      NCCL_CHECK_VOID(
          ncclCommInitRank(&nccl_comm_, worldSize, get_nccl_id(), globalRank));
    }
  }

  void TearDown() override {
    if (nccl_comm_ != nullptr) {
      NCCL_CHECK_VOID(ncclCommDestroy(nccl_comm_));
    }
    if (stream_ != nullptr) {
      CUDA_CHECK_VOID(cudaStreamDestroy(stream_));
    }
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
      const HierarchicalAllGatherBenchmarkConfig& config,
      float& latency_us) {
    const std::size_t recvcount = config.sendcount * worldSize;

    DeviceBuffer send_buf(config.sendcount);
    DeviceBuffer recv_buf(recvcount);
    CUDA_CHECK(cudaMemset(send_buf.get(), 1, config.sendcount));
    CUDA_CHECK(cudaMemset(recv_buf.get(), 0, recvcount));

    CudaEvent start, stop;
    const bool full_bench = std::getenv("HIER_AG_BENCH_FULL") != nullptr;
    const int n_warmup = full_bench ? 5 : 2;
    const int n_iter = full_bench ? 100 : 10;

    bootstrap->barrierAll();
    for (int i = 0; i < n_warmup; i++) {
      NCCL_CHECK(ncclAllGather(
          send_buf.get(),
          recv_buf.get(),
          config.sendcount,
          ncclChar,
          nccl_comm_,
          stream_));
    }

    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < n_iter; i++) {
      NCCL_CHECK(ncclAllGather(
          send_buf.get(),
          recv_buf.get(),
          config.sendcount,
          ncclChar,
          nccl_comm_,
          stream_));
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start.get(), stop.get()));
    float avg_ms = total_ms / n_iter;
    latency_us = avg_ms * 1000.0f;

    float bw = (recvcount / (1e9f)) / (avg_ms / 1e3f);
    bootstrap->barrierAll();
    return bw;
  }

  float run_hierarchical_benchmark(
      const HierarchicalAllGatherBenchmarkConfig& config,
      int& ib_nics,
      float& latency_us) {
    const std::size_t recvcount = config.sendcount * worldSize;
    const int nvl_rank = globalRank % config.nvl_size;
    const int ib_rank = globalRank / config.nvl_size;

    DeviceBuffer send_buf(config.sendcount);
    DeviceBuffer recv_buf(recvcount);
    CUDA_CHECK(cudaMemset(send_buf.get(), 1, config.sendcount));
    CUDA_CHECK(cudaMemset(recv_buf.get(), 0, recvcount));

    std::unique_ptr<MultipeerIbgdaTransport> ib_transport;
    try {
      MultipeerIbgdaTransportConfig transport_config{
          .cudaDevice = localRank,
          .dataBufferSize = config.data_buffer_size,
          .maxGroups = config.num_blocks,
          .sendRecv =
              MultipeerIbgdaTransportConfig::SendRecvConfig{
                  .maxGroups = config.num_blocks,
                  .pipelineDepth = config.pipeline_depth,
              },
          .qpsPerBlockPerNic = config.num_qps,
      };
      transport_config.ibHca = config.ib_hca;
      ib_transport = std::make_unique<MultipeerIbgdaTransport>(
          globalRank, worldSize, bootstrap, transport_config);
      ib_transport->exchange();
      ib_nics = ib_transport->numNics();
    } catch (const std::exception& e) {
      XLOGF(ERR, "IBGDA transport not available: {}", e.what());
      latency_us = 0.0f;
      return 0.0f;
    }

    std::unique_ptr<MultiPeerNvlTransport> nvl_transport;
    if (config.nvl_size > 1) {
      try {
        std::vector<int> nvl_rank_to_global(config.nvl_size);
        for (int peer = 0; peer < config.nvl_size; ++peer) {
          nvl_rank_to_global[peer] = ib_rank * config.nvl_size + peer;
        }
        auto nvl_bootstrap = std::make_shared<NvlBootstrapAdapter>(
            bootstrap, std::move(nvl_rank_to_global));
        MultiPeerNvlTransportConfig transport_config{
            .dataBufferSize = config.data_buffer_size,
            .chunkSize = config.data_buffer_size,
            .pipelineDepth = static_cast<std::size_t>(config.pipeline_depth),
            .p2pSignalCount = static_cast<std::size_t>(config.num_blocks),
            .tile_max_groups = config.num_blocks,
            .memSharingMode = MemSharingMode::kCudaIpc,
        };
        nvl_transport = std::make_unique<MultiPeerNvlTransport>(
            nvl_rank, config.nvl_size, nvl_bootstrap, transport_config);
        nvl_transport->exchange();
      } catch (const std::exception& e) {
        XLOGF(ERR, "NVLink transport not available: {}", e.what());
        latency_us = 0.0f;
        return 0.0f;
      }
    }

    HierarchicalAllgatherLaunchParams launch_params{};
    launch_params.num_ranks = worldSize;
    launch_params.ib_rank = ib_rank;
    launch_params.ib_size = config.ib_size;
    launch_params.nvl_rank = nvl_rank;
    launch_params.nvl_size = config.nvl_size;
    launch_params.sendcount = config.sendcount;
    launch_params.ib_signaling_data_size = config.ib_signaling_data_size;
    launch_params.nvl_signaling_data_size = config.nvl_signaling_data_size;
    launch_params.sendbuf = static_cast<const char*>(send_buf.get());
    launch_params.recvbuf = static_cast<char*>(recv_buf.get());
    launch_params.ib_num_blocks = config.num_blocks;
    launch_params.timeout_ms = benchmark_timeout_ms();
    launch_params.stream = stream_;

    if (config.ib_size > 1) {
      auto ib_rings_opt = make_standard_rings(config.ib_size, ib_rank, 1);
      if (!ib_rings_opt) {
        XLOGF(ERR, "Cannot construct IB ring for {} IB ranks", config.ib_size);
        latency_us = 0.0f;
        return 0.0f;
      }
      const auto& ib_ring = (*ib_rings_opt)[0];
      const int prev_global = ib_ring.prev_rank * config.nvl_size + nvl_rank;
      const int next_global = ib_ring.next_rank * config.nvl_size + nvl_rank;
      launch_params.ib_ring.prev_rank = ib_ring.prev_rank;
      launch_params.ib_ring.next_rank = ib_ring.next_rank;
      launch_params.ib_ring.prev = P2pIbTransportDevice(
          ib_transport->getP2pTransportDevice(prev_global));
      launch_params.ib_ring.next = P2pIbTransportDevice(
          ib_transport->getP2pTransportDevice(next_global));
    }

    for (int peer = 0; peer < config.nvl_size; ++peer) {
      if (peer == nvl_rank) {
        continue;
      }
      new (&launch_params.nvl_peers[peer])
          P2pNvlTransportDevice(nvl_transport->getP2pTransportDevice(peer));
    }

    CudaEvent start, stop;
    const bool full_bench = std::getenv("HIER_AG_BENCH_FULL") != nullptr;
    const int n_warmup = full_bench ? 5 : 2;
    const int n_iter = full_bench ? 100 : 10;

    bootstrap->barrierAll();
    for (int i = 0; i < n_warmup; i++) {
      launch_hierarchical_allgather_fused(launch_params);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    bootstrap->barrierAll();

    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < n_iter; i++) {
      launch_hierarchical_allgather_fused(launch_params);
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start.get(), stop.get()));
    float avg_ms = total_ms / n_iter;
    latency_us = avg_ms * 1000.0f;

    float bw = (recvcount / (1e9f)) / (avg_ms / 1e3f);
    bootstrap->barrierAll();
    return bw;
  }

  void print_results(
      const std::vector<HierarchicalAllGatherBenchmarkResult>& results) {
    if (globalRank != 0) {
      return;
    }

    std::stringstream ss;
    ss << "\n";
    ss << "================================================================================\n";
    ss << "      NCCL AllGather vs Hierarchical AllGather (IB Ring + NVLink AG)\n";
    ss << "================================================================================\n";
    ss << std::left << std::setw(14) << "Test" << std::right << std::setw(10)
       << "Size" << std::right << std::setw(10) << "Chunk" << std::right
       << std::setw(8) << "Blocks" << std::right << std::setw(6) << "IB"
       << std::right << std::setw(6) << "NVL" << std::right << std::setw(8)
       << "IBNics" << std::right << std::setw(9) << "IBLinks" << std::right
       << std::setw(10) << "IBSig" << std::right << std::setw(10) << "DBuf"
       << std::right << std::setw(7) << "Pipe" << std::right << std::setw(12)
       << "NCCL Alg" << std::right << std::setw(12) << "Hier Alg" << std::right
       << std::setw(10) << "Speedup" << std::right << std::setw(12)
       << "NCCL Lat" << std::right << std::setw(12) << "Hier Lat\n";
    ss << std::left << std::setw(14) << "" << std::right << std::setw(10) << ""
       << std::right << std::setw(10) << "" << std::right << std::setw(8) << ""
       << std::right << std::setw(6) << "" << std::right << std::setw(6) << ""
       << std::right << std::setw(8) << "" << std::right << std::setw(9)
       << "(QPs/NIC)" << std::right << std::setw(10) << "" << std::right
       << std::setw(10) << "" << std::right << std::setw(7) << "" << std::right
       << std::setw(12) << "(GB/s)" << std::right << std::setw(12) << "(GB/s)"
       << std::right << std::setw(10) << "" << std::right << std::setw(12)
       << "(us)" << std::right << std::setw(12) << "(us)\n";
    ss << "--------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      ss << std::left << std::setw(14) << r.test_name << std::right
         << std::setw(10) << format_size(r.total_bytes) << std::right
         << std::setw(10) << format_size(r.sendcount) << std::right
         << std::setw(8) << r.num_blocks << std::right << std::setw(6)
         << r.ib_size << std::right << std::setw(6) << r.nvl_size << std::right
         << std::setw(8) << r.ib_nics << std::right << std::setw(9)
         << r.ib_links << std::right << std::setw(10)
         << (r.ib_signaling_data_size == 0
                 ? "auto"
                 : format_size(r.ib_signaling_data_size))
         << std::right << std::setw(10) << format_size(r.data_buffer_size)
         << std::right << std::setw(7) << r.pipeline_depth << std::right
         << std::setw(12) << std::fixed << std::setprecision(2)
         << r.nccl_bandwidth << std::right << std::setw(12) << std::fixed
         << std::setprecision(2) << r.hierarchical_bandwidth << std::right
         << std::setw(9) << std::fixed << std::setprecision(2) << r.speedup
         << "x" << std::right << std::setw(12) << std::fixed
         << std::setprecision(1) << r.nccl_latency << std::right
         << std::setw(12) << std::fixed << std::setprecision(1)
         << r.hierarchical_latency << "\n";
    }

    ss << "================================================================================\n";
    ss << worldSize
       << " ranks, size = total input bytes, chunk = per-rank send bytes\n";
    ss << "Algorithmic bandwidth = total input bytes / latency, not NIC wire bandwidth\n";
    if (!results.empty()) {
      ss << "NCCL baseline: Ring/Simple, NVLS disabled, channels matched to blocks, P2P_DISABLE="
         << (results.front().nccl_p2p_disabled ? "1" : "0")
         << ", SHM_DISABLE=" << (results.front().nccl_shm_disabled ? "1" : "0")
         << "\n";
    }
    ss << "================================================================================\n";

    XLOG(INFO) << ss.str();
  }

  ncclComm_t nccl_comm_{};
  cudaStream_t stream_{};
};

TEST_F(HierarchicalAllGatherBenchmarkFixture, VsNccl) {
  const int nvl_size = benchmark_nvl_size(worldSize);
  if (nvl_size <= 0 || nvl_size > kDirectNvlMaxRanks ||
      worldSize % nvl_size != 0) {
    XLOGF(
        ERR,
        "Invalid HIER_AG_BENCH_NVL_SIZE={} for worldSize={}",
        nvl_size,
        worldSize);
    std::abort();
  }

  if (globalRank == 0) {
    XLOGF(
        INFO,
        "\n=== Hierarchical AllGather vs NCCL Comparison (IB groups={}, NVL groups={}) ===\n",
        worldSize / nvl_size,
        nvl_size);
  }

  std::vector<HierarchicalAllGatherBenchmarkConfig> configs;
  const int num_blocks = benchmark_num_blocks();
  const int ib_links = benchmark_ib_links();
  const std::size_t data_buffer_size = benchmark_data_buffer_size();
  const int pipeline_depth = benchmark_pipeline_depth();
  const std::string ib_hca = benchmark_ib_hca();
  if (ib_links <= 0) {
    XLOGF(ERR, "Invalid HIER_AG_BENCH_IB_LINKS={}", ib_links);
    std::abort();
  }
  if (data_buffer_size == 0) {
    XLOG(ERR, "Invalid HIER_AG_BENCH_DATA_BUFFER_SIZE=0");
    std::abort();
  }
  if (pipeline_depth <= 0) {
    XLOGF(ERR, "Invalid HIER_AG_BENCH_PIPELINE_DEPTH={}", pipeline_depth);
    std::abort();
  }
  const std::size_t ib_signal_bytes =
      benchmark_ib_signal_bytes(data_buffer_size, num_blocks);
  const std::size_t nvl_signal_bytes = benchmark_nvl_signal_bytes();
  const bool nccl_p2p_disabled = benchmark_nccl_disable_p2p();
  const bool nccl_shm_disabled = benchmark_nccl_disable_shm();
  const bool skip_nccl = benchmark_skip_nccl();
  for (std::size_t total_bytes : configured_benchmark_sizes()) {
    if (total_bytes % static_cast<std::size_t>(worldSize) != 0) {
      XLOGF(
          ERR,
          "AllGather size {} is not divisible by {} ranks",
          total_bytes,
          worldSize);
      std::abort();
    }
    configs.push_back({
        .total_bytes = total_bytes,
        .sendcount = total_bytes / static_cast<std::size_t>(worldSize),
        .ib_signaling_data_size = ib_signal_bytes,
        .nvl_signaling_data_size = nvl_signal_bytes,
        .num_blocks = num_blocks,
        .ib_links = ib_links,
        .ib_size = worldSize / nvl_size,
        .nvl_size = nvl_size,
        .ib_hca = ib_hca,
        .data_buffer_size = data_buffer_size,
        .pipeline_depth = pipeline_depth,
        .num_qps = ib_links,
        .name =
            format_size(total_bytes) + "_" + std::to_string(num_blocks) + "B",
    });
  }

  std::vector<HierarchicalAllGatherBenchmarkResult> results;

  for (const auto& config : configs) {
    if (globalRank == 0) {
      XLOGF(
          INFO,
          "Running hierarchical AllGather {}: size={}, chunk={}, blocks={}, ib_size={}, nvl_size={}, ib_links_per_nic={}, ib_signal_bytes={}, nvl_signal_bytes={}",
          config.name,
          format_size(config.total_bytes),
          format_size(config.sendcount),
          config.num_blocks,
          config.ib_size,
          config.nvl_size,
          config.ib_links,
          config.ib_signaling_data_size,
          config.nvl_signaling_data_size);
    }

    float nccl_lat = 0.0f;
    float nccl_bw = 0.0f;
    if (!skip_nccl) {
      nccl_bw = run_nccl_benchmark(config, nccl_lat);
    }

    float hierarchical_lat = 0.0f;
    int ib_nics = 0;
    float hierarchical_bw =
        run_hierarchical_benchmark(config, ib_nics, hierarchical_lat);

    if (globalRank == 0) {
      XLOGF(
          INFO,
          "AllGather {} result: NCCL={:.2f} GB/s ({:.1f} us), Hier={:.2f} GB/s ({:.1f} us), Hier/NCCL={:.2f}x",
          config.name,
          nccl_bw,
          nccl_lat,
          hierarchical_bw,
          hierarchical_lat,
          (nccl_bw > 0) ? hierarchical_bw / nccl_bw : 0.0f);
      results.push_back({
          .test_name = config.name,
          .sendcount = config.sendcount,
          .total_bytes = config.total_bytes,
          .num_blocks = config.num_blocks,
          .ib_links = config.ib_links,
          .ib_nics = ib_nics,
          .ib_signaling_data_size = config.ib_signaling_data_size,
          .data_buffer_size = config.data_buffer_size,
          .pipeline_depth = config.pipeline_depth,
          .ib_size = config.ib_size,
          .nvl_size = config.nvl_size,
          .nccl_bandwidth = nccl_bw,
          .hierarchical_bandwidth = hierarchical_bw,
          .nccl_latency = nccl_lat,
          .hierarchical_latency = hierarchical_lat,
          .speedup = (nccl_bw > 0) ? hierarchical_bw / nccl_bw : 0.0f,
          .nccl_p2p_disabled = nccl_p2p_disabled,
          .nccl_shm_disabled = nccl_shm_disabled,
      });
    }
    bootstrap->barrierAll();
  }

  print_results(results);
}

} // namespace

} // namespace comms::prims::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::BenchmarkEnvironment());
  return RUN_ALL_TESTS();
}
