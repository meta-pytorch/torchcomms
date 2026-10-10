// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/MultipeerIbgdaTransport.h"
#include "comms/pipes/benchmarks/BenchmarkMacros.h"
#include "comms/pipes/bootstrap/NvlBootstrapAdapter.h"
#include "comms/pipes/collectives/HierarchicalReduceScatterLauncher.h"
#include "comms/pipes/collectives/RingUtils.h"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

namespace {

constexpr int kBenchmarkNumBlocks = 8;
constexpr int kBenchmarkIbLinks = 4;
constexpr std::size_t kDataBufferSize = 32UL * 1024 * 1024;
constexpr int kPipelineDepth = 4;
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
  const char* only_bytes = std::getenv("HIER_RS_BENCH_ONLY_BYTES");
  if (only_bytes != nullptr) {
    return {std::strtoull(only_bytes, nullptr, 10)};
  }
  return benchmark_sizes();
}

int read_int_env(const char* name, int default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }
  return static_cast<int>(std::strtol(value, nullptr, 10));
}

std::size_t read_size_env(const char* name, std::size_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }
  return std::strtoull(value, nullptr, 10);
}

bool env_flag_enabled(const char* name, bool default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }
  return std::strtol(value, nullptr, 10) != 0;
}

int benchmark_num_blocks() {
  return read_int_env("HIER_RS_BENCH_NUM_BLOCKS", kBenchmarkNumBlocks);
}

int benchmark_nvl_size(int world_size) {
  const char* nvl_size = std::getenv("HIER_RS_BENCH_NVL_SIZE");
  if (nvl_size != nullptr) {
    return static_cast<int>(std::strtol(nvl_size, nullptr, 10));
  }
  const int local_size = read_int_env("LOCAL_SIZE", 0);
  if (local_size > 0 && local_size <= kDirectNvlMaxRanks &&
      world_size % local_size == 0) {
    return local_size;
  }
  return world_size % 4 == 0 ? 4 : world_size;
}

std::vector<int> parse_int_csv(const char* env_name) {
  std::vector<int> out;
  const char* csv = std::getenv(env_name);
  if (csv == nullptr) {
    return out;
  }
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    int v = static_cast<int>(std::strtol(item.c_str(), nullptr, 10));
    if (v > 0) {
      out.push_back(v);
    }
  }
  return out;
}

std::vector<int> benchmark_pipeline_depths(int default_depth) {
  auto v = parse_int_csv("HIER_RS_BENCH_PIPELINE_DEPTHS");
  if (!v.empty()) {
    return v;
  }
  return {read_int_env("HIER_RS_BENCH_PIPELINE_DEPTH", default_depth)};
}

std::vector<int> benchmark_nvl_sizes(int world_size) {
  std::vector<int> valid;
  auto admit = [&](int n) {
    if (n > 0 && n <= kDirectNvlMaxRanks && world_size % n == 0) {
      valid.push_back(n);
    }
  };
  auto v = parse_int_csv("HIER_RS_BENCH_NVL_SIZES");
  if (!v.empty()) {
    for (int n : v) {
      admit(n);
    }
    if (!valid.empty()) {
      return valid;
    }
  }
  admit(benchmark_nvl_size(world_size));
  return valid;
}

std::size_t benchmark_ib_signal_bytes(
    std::size_t data_buffer_size,
    int num_blocks) {
  const char* signal_bytes = std::getenv("HIER_RS_BENCH_IB_SIGNAL_BYTES");
  if (signal_bytes != nullptr) {
    return std::strtoull(signal_bytes, nullptr, 10);
  }
  const std::size_t per_block_slot =
      (data_buffer_size / static_cast<std::size_t>(num_blocks)) & ~15ULL;
  return std::min(per_block_slot, kDefaultIbSignalBytes);
}

struct HierarchicalReduceScatterBenchmarkConfig {
  std::size_t chunk_elements;
  std::size_t total_bytes;
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

struct HierarchicalReduceScatterBenchmarkResult {
  std::string test_name;
  std::size_t chunk_elements{};
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

class HierarchicalReduceScatterBenchmarkFixture
    : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    CUDA_CHECK_VOID(cudaSetDevice(localRank));

    if (localSize <= 1) {
      setenv("NCCL_ALGO", "Ring", 1);
      setenv("NCCL_PROTO", "Simple", 1);
    }
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

  std::string benchmark_ib_hca() {
    const char* ib_hca = std::getenv("HIER_RS_BENCH_IB_HCA");
    if (ib_hca != nullptr) {
      return ib_hca;
    }
    return "";
  }

  bool benchmark_nccl_disable_p2p() {
    return env_flag_enabled("HIER_RS_BENCH_NCCL_P2P_DISABLE", true);
  }

  bool benchmark_nccl_disable_shm() {
    return env_flag_enabled("HIER_RS_BENCH_NCCL_SHM_DISABLE", true);
  }

  bool benchmark_skip_nccl() {
    return env_flag_enabled("HIER_RS_BENCH_SKIP_NCCL", false);
  }

  float benchmark_timeout_ms() {
    const char* timeout_ms = std::getenv("HIER_RS_BENCH_TIMEOUT_MS");
    if (timeout_ms != nullptr) {
      return std::strtof(timeout_ms, nullptr);
    }
    return 0.0f;
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
      const HierarchicalReduceScatterBenchmarkConfig& config,
      float& latency_us) {
    DeviceBuffer send_buf(config.total_bytes);
    DeviceBuffer recv_buf(config.chunk_elements * sizeof(float));
    CUDA_CHECK(cudaMemset(send_buf.get(), 0, config.total_bytes));
    CUDA_CHECK(
        cudaMemset(recv_buf.get(), 0, config.chunk_elements * sizeof(float)));

    CudaEvent start, stop;
    const bool full_bench = std::getenv("HIER_RS_BENCH_FULL") != nullptr;
    const int n_warmup = full_bench ? 5 : 2;
    const int n_iter = full_bench ? 100 : 10;

    bootstrap->barrierAll();
    for (int i = 0; i < n_warmup; i++) {
      NCCL_CHECK(ncclReduceScatter(
          send_buf.get(),
          recv_buf.get(),
          config.chunk_elements,
          ncclFloat,
          ncclSum,
          nccl_comm_,
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
          nccl_comm_,
          stream_));
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start.get(), stop.get()));
    const float avg_ms = total_ms / n_iter;
    latency_us = avg_ms * 1000.0f;

    const float bw = (config.total_bytes / 1e9f) / (avg_ms / 1e3f);
    bootstrap->barrierAll();
    return bw;
  }

  float run_hierarchical_benchmark(
      const HierarchicalReduceScatterBenchmarkConfig& config,
      int& ib_nics,
      float& latency_us) {
    const int nvl_rank = globalRank % config.nvl_size;
    const int ib_rank = globalRank / config.nvl_size;
    const std::size_t chunk_bytes = config.chunk_elements * sizeof(float);
    const std::size_t workspace_bytes =
        static_cast<std::size_t>(config.ib_size) * chunk_bytes;

    DeviceBuffer send_buf(config.total_bytes);
    DeviceBuffer recv_buf(chunk_bytes);
    DeviceBuffer workspace_buf(workspace_bytes);
    CUDA_CHECK(cudaMemset(send_buf.get(), 0, config.total_bytes));
    CUDA_CHECK(cudaMemset(recv_buf.get(), 0, chunk_bytes));
    CUDA_CHECK(cudaMemset(workspace_buf.get(), 0, workspace_bytes));

    std::unique_ptr<MultipeerIbgdaTransport> ib_transport;
    if (config.ib_size > 1) {
      try {
        MultipeerIbgdaTransportConfig transport_config{
            .cudaDevice = localRank,
            .dataBufferSize = config.data_buffer_size,
            .sendRecv =
                MultipeerIbgdaTransportConfig::SendRecvConfig{
                    .maxGroups = config.num_blocks,
                    .pipelineDepth = config.pipeline_depth,
                },
            .numQpsPerPeerPerNic = config.num_qps,
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

    HierarchicalReduceScatterLaunchParams launch_params{};
    launch_params.num_ranks = worldSize;
    launch_params.ib_rank = ib_rank;
    launch_params.ib_size = config.ib_size;
    launch_params.nvl_rank = nvl_rank;
    launch_params.nvl_size = config.nvl_size;
    launch_params.chunk_elements = config.chunk_elements;
    launch_params.ib_signaling_data_size = config.ib_signaling_data_size;
    launch_params.nvl_signaling_data_size = config.nvl_signaling_data_size;
    launch_params.input = static_cast<const float*>(send_buf.get());
    launch_params.output = static_cast<float*>(recv_buf.get());
    launch_params.workspace = static_cast<float*>(workspace_buf.get());
    launch_params.num_blocks = config.num_blocks;
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
      launch_params.ib_ring.prev =
          ib_transport->getP2pTransportDevice(prev_global);
      launch_params.ib_ring.next =
          ib_transport->getP2pTransportDevice(next_global);
    }

    for (int peer = 0; peer < config.nvl_size; ++peer) {
      if (peer == nvl_rank) {
        continue;
      }
      new (&launch_params.nvl_peers[peer])
          P2pNvlTransportDevice(nvl_transport->getP2pTransportDevice(peer));
    }

    CudaEvent start, stop;
    const bool full_bench = std::getenv("HIER_RS_BENCH_FULL") != nullptr;
    const int n_warmup = full_bench ? 5 : 2;
    const int n_iter = full_bench ? 100 : 10;

    bootstrap->barrierAll();
    for (int i = 0; i < n_warmup; i++) {
      launch_hierarchical_reduce_scatter_fused(launch_params);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    bootstrap->barrierAll();

    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < n_iter; i++) {
      launch_hierarchical_reduce_scatter_fused(launch_params);
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start.get(), stop.get()));
    const float avg_ms = total_ms / n_iter;
    latency_us = avg_ms * 1000.0f;

    const float bw = (config.total_bytes / 1e9f) / (avg_ms / 1e3f);
    bootstrap->barrierAll();
    return bw;
  }

  void print_results(
      const std::vector<HierarchicalReduceScatterBenchmarkResult>& results) {
    if (globalRank != 0) {
      return;
    }

    std::stringstream ss;
    ss << "\n";
    ss << "================================================================================\n";
    ss << "  NCCL ReduceScatter vs Hierarchical ReduceScatter (NVLink RS + IB Ring RS)\n";
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
    ss << "--------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      ss << std::left << std::setw(14) << r.test_name << std::right
         << std::setw(10) << format_size(r.total_bytes) << std::right
         << std::setw(10) << format_size(r.chunk_elements * sizeof(float))
         << std::right << std::setw(8) << r.num_blocks << std::right
         << std::setw(6) << r.ib_size << std::right << std::setw(6)
         << r.nvl_size << std::right << std::setw(8) << r.ib_nics << std::right
         << std::setw(9) << r.ib_links << std::right << std::setw(10)
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
       << " ranks, size = total input bytes, chunk = per-rank output bytes\n";
    ss << "Algorithmic bandwidth = total input bytes / latency, not NIC wire bandwidth\n";
    if (!results.empty()) {
      ss << "NCCL baseline: P2P_DISABLE="
         << (results.front().nccl_p2p_disabled ? "1" : "0")
         << ", SHM_DISABLE=" << (results.front().nccl_shm_disabled ? "1" : "0")
         << "\n";
    }
    ss << "================================================================================\n";

    XLOG(INFO) << ss.str();
  }

  void print_best_per_size(
      const std::vector<HierarchicalReduceScatterBenchmarkResult>& results) {
    if (globalRank != 0) {
      return;
    }

    std::map<std::size_t, const HierarchicalReduceScatterBenchmarkResult*> best;
    for (const auto& r : results) {
      // Skip rows where the config was skipped at runtime (transport
      // unavailable) — those report bandwidth=0 and are noise.
      if (r.hierarchical_bandwidth <= 0.0f) {
        continue;
      }
      auto it = best.find(r.total_bytes);
      if (it == best.end() ||
          r.hierarchical_bandwidth > it->second->hierarchical_bandwidth) {
        best[r.total_bytes] = &r;
      }
    }

    if (best.empty()) {
      XLOG(
          INFO,
          "No successful Hier measurements — Best Config Per Size table empty.");
      return;
    }

    std::stringstream ss;
    ss << "\n";
    ss << "================================================================================\n";
    ss << "  Best Hier Config Per Size (max bandwidth across nvl_size × pipeline_depth)\n";
    ss << "================================================================================\n";
    ss << std::left << std::setw(10) << "Size" << std::right << std::setw(6)
       << "NVL" << std::right << std::setw(6) << "IB" << std::right
       << std::setw(7) << "Pipe" << std::right << std::setw(14) << "Hier GB/s"
       << std::right << std::setw(14) << "Hier us" << std::right
       << std::setw(14) << "NCCL GB/s" << std::right << std::setw(14)
       << "NCCL us" << std::right << std::setw(12) << "Hier/NCCL\n";
    ss << "--------------------------------------------------------------------------------\n";
    for (const auto& kv : best) {
      const auto* r = kv.second;
      ss << std::left << std::setw(10) << format_size(r->total_bytes)
         << std::right << std::setw(6) << r->nvl_size << std::right
         << std::setw(6) << r->ib_size << std::right << std::setw(7)
         << r->pipeline_depth << std::right << std::setw(14) << std::fixed
         << std::setprecision(2) << r->hierarchical_bandwidth << std::right
         << std::setw(14) << std::fixed << std::setprecision(1)
         << r->hierarchical_latency << std::right << std::setw(14) << std::fixed
         << std::setprecision(2) << r->nccl_bandwidth << std::right
         << std::setw(14) << std::fixed << std::setprecision(1)
         << r->nccl_latency << std::right << std::setw(11) << std::fixed
         << std::setprecision(2) << r->speedup << "x\n";
    }
    ss << "================================================================================\n";
    XLOG(INFO) << ss.str();
  }

  ncclComm_t nccl_comm_{};
  cudaStream_t stream_{};
};

TEST_F(HierarchicalReduceScatterBenchmarkFixture, VsNccl) {
  const std::vector<int> nvl_sizes = benchmark_nvl_sizes(worldSize);
  if (nvl_sizes.empty()) {
    XLOGF(
        ERR,
        "No valid nvl_size values for worldSize={} (expect divisors of worldSize, <= {})",
        worldSize,
        kDirectNvlMaxRanks);
    std::abort();
  }

  const int num_blocks = benchmark_num_blocks();
  const int ib_links =
      read_int_env("HIER_RS_BENCH_IB_LINKS", kBenchmarkIbLinks);
  const std::size_t data_buffer_size =
      read_size_env("HIER_RS_BENCH_DATA_BUFFER_SIZE", kDataBufferSize);
  const std::vector<int> pipeline_depths =
      benchmark_pipeline_depths(kPipelineDepth);
  if (num_blocks <= 0 || ib_links <= 0 || data_buffer_size == 0 ||
      pipeline_depths.empty()) {
    XLOG(ERR, "Invalid hierarchical reduce-scatter benchmark configuration");
    std::abort();
  }

  if (globalRank == 0) {
    std::stringstream nvl_str;
    for (size_t i = 0; i < nvl_sizes.size(); ++i) {
      nvl_str << (i ? "," : "") << nvl_sizes[i];
    }
    std::stringstream depth_str;
    for (size_t i = 0; i < pipeline_depths.size(); ++i) {
      depth_str << (i ? "," : "") << pipeline_depths[i];
    }
    XLOGF(
        INFO,
        "\n=== Hierarchical ReduceScatter vs NCCL Sweep (worldSize={}, nvl_sizes=[{}], pipeline_depths=[{}]) ===\n",
        worldSize,
        nvl_str.str(),
        depth_str.str());
  }

  const bool skip_nccl = benchmark_skip_nccl();
  const std::size_t ib_signal_bytes =
      benchmark_ib_signal_bytes(data_buffer_size, num_blocks);
  const std::size_t nvl_signal_bytes =
      read_size_env("HIER_RS_BENCH_NVL_SIGNAL_BYTES", 0);
  const std::string ib_hca = benchmark_ib_hca();
  const bool nccl_p2p_disabled = benchmark_nccl_disable_p2p();
  const bool nccl_shm_disabled = benchmark_nccl_disable_shm();

  std::vector<HierarchicalReduceScatterBenchmarkResult> results;
  std::map<std::size_t, std::pair<float, float>> nccl_cache;

  for (int nvl_size : nvl_sizes) {
    const int ib_size = worldSize / nvl_size;
    if (ib_size == 1 && nvl_size == 1) {
      if (globalRank == 0) {
        XLOG(
            INFO,
            "Skipping hierarchical ReduceScatter nvl_size=1 ib_size=1: degenerate local-copy topology with no IB/NVL transport to benchmark.");
      }
      continue;
    }

    if (globalRank == 0) {
      XLOGF(
          INFO, "--- Sweeping nvl_size={} (ib_size={}) ---", nvl_size, ib_size);
    }

    for (std::size_t total_bytes : configured_benchmark_sizes()) {
      const std::size_t total_elements = total_bytes / sizeof(float);
      if (total_bytes % sizeof(float) != 0 ||
          total_elements % static_cast<std::size_t>(worldSize) != 0) {
        XLOGF(
            ERR,
            "ReduceScatter size {} is not divisible by {} float ranks",
            total_bytes,
            worldSize);
        std::abort();
      }
      const std::size_t chunk_elements =
          total_elements / static_cast<std::size_t>(worldSize);

      // NCCL is invariant w.r.t. nvl_size and pipeline_depth — run once per
      // total_bytes and cache the result.
      float nccl_lat = 0.0f;
      float nccl_bw = 0.0f;
      if (!skip_nccl) {
        auto it = nccl_cache.find(total_bytes);
        if (it != nccl_cache.end()) {
          nccl_bw = it->second.first;
          nccl_lat = it->second.second;
        } else {
          HierarchicalReduceScatterBenchmarkConfig nccl_cfg{
              .chunk_elements = chunk_elements,
              .total_bytes = total_bytes,
              .ib_signaling_data_size = ib_signal_bytes,
              .nvl_signaling_data_size = nvl_signal_bytes,
              .num_blocks = num_blocks,
              .ib_links = ib_links,
              .ib_size = ib_size,
              .nvl_size = nvl_size,
              .ib_hca = ib_hca,
              .data_buffer_size = data_buffer_size,
              .pipeline_depth = pipeline_depths.front(),
              .num_qps = ib_links,
              .name = format_size(total_bytes),
          };
          nccl_bw = run_nccl_benchmark(nccl_cfg, nccl_lat);
          nccl_cache[total_bytes] = {nccl_bw, nccl_lat};
        }
      }

      for (int depth : pipeline_depths) {
        HierarchicalReduceScatterBenchmarkConfig config{
            .chunk_elements = chunk_elements,
            .total_bytes = total_bytes,
            .ib_signaling_data_size = ib_signal_bytes,
            .nvl_signaling_data_size = nvl_signal_bytes,
            .num_blocks = num_blocks,
            .ib_links = ib_links,
            .ib_size = ib_size,
            .nvl_size = nvl_size,
            .ib_hca = ib_hca,
            .data_buffer_size = data_buffer_size,
            .pipeline_depth = depth,
            .num_qps = ib_links,
            .name = format_size(total_bytes) + "_nvl" +
                std::to_string(nvl_size) + "_d" + std::to_string(depth),
        };

        float hierarchical_lat = 0.0f;
        int ib_nics = 0;
        float hierarchical_bw =
            run_hierarchical_benchmark(config, ib_nics, hierarchical_lat);

        if (globalRank == 0) {
          XLOGF(
              INFO,
              "ReduceScatter {} result: NCCL={:.2f} GB/s ({:.1f} us), Hier={:.2f} GB/s ({:.1f} us), Hier/NCCL={:.2f}x",
              config.name,
              nccl_bw,
              nccl_lat,
              hierarchical_bw,
              hierarchical_lat,
              (nccl_bw > 0) ? hierarchical_bw / nccl_bw : 0.0f);
          results.push_back({
              .test_name = config.name,
              .chunk_elements = config.chunk_elements,
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
    }
  }

  print_results(results);
  print_best_per_size(results);
}

} // namespace

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::BenchmarkEnvironment());
  return RUN_ALL_TESTS();
}
