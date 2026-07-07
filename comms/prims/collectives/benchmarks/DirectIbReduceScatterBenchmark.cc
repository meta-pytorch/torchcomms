// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <glog/logging.h>
#include <nccl.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "comms/prims/collectives/ReduceScatterDirectIbLauncher.h"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;

namespace comms::prims::benchmark {
namespace {

void check_cuda(cudaError_t result, const char* what) {
  CHECK_EQ(result, cudaSuccess) << what << ": " << cudaGetErrorString(result);
}

void check_nccl(ncclResult_t result, const char* what) {
  CHECK_EQ(result, ncclSuccess) << what << ": " << ncclGetErrorString(result);
}

std::size_t env_size(const char* name, std::size_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  return static_cast<std::size_t>(std::strtoull(value, nullptr, 10));
}

int env_int(const char* name, int fallback) {
  return static_cast<int>(env_size(name, static_cast<std::size_t>(fallback)));
}

struct DirectIbReduceScatterConfig {
  std::size_t total_bytes{16UL * 1024 * 1024};
  int num_blocks{8};
  int warmup{20};
  int repeats{20};
  std::size_t per_channel_size{512UL * 1024};
  int pipeline_depth{4};
  int qps_per_connection{1};
  std::size_t signal_bytes{0};
  float timeout_ms{30000.0f};
};

struct TimedResult {
  float latency_us{0.0f};
  float bandwidth_gbps{0.0f};
};

struct SweepResult {
  std::string size_label;
  TimedResult nccl_ring;
  TimedResult nccl_pat;
  TimedResult rsdirect_ib;
};

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const char* value) : name_(name) {
    const char* previous = std::getenv(name);
    if (previous != nullptr) {
      previous_ = previous;
      had_previous_ = true;
    }
    setenv(name, value, 1);
  }

  ~ScopedEnvVar() {
    if (had_previous_) {
      setenv(name_.c_str(), previous_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::string previous_;
  bool had_previous_{false};
};

class DirectIbReduceScatterBenchmark
    : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    check_cuda(cudaSetDevice(localRank), "cudaSetDevice");
    nccl_envs_.push_back(
        std::make_unique<ScopedEnvVar>("NCCL_P2P_DISABLE", "1"));
    check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
  }

  void TearDown() override {
    check_cuda(cudaStreamDestroy(stream_), "cudaStreamDestroy");
    nccl_envs_.clear();
    BenchmarkTestFixture::TearDown();
  }

  DirectIbReduceScatterConfig config_from_env() const {
    DirectIbReduceScatterConfig config{};
    config.num_blocks = env_int("PRIMS_RSDIRECT_IB_BLOCKS", config.num_blocks);
    config.warmup = env_int("PRIMS_RSDIRECT_IB_WARMUP", config.warmup);
    config.repeats = env_int("PRIMS_RSDIRECT_IB_REPEATS", config.repeats);
    config.per_channel_size =
        env_size("PRIMS_RSDIRECT_IB_PER_CHANNEL_SIZE", config.per_channel_size);
    config.pipeline_depth =
        env_int("PRIMS_RSDIRECT_IB_PIPELINE_DEPTH", config.pipeline_depth);
    config.qps_per_connection = env_int(
        "PRIMS_RSDIRECT_IB_QPS_PER_CONNECTION", config.qps_per_connection);
    config.timeout_ms = static_cast<float>(env_size(
        "PRIMS_RSDIRECT_IB_TIMEOUT_MS",
        static_cast<std::size_t>(config.timeout_ms)));
    config.num_blocks = std::max(config.num_blocks, 1);
    config.repeats = std::max(config.repeats, 1);
    config.warmup = std::max(config.warmup, 0);
    return config;
  }

  ncclUniqueId get_nccl_id() {
    ncclUniqueId id{};
    if (globalRank == 0) {
      check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
    }

    std::vector<ncclUniqueId> all_ids(worldSize);
    all_ids[globalRank] = id;
    const int result =
        bootstrap
            ->allGather(
                all_ids.data(), sizeof(ncclUniqueId), globalRank, worldSize)
            .get();
    CHECK_EQ(result, 0);
    return all_ids[0];
  }

  ncclComm_t make_nccl_comm() {
    ncclComm_t nccl_comm{};
    check_nccl(
        ncclCommInitRank(&nccl_comm, worldSize, get_nccl_id(), globalRank),
        "ncclCommInitRank");
    return nccl_comm;
  }

  void destroy_nccl_comm(ncclComm_t nccl_comm) {
    const ncclResult_t res = ncclCommDestroy(nccl_comm);
    if (res != ncclSuccess) {
      XLOGF(ERR, "ncclCommDestroy failed: {}", ncclGetErrorString(res));
    }
  }

  std::unique_ptr<MultipeerIbgdaTransport> make_transport(
      const DirectIbReduceScatterConfig& config) {
    MultipeerIbgdaTransportConfig transport_config{
        .cudaDevice = localRank,
        .perChannelSize = config.per_channel_size,
        .max_num_channels = config.num_blocks,
        .pipelineDepth = config.pipeline_depth,
        .qpsPerConnection = config.qps_per_connection,
    };

    auto transport = std::make_unique<MultipeerIbgdaTransport>(
        globalRank, worldSize, bootstrap, transport_config);
    transport->exchange();
    for (int peer = 0; peer < worldSize; ++peer) {
      if (peer == globalRank) {
        continue;
      }
      transport->queuePeerForMaterialization(peer);
    }
    transport->connectPeers();
    return transport;
  }

  std::vector<P2pIbTransportDevice> get_peer_devices(
      MultipeerIbgdaTransport& transport) const {
    std::vector<P2pIbTransportDevice> peers(worldSize);
    for (int peer = 0; peer < worldSize; ++peer) {
      if (peer == globalRank) {
        continue;
      }
      peers[peer] = P2pIbTransportDevice(transport.getP2pTransportDevice(peer));
    }
    return peers;
  }

  TimedResult run_nccl(
      const DirectIbReduceScatterConfig& config,
      ncclComm_t nccl_comm,
      DeviceBuffer& send_buf,
      DeviceBuffer& recv_buf) {
    const std::size_t total_elements = config.total_bytes / sizeof(float);
    const std::size_t chunk_elements = total_elements / worldSize;
    const std::size_t total_bytes =
        chunk_elements * static_cast<std::size_t>(worldSize) * sizeof(float);
    const std::size_t recv_bytes = chunk_elements * sizeof(float);

    check_cuda(cudaMemset(send_buf.get(), 0, total_bytes), "cudaMemset send");
    check_cuda(cudaMemset(recv_buf.get(), 0, recv_bytes), "cudaMemset recv");

    bootstrap->barrierAll();
    for (int i = 0; i < config.warmup; ++i) {
      check_nccl(
          ncclReduceScatter(
              send_buf.get(),
              recv_buf.get(),
              chunk_elements,
              ncclFloat,
              ncclSum,
              nccl_comm,
              stream_),
          "ncclReduceScatter warmup");
    }

    CudaEvent start, stop;
    check_cuda(cudaEventRecord(start.get(), stream_), "cudaEventRecord start");
    for (int i = 0; i < config.repeats; ++i) {
      check_nccl(
          ncclReduceScatter(
              send_buf.get(),
              recv_buf.get(),
              chunk_elements,
              ncclFloat,
              ncclSum,
              nccl_comm,
              stream_),
          "ncclReduceScatter timed");
    }
    check_cuda(cudaEventRecord(stop.get(), stream_), "cudaEventRecord stop");
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    bootstrap->barrierAll();

    float total_ms = 0.0f;
    check_cuda(
        cudaEventElapsedTime(&total_ms, start.get(), stop.get()),
        "cudaEventElapsedTime");
    const float avg_ms = total_ms / static_cast<float>(config.repeats);
    return TimedResult{
        .latency_us = avg_ms * 1000.0f,
        .bandwidth_gbps = (total_bytes / 1e9f) / (avg_ms / 1e3f),
    };
  }

  TimedResult run_direct(
      const DirectIbReduceScatterConfig& config,
      const std::vector<P2pIbTransportDevice>& peers,
      DeviceBuffer& send_buf,
      DeviceBuffer& recv_buf) {
    const std::size_t total_elements = config.total_bytes / sizeof(float);
    const std::size_t chunk_elements = total_elements / worldSize;
    const std::size_t total_bytes =
        chunk_elements * static_cast<std::size_t>(worldSize) * sizeof(float);
    const std::size_t recv_bytes = chunk_elements * sizeof(float);

    check_cuda(cudaMemset(send_buf.get(), 0, total_bytes), "cudaMemset send");
    check_cuda(cudaMemset(recv_buf.get(), 0, recv_bytes), "cudaMemset recv");

    DirectReduceScatterIbLaunchParams params{};
    params.my_rank = globalRank;
    params.num_ranks = worldSize;
    params.chunk_elements = chunk_elements;
    params.signaling_data_size = config.signal_bytes;
    params.input = static_cast<const float*>(send_buf.get());
    params.output = static_cast<float*>(recv_buf.get());
    params.num_blocks = config.num_blocks;
    params.timeout_ms = config.timeout_ms;
    params.stream = stream_;
    for (int peer = 0; peer < worldSize; ++peer) {
      if (peer == globalRank) {
        continue;
      }
      params.peers[peer] = peers[peer];
    }

    bootstrap->barrierAll();
    for (int i = 0; i < config.warmup; ++i) {
      launch_direct_reduce_scatter_ib(params);
      check_cuda(cudaStreamSynchronize(stream_), "direct warmup sync");
    }
    bootstrap->barrierAll();

    CudaEvent start, stop;
    check_cuda(cudaEventRecord(start.get(), stream_), "cudaEventRecord start");
    for (int i = 0; i < config.repeats; ++i) {
      launch_direct_reduce_scatter_ib(params);
    }
    check_cuda(cudaEventRecord(stop.get(), stream_), "cudaEventRecord stop");
    check_cuda(cudaStreamSynchronize(stream_), "direct timed sync");
    bootstrap->barrierAll();

    float total_ms = 0.0f;
    check_cuda(
        cudaEventElapsedTime(&total_ms, start.get(), stop.get()),
        "cudaEventElapsedTime");
    const float avg_ms = total_ms / static_cast<float>(config.repeats);
    return TimedResult{
        .latency_us = avg_ms * 1000.0f,
        .bandwidth_gbps = (total_bytes / 1e9f) / (avg_ms / 1e3f),
    };
  }

  static std::vector<std::pair<std::string, std::size_t>> sweep_sizes() {
    return {
        {"1KB", 1UL * 1024},
        {"2KB", 2UL * 1024},
        {"4KB", 4UL * 1024},
        {"8KB", 8UL * 1024},
        {"16KB", 16UL * 1024},
        {"32KB", 32UL * 1024},
        {"64KB", 64UL * 1024},
        {"128KB", 128UL * 1024},
        {"256KB", 256UL * 1024},
        {"512KB", 512UL * 1024},
        {"1MB", 1UL * 1024 * 1024},
        {"2MB", 2UL * 1024 * 1024},
        {"4MB", 4UL * 1024 * 1024},
        {"8MB", 8UL * 1024 * 1024},
        {"16MB", 16UL * 1024 * 1024},
        {"32MB", 32UL * 1024 * 1024},
        {"64MB", 64UL * 1024 * 1024},
        {"128MB", 128UL * 1024 * 1024},
        {"256MB", 256UL * 1024 * 1024},
        {"512MB", 512UL * 1024 * 1024},
        {"1GB", 1UL * 1024 * 1024 * 1024},
        {"2GB", 2UL * 1024 * 1024 * 1024},
    };
  }

  static DirectIbReduceScatterConfig config_for_size(
      DirectIbReduceScatterConfig base_config,
      std::size_t bytes,
      std::size_t signal_bytes_small) {
    base_config.total_bytes = bytes;
    base_config.signal_bytes =
        bytes <= 32UL * 1024 * 1024 ? signal_bytes_small : 0;
    return base_config;
  }

  void print_results(const std::vector<SweepResult>& results) const {
    if (globalRank != 0) {
      return;
    }
    std::stringstream ss;
    ss << "\nDirect IB ReduceScatter comparison sweep\n";
    ss << "GB/s = full per-rank reduce-scatter input bytes / average time\n";
    ss << std::fixed << std::setprecision(2);
    ss << std::setw(8) << "Size" << std::setw(14) << "NCCL Ring"
       << std::setw(14) << "NCCL PAT" << std::setw(14) << "RSDirect"
       << std::setw(18) << "RSDirect/NCCL" << "\n";
    for (const auto& result : results) {
      const float ratio = result.nccl_ring.bandwidth_gbps > 0.0f
          ? result.rsdirect_ib.bandwidth_gbps / result.nccl_ring.bandwidth_gbps
          : 0.0f;
      ss << std::setw(8) << result.size_label << std::setw(14)
         << result.nccl_ring.bandwidth_gbps << std::setw(14)
         << result.nccl_pat.bandwidth_gbps << std::setw(14)
         << result.rsdirect_ib.bandwidth_gbps << std::setw(17) << ratio
         << "x\n";
    }
    XLOG(INFO) << ss.str();
  }

  cudaStream_t stream_{nullptr};
  std::vector<std::unique_ptr<ScopedEnvVar>> nccl_envs_;
};

TEST_F(DirectIbReduceScatterBenchmark, DirectIbVsNcclRingSweep) {
  auto base_config = config_from_env();
  const std::size_t signal_bytes_small = base_config.per_channel_size / 2;
  const auto sizes = sweep_sizes();

  std::size_t max_total_bytes = 0;
  for (const auto& [label, bytes] : sizes) {
    (void)label;
    max_total_bytes = std::max(max_total_bytes, bytes);
  }
  const std::size_t max_total_elements = max_total_bytes / sizeof(float);
  const std::size_t max_chunk_elements = max_total_elements / worldSize;
  max_total_bytes =
      max_chunk_elements * static_cast<std::size_t>(worldSize) * sizeof(float);
  const std::size_t max_recv_bytes = max_chunk_elements * sizeof(float);

  DeviceBuffer nccl_send_buf(max_total_bytes);
  DeviceBuffer nccl_recv_buf(max_recv_bytes);
  DeviceBuffer direct_send_buf(max_total_bytes);
  DeviceBuffer direct_recv_buf(max_recv_bytes);

  if (globalRank == 0) {
    XLOGF(
        INFO,
        "Running direct IB RS sweep blocks={} per_channel={} PD={} "
        "qps_per_connection={} warmup={} repeats={}",
        base_config.num_blocks,
        base_config.per_channel_size,
        base_config.pipeline_depth,
        base_config.qps_per_connection,
        base_config.warmup,
        base_config.repeats);
  }

  std::vector<SweepResult> results;
  results.reserve(sizes.size());
  for (const auto& [label, bytes] : sizes) {
    (void)bytes;
    results.push_back(SweepResult{.size_label = label});
  }

  {
    ScopedEnvVar algo("NCCL_ALGO", "Ring");
    ScopedEnvVar proto("NCCL_PROTO", "Simple");
    auto nccl_comm = make_nccl_comm();
    auto guard = folly::makeGuard([&] { destroy_nccl_comm(nccl_comm); });

    for (std::size_t i = 0; i < sizes.size(); ++i) {
      const auto& [label, bytes] = sizes[i];
      const auto config =
          config_for_size(base_config, bytes, signal_bytes_small);

      if (globalRank == 0) {
        XLOGF(INFO, "Running NCCL Ring size={} bytes={}", label, bytes);
      }
      results[i].nccl_ring =
          run_nccl(config, nccl_comm, nccl_send_buf, nccl_recv_buf);
    }
  }

  {
    ScopedEnvVar pat_enable("NCCL_PAT_ENABLE", "1");
    ScopedEnvVar algo("NCCL_ALGO", "PAT");
    ScopedEnvVar proto("NCCL_PROTO", "Simple");
    auto nccl_comm = make_nccl_comm();
    auto guard = folly::makeGuard([&] { destroy_nccl_comm(nccl_comm); });

    for (std::size_t i = 0; i < sizes.size(); ++i) {
      const auto& [label, bytes] = sizes[i];
      const auto config =
          config_for_size(base_config, bytes, signal_bytes_small);

      if (globalRank == 0) {
        XLOGF(INFO, "Running NCCL PAT size={} bytes={}", label, bytes);
      }
      results[i].nccl_pat =
          run_nccl(config, nccl_comm, nccl_send_buf, nccl_recv_buf);
    }
  }

  auto transport = make_transport(base_config);
  auto peers = get_peer_devices(*transport);

  for (std::size_t i = 0; i < sizes.size(); ++i) {
    const auto& [label, bytes] = sizes[i];
    const auto config = config_for_size(base_config, bytes, signal_bytes_small);

    if (globalRank == 0) {
      XLOGF(
          INFO,
          "Running RSDirect size={} bytes={} signal_bytes={}",
          label,
          bytes,
          config.signal_bytes);
    }

    results[i].rsdirect_ib =
        run_direct(config, peers, direct_send_buf, direct_recv_buf);
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
