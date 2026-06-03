// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>

#include <iomanip>
#include <sstream>
#include <vector>

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
  std::string name;
};

struct RingAllReduceBenchmarkResult {
  std::string test_name;
  std::size_t total_elements{};
  std::size_t total_bytes{};
  int num_rings{};
  float nccl_bandwidth{};
  float ring_bandwidth{};
  float nccl_latency{};
  float ring_latency{};
  float speedup{};
};

class RingAllReduceBenchmarkFixture : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    CUDA_CHECK_VOID(cudaSetDevice(localRank));

    setenv("NCCL_P2P_DISABLE", "1", 1);
    NCCL_CHECK_VOID(
        ncclCommInitRank(&nccl_comm_, worldSize, get_nccl_id(), globalRank));
    CUDA_CHECK_VOID(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    NCCL_CHECK_VOID(ncclCommDestroy(nccl_comm_));
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
          nccl_comm_,
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
          nccl_comm_,
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

    if (config.num_rings <= 0 ||
        config.num_rings > RingAllReduceLaunchParams::kMaxRings) {
      XLOGF(ERR, "Unsupported num_rings={}", config.num_rings);
      latency_us = 0.0f;
      return 0.0f;
    }

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
    ss << "================================================================================\n";
    ss << "         NCCL AllReduce vs Ring AllReduce (Pipes IBGDA) Benchmark\n";
    ss << "================================================================================\n";
    ss << std::left << std::setw(14) << "Test" << std::right << std::setw(10)
       << "Size" << std::right << std::setw(6) << "Rings" << std::right
       << std::setw(10) << "NCCL" << std::right << std::setw(10) << "Ring"
       << std::right << std::setw(10) << "Speedup" << std::right
       << std::setw(12) << "NCCL Lat" << std::right << std::setw(12)
       << "Ring Lat\n";
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
         << std::fixed << std::setprecision(2) << r.nccl_bandwidth << std::right
         << std::setw(10) << std::fixed << std::setprecision(2)
         << r.ring_bandwidth << std::right << std::setw(9) << std::fixed
         << std::setprecision(2) << r.speedup << "x" << std::right
         << std::setw(12) << std::fixed << std::setprecision(1)
         << r.nccl_latency << std::right << std::setw(12) << std::fixed
         << std::setprecision(1) << r.ring_latency << "\n";
    }

    ss << "================================================================================\n";
    ss << worldSize << " ranks, total_elements = per-rank input/output size\n";
    ss << "================================================================================\n";

    XLOG(INFO) << ss.str();
  }

  ncclComm_t nccl_comm_{};
  cudaStream_t stream_{};
};

TEST_F(RingAllReduceBenchmarkFixture, VsNccl) {
  if (globalRank == 0) {
    XLOG(INFO) << "\n=== Ring AllReduce (Pipes IBGDA) vs NCCL Comparison ===\n";
  }

  std::size_t kDataBufferSize = 32UL * 1024 * 1024;
  int kNumQps = 4;

  std::vector<RingAllReduceBenchmarkConfig> configs = {
      {.total_elements = 64 * 1024,
       .num_blocks = 8,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "256KB_8B"},
      {.total_elements = 256 * 1024,
       .num_blocks = 16,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "1MB_16B"},
      {.total_elements = 1024 * 1024,
       .num_blocks = 16,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "4MB_16B"},
      {.total_elements = 4 * 1024 * 1024,
       .num_blocks = 16,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "16MB_16B"},
      {.total_elements = 16 * 1024 * 1024,
       .num_blocks = 16,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "64MB_16B"},
      {.total_elements = 32 * 1024 * 1024,
       .num_blocks = 16,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "128MB_16B"},
      {.total_elements = 64 * 1024 * 1024,
       .num_blocks = 32,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "256MB_32B"},
      {.total_elements = 128 * 1024 * 1024,
       .num_blocks = 32,
       .num_rings = 1,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "512MB_32B"},
      {.total_elements = 1024 * 1024,
       .num_blocks = 16,
       .num_rings = 2,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "4MB_16B_2R"},
      {.total_elements = 16 * 1024 * 1024,
       .num_blocks = 16,
       .num_rings = 2,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "64MB_16B_2R"},
      {.total_elements = 64 * 1024 * 1024,
       .num_blocks = 32,
       .num_rings = 2,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "256MB_32B_2R"},
      {.total_elements = 128 * 1024 * 1024,
       .num_blocks = 32,
       .num_rings = 2,
       .data_buffer_size = kDataBufferSize,
       .pipeline_depth = 2,
       .num_qps = kNumQps,
       .name = "512MB_32B_2R"},
  };

  std::vector<RingAllReduceBenchmarkResult> results;

  for (const auto& config : configs) {
    float nccl_lat = 0.0f;
    float nccl_bw = run_nccl_benchmark(config, nccl_lat);

    float ring_lat = 0.0f;
    float ring_bw = run_ring_benchmark(config, ring_lat);

    if (globalRank == 0) {
      results.push_back({
          .test_name = config.name,
          .total_elements = config.total_elements,
          .total_bytes = config.total_elements * sizeof(float),
          .num_rings = config.num_rings,
          .nccl_bandwidth = nccl_bw,
          .ring_bandwidth = ring_bw,
          .nccl_latency = nccl_lat,
          .ring_latency = ring_lat,
          .speedup = (nccl_bw > 0) ? ring_bw / nccl_bw : 0.0f,
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
