// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>
#include <vector>

#include "comms/common/CudaWrap.h"
#include "comms/pipes/Checks.h"
#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/TimeoutUtils.h"

#include "comms/pipes/collectives/AllToAllvTile.cuh"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

namespace {

constexpr int kNIter = 100;
constexpr int kNWarmup = 5;

std::string format_bytes(std::size_t bytes) {
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

class AllToAllvTileBenchmarkFixture : public meta::comms::BenchmarkTestFixture {
 protected:
  static std::size_t get_data_buffer_size() {
    int major = 0;
    PIPES_CUDA_CHECK(
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    // GB200 (SM 100+): 32MB * 2 pd = 64MB staging; H100: 8MB * 2 pd = 16MB
    return major >= 10 ? 32 * 1024 * 1024 : 8 * 1024 * 1024;
  }

  static int get_max_blocks() {
    int major = 0;
    PIPES_CUDA_CHECK(
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    // Match NCCL's SM usage: 64 on GB200, 32 on H100
    return major >= 10 ? 64 : 32;
  }

  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    PIPES_CUDA_CHECK(cudaSetDevice(localRank));
    PIPES_CUDA_CHECK(cudaStreamCreate(&stream_));

    // Match NCCL IB config to our transport settings for fair comparison.
    setenv("NCCL_NCHANNELS_PER_NET_PEER", "1", 0);
    setenv("NCCL_IB_QPS_PER_CONNECTION", "2", 0);
    setenv("NCCL_IB_SPLIT_DATA_ON_QPS", "1", 0);
    setenv("NCCL_BUFFSIZE", "8388608", 0);
    setenv("NCCL_P2P_NET_CHUNKSIZE", "524288", 0);

    ncclUniqueId id;
    if (globalRank == 0) {
      ncclGetUniqueId(&id);
    }
    std::vector<ncclUniqueId> all_ids(worldSize);
    all_ids[globalRank] = id;
    bootstrap
        ->allGather(all_ids.data(), sizeof(ncclUniqueId), globalRank, worldSize)
        .get();
    auto ncclRet =
        ncclCommInitRank(&nccl_comm_, worldSize, all_ids[0], globalRank);
    if (ncclRet != ncclSuccess) {
      XLOG(WARNING) << "ncclCommInitRank failed (rc=" << ncclRet
                    << "), NCCL baseline will be skipped";
      nccl_comm_ = nullptr;
    }

    std::size_t bufSize = get_data_buffer_size();
    MultiPeerTransportConfig transport_config{
        .nvlConfig =
            {
                .dataBufferSize = bufSize,
                .chunkSize = bufSize,
                .pipelineDepth = 2,
                .tile_max_groups = 32,
            },
    };
    transport_ = std::make_unique<MultiPeerTransport>(
        globalRank, worldSize, localRank, bootstrap, transport_config);
    transport_->exchange();

    int device = 0;
    PIPES_CUDA_CHECK(cudaGetDevice(&device));
    timeout_ = makeTimeout(0, device);
  }

  MultiPeerDeviceHandle make_handle() {
    return transport_->get_device_handle();
  }

  void TearDown() override {
    transport_.reset();
    if (nccl_comm_) {
      ncclCommDestroy(nccl_comm_);
    }
    PIPES_CUDA_CHECK(cudaStreamDestroy(stream_));
    BenchmarkTestFixture::TearDown();
  }

  struct BenchResult {
    float bw_gbps;
    float lat_us;
  };

  BenchResult run_nccl(std::size_t bpp, ncclComm_t comm = nullptr) {
    if (!comm) {
      comm = nccl_comm_;
    }
    if (!comm) {
      return {0.0f, 0.0f};
    }
    const int nranks = worldSize;
    const std::size_t total = bpp * nranks;

    DeviceBuffer send_buf(total);
    DeviceBuffer recv_buf(total);
    PIPES_CUDA_CHECK(cudaMemset(send_buf.get(), 1, total));
    PIPES_CUDA_CHECK(cudaMemset(recv_buf.get(), 0, total));

    std::vector<size_t> counts(nranks, bpp);
    counts[globalRank] = 0;
    std::vector<size_t> displs(nranks);
    for (int i = 0; i < nranks; i++) {
      displs[i] = i * bpp;
    }

    bootstrap->barrierAll();
    for (int i = 0; i < kNWarmup; i++) {
      ncclAllToAllv(
          send_buf.get(),
          counts.data(),
          displs.data(),
          recv_buf.get(),
          counts.data(),
          displs.data(),
          ncclChar,
          comm,
          stream_);
    }
    PIPES_CUDA_CHECK(cudaStreamSynchronize(stream_));
    bootstrap->barrierAll();

    CudaEvent start, stop;
    PIPES_CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kNIter; i++) {
      ncclAllToAllv(
          send_buf.get(),
          counts.data(),
          displs.data(),
          recv_buf.get(),
          counts.data(),
          displs.data(),
          ncclChar,
          comm,
          stream_);
    }
    PIPES_CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    PIPES_CUDA_CHECK(cudaStreamSynchronize(stream_));

    float ms = 0;
    PIPES_CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), stop.get()));
    float avg = ms / kNIter;
    std::size_t algo_bytes = bpp * (nranks - 1);
    float bw = (algo_bytes / 1e9f) / (avg / 1000.0f);
    bootstrap->barrierAll();
    return {bw, avg * 1000.0f};
  }

  BenchResult run_tile(
      MultiPeerDeviceHandle handle,
      std::size_t bpp,
      int num_blocks,
      void* kernel_fn,
      int grid_blocks,
      std::size_t max_signal_bytes = 0,
      std::optional<dim3> cluster_dim = std::nullopt,
      int num_blocks_ib_override = -1) {
    const int nranks = worldSize;
    const std::size_t total = bpp * nranks;

    DeviceBuffer send_buf(total);
    DeviceBuffer recv_buf(total);
    PIPES_CUDA_CHECK(cudaMemset(send_buf.get(), 1, total));
    PIPES_CUDA_CHECK(cudaMemset(recv_buf.get(), 0, total));

    std::vector<char*> h_sp(nranks), h_rp(nranks);
    std::vector<std::size_t> h_c(nranks, bpp);
    h_c[globalRank] = 0;
    for (int r = 0; r < nranks; r++) {
      h_sp[r] = static_cast<char*>(send_buf.get()) + r * bpp;
      h_rp[r] = static_cast<char*>(recv_buf.get()) + r * bpp;
    }

    DeviceBuffer d_sp(nranks * sizeof(char*));
    DeviceBuffer d_rp(nranks * sizeof(char*));
    DeviceBuffer d_c(nranks * sizeof(std::size_t));
    PIPES_CUDA_CHECK(cudaMemcpy(
        d_sp.get(),
        h_sp.data(),
        nranks * sizeof(char*),
        cudaMemcpyHostToDevice));
    PIPES_CUDA_CHECK(cudaMemcpy(
        d_rp.get(),
        h_rp.data(),
        nranks * sizeof(char*),
        cudaMemcpyHostToDevice));
    PIPES_CUDA_CHECK(cudaMemcpy(
        d_c.get(),
        h_c.data(),
        nranks * sizeof(std::size_t),
        cudaMemcpyHostToDevice));

    AllToAllvTileArgs args{
        .handle = handle,
        .send_ptrs = static_cast<char**>(d_sp.get()),
        .send_counts = static_cast<std::size_t*>(d_c.get()),
        .recv_ptrs = static_cast<char**>(d_rp.get()),
        .recv_counts = static_cast<std::size_t*>(d_c.get()),
        .num_blocks_nvl = num_blocks,
        .num_blocks_ib =
            num_blocks_ib_override >= 0 ? num_blocks_ib_override : num_blocks,
        .max_signal_bytes = max_signal_bytes,
    };

    bootstrap->barrierAll();
    for (int i = 0; i < kNWarmup; i++) {
      void* ka[] = {&args, &timeout_};
      comms::common::launchKernel(
          kernel_fn, dim3(grid_blocks), dim3(512), ka, nullptr, cluster_dim);
      PIPES_CUDA_CHECK(cudaDeviceSynchronize());
    }
    bootstrap->barrierAll();

    CudaEvent start, stop;
    PIPES_CUDA_CHECK(cudaEventRecord(start.get()));
    for (int i = 0; i < kNIter; i++) {
      void* ka[] = {&args, &timeout_};
      comms::common::launchKernel(
          kernel_fn, dim3(grid_blocks), dim3(512), ka, nullptr, cluster_dim);
    }
    PIPES_CUDA_CHECK(cudaEventRecord(stop.get()));
    PIPES_CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0;
    PIPES_CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), stop.get()));
    float avg = ms / kNIter;
    std::size_t algo_bytes = bpp * (nranks - 1);
    float bw = (algo_bytes / 1e9f) / (avg / 1000.0f);
    bootstrap->barrierAll();
    return {bw, avg * 1000.0f};
  }

  ncclComm_t nccl_comm_{};
  cudaStream_t stream_{};
  std::unique_ptr<MultiPeerTransport> transport_;
  Timeout timeout_;
};

TEST_F(AllToAllvTileBenchmarkFixture, FullSweep) {
  auto handle = make_handle();
  constexpr dim3 kClusterDim{4, 1, 1};

  // Per-size block counts matching NCCL channel scaling:
  //   NCCL reduces channels while nBytes < nc * 512 * 64
  //   8 channels for < 512KB, 16 for 512KB-128MB, 32 for 512MB+
  struct SizeConfig {
    std::size_t bpp;
    int blocks;
  };
  std::vector<SizeConfig> configs = {
      {8 * 1024, 16},
      {32 * 1024, 16},
      {128 * 1024, 16},
      {512 * 1024, 16},
      {1024 * 1024, 16},
      {4 * 1024 * 1024, 16},
      {16 * 1024 * 1024, 16},
      {64 * 1024 * 1024, 16},
      {256 * 1024 * 1024, 16},
      {1024UL * 1024 * 1024, 32},
  };

  if (globalRank == 0) {
    XLOG(INFO) << "\n=== AllToAllvTile NVL Sweep (1D vs 2D, cluster=4) ===\n"
               << worldSize << " GPUs, " << kNIter
               << " iterations, per-size block counts\n";
    printf(
        "%-10s  %10s  %12s  %12s  %-12s %8s\n",
        "Per-Peer",
        "NCCL",
        "Tile-1D",
        "Tile-2D",
        "Best",
        "vs NCCL");
    printf(
        "%-10s  %10s  %12s  %12s  %-12s %8s\n",
        "--------",
        "----------",
        "------------",
        "------------",
        "------------",
        "--------");
  }

  for (const auto& cfg : configs) {
    int grid = cfg.blocks;
    auto nccl_r = run_nccl(cfg.bpp);

    auto t1d_r = run_tile(
        handle,
        cfg.bpp,
        grid,
        (void*)alltoallv_tile_1d_kernel,
        grid,
        0,
        kClusterDim,
        0);

    auto t2d_r = run_tile(
        handle,
        cfg.bpp,
        grid,
        (void*)alltoallv_tile_2d_kernel,
        grid,
        0,
        kClusterDim,
        0);

    if (globalRank == 0) {
      float best_bw = t1d_r.bw_gbps;
      const char* best_name = "tile_1d";
      if (t2d_r.bw_gbps > best_bw) {
        best_bw = t2d_r.bw_gbps;
        best_name = "tile_2d";
      }
      float speedup = nccl_r.bw_gbps > 0 ? best_bw / nccl_r.bw_gbps : 0;
      char l1[32], l2[32];
      snprintf(l1, sizeof(l1), "%.1f (%d)", t1d_r.bw_gbps, grid);
      snprintf(l2, sizeof(l2), "%.1f (%d)", t2d_r.bw_gbps, grid);
      printf(
          "%-10s  %10.2f  %12s  %12s  %-12s %7.2fx\n",
          format_bytes(cfg.bpp).c_str(),
          nccl_r.bw_gbps,
          l1,
          l2,
          best_name,
          speedup);
    }
    bootstrap->barrierAll();
  }
}

TEST_F(AllToAllvTileBenchmarkFixture, IbSweep) {
  constexpr dim3 kClusterDim{4, 1, 1};
  // Force IB by disabling P2P NVLink in topology discovery.
  std::size_t bufSize = get_data_buffer_size();
  MultiPeerTransportConfig ib_config{
      .nvlConfig =
          {
              .dataBufferSize = bufSize,
              .chunkSize = bufSize,
              .pipelineDepth = 2,
              .tile_max_groups = 32,
          },
      .ibgdaConfig =
          {
              .cudaDevice = localRank,
              .dataBufferSize = bufSize,
              .sendRecv =
                  MultipeerIbgdaTransportConfig::SendRecvConfig{
                      .maxGroups = 8,
                      .pipelineDepth = 2,
                  },
              .numQpsPerPeerPerNic = 4,
          },
      .topoConfig =
          {
              .p2pDisable = true,
          },
  };

  std::unique_ptr<MultiPeerTransport> ib_transport;
  try {
    ib_transport = std::make_unique<MultiPeerTransport>(
        globalRank, worldSize, localRank, bootstrap, ib_config);
    ib_transport->exchange();
  } catch (const std::exception& e) {
    if (globalRank == 0) {
      XLOG(WARN) << "IB transport setup failed: " << e.what()
                 << "\nSkipping IB benchmark (requires multi-node or IB "
                    "loopback support)";
    }
    return;
  }
  auto ib_handle = ib_transport->get_device_handle();

  // Use the existing NCCL comm. When NCCL_P2P_DISABLE=1 is set in the
  // environment before the process starts, NCCL will also use IB transport,
  // making the comparison fair. Run with:
  //   NCCL_P2P_DISABLE=1 buck2 run ... -- --gtest_filter="*IbFullSweep*"

  std::vector<std::size_t> sizes = {
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

  int numIbPeers = ib_handle.numIbPeers;
  int ib_full = 8;
  int ib_half = ib_full / 2;

  if (globalRank == 0) {
    XLOG(INFO) << "\n=== AllToAllvTile IB Sweep (cluster=4) ===\n"
               << worldSize << " GPUs, " << kNIter << " iterations\n"
               << numIbPeers << " IB peers\n"
               << "Run with: NCCL_P2P_DISABLE=1 buck2 run ...\n";
    printf(
        "%-10s  %10s  %12s  %12s  %-10s %8s\n",
        "Per-Peer",
        "NCCL-IB",
        "half",
        "full",
        "Best",
        "vs NCCL");
    printf(
        "%-10s  %10s  %12s  %12s  %-10s %8s\n",
        "--------",
        "----------",
        "------------",
        "------------",
        "----------",
        "--------");
  }

  for (std::size_t bpp : sizes) {
    auto nccl_r = run_nccl(bpp);

    auto ib1_r = run_tile(
        ib_handle,
        bpp,
        0,
        (void*)alltoallv_tile_1d_kernel,
        2 * ib_half,
        0,
        kClusterDim,
        ib_half);

    auto ib2_r = run_tile(
        ib_handle,
        bpp,
        0,
        (void*)alltoallv_tile_1d_kernel,
        2 * ib_full,
        0,
        kClusterDim,
        ib_full);

    if (globalRank == 0) {
      float best_bw =
          ib2_r.bw_gbps > ib1_r.bw_gbps ? ib2_r.bw_gbps : ib1_r.bw_gbps;
      const char* best_name = ib2_r.bw_gbps > ib1_r.bw_gbps ? "full" : "half";
      float speedup = nccl_r.bw_gbps > 0 ? best_bw / nccl_r.bw_gbps : 0;

      char l1[32], l2[32];
      snprintf(l1, sizeof(l1), "%.1f (%d)", ib1_r.bw_gbps, ib_half);
      snprintf(l2, sizeof(l2), "%.1f (%d)", ib2_r.bw_gbps, ib_full);
      printf(
          "%-10s  %10.2f  %12s  %12s  %-10s %7.2fx\n",
          format_bytes(bpp).c_str(),
          nccl_r.bw_gbps,
          l1,
          l2,
          best_name,
          speedup);
    }
    bootstrap->barrierAll();
  }
}

} // namespace

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::BenchmarkEnvironment());
  return RUN_ALL_TESTS();
}
