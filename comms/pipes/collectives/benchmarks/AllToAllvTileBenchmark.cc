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
    ncclCommInitRank(&nccl_comm_, worldSize, all_ids[0], globalRank);

    MultiPeerTransportConfig transport_config{
        .nvlConfig =
            {
                .dataBufferSize = 8 * 1024 * 1024,
                .chunkSize = 8 * 1024 * 1024,
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
  int numPeers = worldSize - 1;

  std::vector<std::size_t> sizes = {
      8 * 1024,
      32 * 1024,
      128 * 1024,
      512 * 1024,
      1024 * 1024,
      4 * 1024 * 1024,
      16 * 1024 * 1024,
      64 * 1024 * 1024,
      256 * 1024 * 1024,
      1024UL * 1024 * 1024,
  };

  dim3 clus(comms::common::kDefaultClusterSize, 1, 1);
  int bpp = 4;
  int nvl_blocks = numPeers * bpp;
  int grid = 2 * nvl_blocks;

  if (globalRank == 0) {
    XLOG(INFO) << "\n=== AllToAllvTile NVL Sweep (1D vs 2D) ===\n"
               << worldSize << " GPUs, " << kNIter << " iterations, " << grid
               << " total blocks\n";
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

  for (std::size_t bpp_size : sizes) {
    auto nccl_r = run_nccl(bpp_size);

    auto t1d_r = run_tile(
        handle,
        bpp_size,
        nvl_blocks,
        (void*)alltoallv_tile_1d_kernel,
        grid,
        0,
        clus,
        0);

    auto t2d_r = run_tile(
        handle,
        bpp_size,
        nvl_blocks,
        (void*)alltoallv_tile_2d_kernel,
        grid,
        0,
        clus,
        0);

    if (globalRank == 0) {
      float best_bw =
          t2d_r.bw_gbps > t1d_r.bw_gbps ? t2d_r.bw_gbps : t1d_r.bw_gbps;
      const char* best_name =
          t2d_r.bw_gbps > t1d_r.bw_gbps ? "tile_2d" : "tile_1d";
      float speedup = nccl_r.bw_gbps > 0 ? best_bw / nccl_r.bw_gbps : 0;
      char l1[32], l2[32];
      snprintf(l1, sizeof(l1), "%.1f (%d)", t1d_r.bw_gbps, grid);
      snprintf(l2, sizeof(l2), "%.1f (%d)", t2d_r.bw_gbps, grid);
      printf(
          "%-10s  %10.2f  %12s  %12s  %-12s %7.2fx\n",
          format_bytes(bpp_size).c_str(),
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
  // Force IB by disabling P2P NVLink in topology discovery.
  MultiPeerTransportConfig ib_config{
      .nvlConfig =
          {
              .dataBufferSize = 8 * 1024 * 1024,
              .chunkSize = 8 * 1024 * 1024,
              .pipelineDepth = 2,
              .tile_max_groups = 32,
          },
      .ibgdaConfig =
          {
              .cudaDevice = localRank,
              .dataBufferSize = 8 * 1024 * 1024,
              .sendRecv =
                  MultipeerIbgdaTransportConfig::SendRecvConfig{
                      .maxGroups = 32,
                      .pipelineDepth = 2,
                  },
              .numQpsPerPeer = 2,
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

  // IB configs: 1 block/peer, 2 blocks/peer (matching NCCL's 2 QPs)
  int kIb1bpp = numIbPeers; // 1 block per peer
  int kIb2bpp = 2 * numIbPeers; // 2 blocks per peer
  int grid_1bpp = 2 * kIb1bpp;
  int grid_2bpp = 2 * kIb2bpp;

  if (globalRank == 0) {
    XLOG(INFO) << "\n=== AllToAllvTile IB Sweep ===\n"
               << worldSize << " GPUs, " << kNIter << " iterations\n"
               << numIbPeers << " IB peers\n"
               << "Run with: NCCL_P2P_DISABLE=1 buck2 run ...\n";
    printf(
        "%-10s  %10s  %12s  %12s  %-10s %8s\n",
        "Per-Peer",
        "NCCL-IB",
        "1blk/peer",
        "2blk/peer",
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

    // 1 block per IB peer
    auto ib1_r = run_tile(
        ib_handle,
        bpp,
        0,
        (void*)alltoallv_tile_1d_kernel,
        grid_1bpp,
        0,
        std::nullopt,
        kIb1bpp);

    // 2 blocks per IB peer (matching NCCL's 2 QPs)
    auto ib2_r = run_tile(
        ib_handle,
        bpp,
        0,
        (void*)alltoallv_tile_1d_kernel,
        grid_2bpp,
        0,
        std::nullopt,
        kIb2bpp);

    if (globalRank == 0) {
      float best_bw =
          ib2_r.bw_gbps > ib1_r.bw_gbps ? ib2_r.bw_gbps : ib1_r.bw_gbps;
      const char* best_name =
          ib2_r.bw_gbps > ib1_r.bw_gbps ? "2blk/peer" : "1blk/peer";
      float speedup = nccl_r.bw_gbps > 0 ? best_bw / nccl_r.bw_gbps : 0;

      char l1[32], l2[32];
      snprintf(l1, sizeof(l1), "%.1f (%d)", ib1_r.bw_gbps, grid_1bpp);
      snprintf(l2, sizeof(l2), "%.1f (%d)", ib2_r.bw_gbps, grid_2bpp);
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
