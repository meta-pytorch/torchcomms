// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Benchmark for the tile-based point-to-point send/recv collective
// (`sendrecv_tile_kernel` / `sendrecv_tile_compressed_kernel`), modelled
// on AllToAllvTileBenchmark.cc. Ranks are paired (0<->1, 2<->3, ...);
// each even rank sends a contiguous buffer to its odd partner and the
// partner receives it. NCCL point-to-point (ncclSend/ncclRecv) is the
// baseline.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include <cuda.h> // driver API: green contexts for SM partitioning

#include "comms/common/CudaWrap.h"
#include "comms/prims/core/Checks.h"
#include "comms/prims/core/TimeoutUtils.h"
#include "comms/prims/transport/MultiPeerTransport.h"

#include "comms/prims/collectives/SendRecvTile.cuh"
#include "comms/prims/collectives/SendRecvTileCompressed.cuh"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;

// Fail loudly on NCCL errors so a transient failure can't be silently timed and
// reported as a bogus baseline bandwidth.
#define PIPES_NCCL_CHECK(EXPR)                                          \
  do {                                                                  \
    ncclResult_t _pipes_nccl_rc = (EXPR);                               \
    if (_pipes_nccl_rc != ncclSuccess) {                                \
      XLOG(FATAL) << #EXPR                                              \
                  << " failed: " << ncclGetErrorString(_pipes_nccl_rc); \
    }                                                                   \
  } while (0)

namespace comms::prims::benchmark {

namespace {

constexpr int kNIter = 100;
constexpr int kNWarmup = 5;

// Optional thread-block-cluster launch dimension for spreading blocks across
// the H100's 8 GPCs. When clusterDim is in [2, 8] and evenly divides the
// per-launch grid, returns dim3(clusterDim, 1, 1) so launchKernel() uses
// cudaClusterSchedulingPolicySpread to distribute the clusters across all
// GPCs (even GPC memory-bandwidth utilisation, less L2 churn). Returns
// nullopt (standard launch) when disabled (<=1), out of range (>8, the H100
// portable cluster max), or when the grid is not a multiple of the cluster
// size. For fully even GPC coverage the resulting cluster count
// (grid / clusterDim) should be a multiple of 8.
inline std::optional<dim3> clusterDimForGrid(int clusterDim, int gridBlocks) {
  if (clusterDim <= 1) {
    return std::nullopt;
  }
  if (clusterDim > 8) {
    XLOG_FIRST_N(WARN, 1)
        << "[PIPES] PIPES_SENDRECV_BENCH_CLUSTER_DIM=" << clusterDim
        << " exceeds the H100 portable cluster max of 8; using a standard launch";
    return std::nullopt;
  }
  if (gridBlocks % clusterDim != 0) {
    XLOG_FIRST_N(WARN, 1) << "[PIPES] grid (" << gridBlocks
                          << ") not divisible by cluster dim (" << clusterDim
                          << "); using a standard launch";
    return std::nullopt;
  }
  return std::optional<dim3>(dim3(clusterDim, 1, 1));
}

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

// Driver-API error check (runtime calls use PIPES_CUDA_CHECK).
#define PIPES_CU_CHECK(expr)                                                   \
  do {                                                                         \
    CUresult _res = (expr);                                                    \
    if (_res != CUDA_SUCCESS) {                                                \
      const char* _msg = nullptr;                                              \
      cuGetErrorString(_res, &_msg);                                           \
      XLOG(FATAL) << "[SendRecvTileBench] CUDA driver error: " #expr << " -> " \
                  << (_msg ? _msg : "unknown");                                \
    }                                                                          \
  } while (0)

// A stream confined to a green context spanning >= `numSms` SMs of the
// current device. Launching the bench kernels on this stream restricts
// them to that SM partition (e.g. 512 blocks on 64 SMs => 8 blocks/SM).
// `numSms <= 0` returns an empty handle (stream == nullptr) = the default
// stream over the whole GPU. Requires CUDA driver >= 12.4 (R550).
struct GreenCtxStream {
  CUgreenCtx ctx = nullptr;
  cudaStream_t stream = nullptr;
};

GreenCtxStream makeGreenCtxStream(int numSms) {
  GreenCtxStream out;
  if (numSms <= 0) {
    return out;
  }
  int ordinal = 0;
  PIPES_CUDA_CHECK(cudaGetDevice(&ordinal));
  CUdevice dev = 0;
  PIPES_CU_CHECK(cuDeviceGet(&dev, ordinal));
  CUdevResource sm{};
  const CUresult smRes =
      cuDeviceGetDevResource(dev, &sm, CU_DEV_RESOURCE_TYPE_SM);
  if (smRes != CUDA_SUCCESS) {
    const char* nm = nullptr;
    cuGetErrorName(smRes, &nm);
    XLOG(FATAL) << "[SendRecvTileBench] green-context SM partitioning "
                << "unavailable (cuDeviceGetDevResource -> " << (nm ? nm : "?")
                << "). PIPES_SENDRECV_BENCH_NUM_SMS "
                << "requires CUDA driver >= 12.4 (R550).";
  }
  CUdevResource group{};
  CUdevResource remaining{};
  unsigned int nGroups = 1;
  PIPES_CU_CHECK(cuDevSmResourceSplitByCount(
      &group,
      &nGroups,
      &sm,
      &remaining,
      /*useFlags=*/0,
      /*minCount=*/static_cast<unsigned int>(numSms)));
  CUdevResourceDesc desc{};
  PIPES_CU_CHECK(cuDevResourceGenerateDesc(&desc, &group, 1));
  PIPES_CU_CHECK(
      cuGreenCtxCreate(&out.ctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
  CUstream cuStream = nullptr;
  PIPES_CU_CHECK(cuGreenCtxStreamCreate(
      &cuStream, out.ctx, CU_STREAM_NON_BLOCKING, /*priority=*/0));
  out.stream = reinterpret_cast<cudaStream_t>(cuStream);
  return out;
}

// Two DISJOINT green-context partitions, each ~`totalSms/2` SMs, with one
// stream each. Used to physically separate the compress (send-only) and
// decompress (recv-only) kernels onto non-overlapping SM sets so they don't
// contend on the same SMs/L2. `totalSms <= 0` returns an empty handle.
struct SplitGreenCtx {
  CUgreenCtx ctxA = nullptr;
  CUgreenCtx ctxB = nullptr;
  cudaStream_t streamA = nullptr;
  cudaStream_t streamB = nullptr;
};

SplitGreenCtx makeSplitGreenCtxStreams(int totalSms) {
  SplitGreenCtx out;
  if (totalSms <= 0) {
    return out;
  }
  int ordinal = 0;
  PIPES_CUDA_CHECK(cudaGetDevice(&ordinal));
  CUdevice dev = 0;
  PIPES_CU_CHECK(cuDeviceGet(&dev, ordinal));

  CUdevResource sm{};
  const CUresult smRes =
      cuDeviceGetDevResource(dev, &sm, CU_DEV_RESOURCE_TYPE_SM);
  if (smRes != CUDA_SUCCESS) {
    const char* nm = nullptr;
    cuGetErrorName(smRes, &nm);
    XLOG(FATAL) << "[SendRecvTileBench] green-context split unavailable "
                << "(cuDeviceGetDevResource -> " << (nm ? nm : "?")
                << "). Requires CUDA driver >= 12.4 (R550).";
  }

  CUdevResource groups[2]{};
  CUdevResource remaining{};
  unsigned int nGroups = 2;
  const unsigned int perPartition =
      static_cast<unsigned int>(std::max(totalSms / 2, 1));
  PIPES_CU_CHECK(cuDevSmResourceSplitByCount(
      groups, &nGroups, &sm, &remaining, /*useFlags=*/0, perPartition));
  if (nGroups < 2) {
    XLOG(FATAL) << "[SendRecvTileBench] could not split into 2 SM partitions "
                << "of " << perPartition << " SMs (got " << nGroups << ").";
  }

  CUdevResourceDesc descA{};
  CUdevResourceDesc descB{};
  PIPES_CU_CHECK(cuDevResourceGenerateDesc(&descA, &groups[0], 1));
  PIPES_CU_CHECK(cuDevResourceGenerateDesc(&descB, &groups[1], 1));
  PIPES_CU_CHECK(
      cuGreenCtxCreate(&out.ctxA, descA, dev, CU_GREEN_CTX_DEFAULT_STREAM));
  PIPES_CU_CHECK(
      cuGreenCtxCreate(&out.ctxB, descB, dev, CU_GREEN_CTX_DEFAULT_STREAM));

  CUstream a = nullptr;
  CUstream b = nullptr;
  PIPES_CU_CHECK(
      cuGreenCtxStreamCreate(&a, out.ctxA, CU_STREAM_NON_BLOCKING, 0));
  PIPES_CU_CHECK(
      cuGreenCtxStreamCreate(&b, out.ctxB, CU_STREAM_NON_BLOCKING, 0));
  out.streamA = reinterpret_cast<cudaStream_t>(a);
  out.streamB = reinterpret_cast<cudaStream_t>(b);
  return out;
}

class SendRecvTileBenchmarkFixture : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();

    // BenchmarkTestFixture's localRank is sometimes 0 in MPI launches;
    // fall back to OMPI_COMM_WORLD_LOCAL_RANK so each rank pins to its
    // own GPU.
    const char* localRankEnv = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (localRankEnv) {
      localRank = std::atoi(localRankEnv);
    }

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
      PIPES_NCCL_CHECK(ncclGetUniqueId(&id));
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

    MultiPeerTransportConfig transport_config{
        .nvlConfig =
            {
                .dataBufferSize = 8 * 1024 * 1024,
                .chunkSize = 8 * 1024 * 1024,
                .pipelineDepth = 2,
                .maxNumChannels = 32,
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

  // Ranks are paired (0<->1, 2<->3, ...); even ranks send, odd receive.
  int partner() const {
    return globalRank ^ 1;
  }
  bool is_sender() const {
    return (globalRank % 2) == 0;
  }

  struct BenchResult {
    float bw_gbps;
    float lat_us;
    uint64_t comp_uncomp_bytes{0};
    uint64_t comp_comp_bytes{0};
  };

  BenchResult run_nccl(std::size_t bytes, ncclComm_t comm = nullptr) {
    if (!comm) {
      comm = nccl_comm_;
    }
    if (!comm) {
      return {0.0f, 0.0f};
    }
    const int peer = partner();
    const bool send = is_sender();

    DeviceBuffer buf(bytes);
    PIPES_CUDA_CHECK(cudaMemset(buf.get(), send ? 1 : 0, bytes));

    auto do_one = [&]() {
      if (send) {
        PIPES_NCCL_CHECK(
            ncclSend(buf.get(), bytes, ncclChar, peer, comm, stream_));
      } else {
        PIPES_NCCL_CHECK(
            ncclRecv(buf.get(), bytes, ncclChar, peer, comm, stream_));
      }
    };

    bootstrap->barrierAll();
    for (int i = 0; i < kNWarmup; i++) {
      do_one();
    }
    PIPES_CUDA_CHECK(cudaStreamSynchronize(stream_));
    bootstrap->barrierAll();

    CudaEvent start, stop;
    PIPES_CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kNIter; i++) {
      do_one();
    }
    PIPES_CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    PIPES_CUDA_CHECK(cudaStreamSynchronize(stream_));

    float ms = 0;
    PIPES_CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), stop.get()));
    float avg = ms / kNIter;
    float bw = (bytes / 1e9f) / (avg / 1000.0f);
    bootstrap->barrierAll();
    return {bw, avg * 1000.0f};
  }

  BenchResult run_tile(
      MultiPeerDeviceHandle handle,
      std::size_t bytes,
      int num_blocks,
      std::optional<dim3> cluster_dim = std::nullopt,
      std::size_t max_signal_bytes = 0) {
    const int peer = partner();
    const bool send = is_sender();

    DeviceBuffer buf(bytes);
    PIPES_CUDA_CHECK(cudaMemset(buf.get(), send ? 1 : 0, bytes));

    SendRecvTileArgs args{
        .handle = handle,
        .is_send = send,
        .is_recv = !send,
        .send_peer = peer,
        .recv_peer = peer,
        .send_data = send ? static_cast<char*>(buf.get()) : nullptr,
        .send_count = send ? bytes : 0,
        .recv_data = send ? nullptr : static_cast<char*>(buf.get()),
        .recv_count = send ? 0 : bytes,
        .max_signal_bytes = max_signal_bytes,
    };

    bootstrap->barrierAll();
    for (int i = 0; i < kNWarmup; i++) {
      void* ka[] = {&args, &timeout_};
      comms::common::launchKernel(
          (void*)sendrecv_tile_kernel,
          dim3(num_blocks),
          dim3(512),
          ka,
          nullptr,
          cluster_dim);
      PIPES_CUDA_CHECK(cudaDeviceSynchronize());
    }
    bootstrap->barrierAll();

    CudaEvent start, stop;
    PIPES_CUDA_CHECK(cudaEventRecord(start.get()));
    for (int i = 0; i < kNIter; i++) {
      void* ka[] = {&args, &timeout_};
      comms::common::launchKernel(
          (void*)sendrecv_tile_kernel,
          dim3(num_blocks),
          dim3(512),
          ka,
          nullptr,
          cluster_dim);
    }
    PIPES_CUDA_CHECK(cudaEventRecord(stop.get()));
    PIPES_CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0;
    PIPES_CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), stop.get()));
    float avg = ms / kNIter;
    float bw = (bytes / 1e9f) / (avg / 1000.0f);
    bootstrap->barrierAll();
    return {bw, avg * 1000.0f};
  }

  // Bidirectional tile: every rank simultaneously sends to and receives
  // from its partner; the kernel splits the grid half-send / half-recv.
  // Reported BW is single-direction (`bytes / time`).
  BenchResult run_tile_twoway(
      MultiPeerDeviceHandle handle,
      std::size_t bytes,
      int num_blocks,
      std::optional<dim3> cluster_dim = std::nullopt,
      std::size_t max_signal_bytes = 0) {
    const int peer = partner();

    DeviceBuffer send_buf(bytes);
    DeviceBuffer recv_buf(bytes);
    PIPES_CUDA_CHECK(cudaMemset(send_buf.get(), 1, bytes));
    PIPES_CUDA_CHECK(cudaMemset(recv_buf.get(), 0, bytes));

    SendRecvTileArgs args{
        .handle = handle,
        .is_send = true,
        .is_recv = true,
        .send_peer = peer,
        .recv_peer = peer,
        .send_data = static_cast<char*>(send_buf.get()),
        .send_count = bytes,
        .recv_data = static_cast<char*>(recv_buf.get()),
        .recv_count = bytes,
        .max_signal_bytes = max_signal_bytes,
    };

    bootstrap->barrierAll();
    for (int i = 0; i < kNWarmup; i++) {
      void* ka[] = {&args, &timeout_};
      comms::common::launchKernel(
          (void*)sendrecv_tile_kernel,
          dim3(num_blocks),
          dim3(512),
          ka,
          nullptr,
          cluster_dim);
      PIPES_CUDA_CHECK(cudaDeviceSynchronize());
    }
    bootstrap->barrierAll();

    CudaEvent start, stop;
    PIPES_CUDA_CHECK(cudaEventRecord(start.get()));
    for (int i = 0; i < kNIter; i++) {
      void* ka[] = {&args, &timeout_};
      comms::common::launchKernel(
          (void*)sendrecv_tile_kernel,
          dim3(num_blocks),
          dim3(512),
          ka,
          nullptr,
          cluster_dim);
    }
    PIPES_CUDA_CHECK(cudaEventRecord(stop.get()));
    PIPES_CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0;
    PIPES_CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), stop.get()));
    float avg = ms / kNIter;
    float bw = (bytes / 1e9f) / (avg / 1000.0f);
    bootstrap->barrierAll();
    return {bw, avg * 1000.0f};
  }

  // NCCL bidirectional baseline: each rank issues a grouped send+recv to
  // its partner. Reported BW is single-direction (`bytes / time`).
  BenchResult run_nccl_twoway(std::size_t bytes, ncclComm_t comm = nullptr) {
    if (!comm) {
      comm = nccl_comm_;
    }
    if (!comm) {
      return {0.0f, 0.0f};
    }
    const int peer = partner();

    DeviceBuffer send_buf(bytes);
    DeviceBuffer recv_buf(bytes);
    PIPES_CUDA_CHECK(cudaMemset(send_buf.get(), 1, bytes));
    PIPES_CUDA_CHECK(cudaMemset(recv_buf.get(), 0, bytes));

    auto do_one = [&]() {
      PIPES_NCCL_CHECK(ncclGroupStart());
      PIPES_NCCL_CHECK(
          ncclSend(send_buf.get(), bytes, ncclChar, peer, comm, stream_));
      PIPES_NCCL_CHECK(
          ncclRecv(recv_buf.get(), bytes, ncclChar, peer, comm, stream_));
      PIPES_NCCL_CHECK(ncclGroupEnd());
    };

    bootstrap->barrierAll();
    for (int i = 0; i < kNWarmup; i++) {
      do_one();
    }
    PIPES_CUDA_CHECK(cudaStreamSynchronize(stream_));
    bootstrap->barrierAll();

    CudaEvent start, stop;
    PIPES_CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kNIter; i++) {
      do_one();
    }
    PIPES_CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    PIPES_CUDA_CHECK(cudaStreamSynchronize(stream_));

    float ms = 0;
    PIPES_CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), stop.get()));
    float avg = ms / kNIter;
    float bw = (bytes / 1e9f) / (avg / 1000.0f);
    bootstrap->barrierAll();
    return {bw, avg * 1000.0f};
  }

  ncclComm_t nccl_comm_{};
  cudaStream_t stream_{};
  std::unique_ptr<MultiPeerTransport> transport_;
  Timeout timeout_;
};

TEST_F(SendRecvTileBenchmarkFixture, NvlSweep) {
  auto handle = make_handle();

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
  const int blocks_a = 8;
  const int blocks_b = 16;

  char hdr_a[16], hdr_b[16];
  snprintf(hdr_a, sizeof(hdr_a), "Tile-%dblk", blocks_a);
  snprintf(hdr_b, sizeof(hdr_b), "Tile-%dblk", blocks_b);

  if (globalRank == 0) {
    XLOG(INFO) << "\n=== SendRecvTile NVL Sweep ===\n"
               << worldSize << " GPUs (paired), " << kNIter << " iterations\n"
               << "Block counts: " << blocks_a << " vs " << blocks_b << "\n";
    printf(
        "%-10s  %10s  %14s  %14s  %-10s %8s\n",
        "Size",
        "NCCL",
        hdr_a,
        hdr_b,
        "Best",
        "vs NCCL");
    printf(
        "%-10s  %10s  %14s  %14s  %-10s %8s\n",
        "--------",
        "----------",
        "--------------",
        "--------------",
        "----------",
        "--------");
  }

  for (std::size_t bytes : sizes) {
    auto nccl_r = run_nccl(bytes);
    auto ta = run_tile(handle, bytes, blocks_a, clus);
    auto tb = run_tile(handle, bytes, blocks_b, clus);

    if (globalRank == 0) {
      float best_bw = tb.bw_gbps > ta.bw_gbps ? tb.bw_gbps : ta.bw_gbps;
      const char* best_name = tb.bw_gbps > ta.bw_gbps ? hdr_b : hdr_a;
      float speedup = nccl_r.bw_gbps > 0 ? best_bw / nccl_r.bw_gbps : 0;
      char l1[32], l2[32];
      snprintf(l1, sizeof(l1), "%.1f (%d)", ta.bw_gbps, blocks_a);
      snprintf(l2, sizeof(l2), "%.1f (%d)", tb.bw_gbps, blocks_b);
      printf(
          "%-10s  %10.2f  %14s  %14s  %-10s %7.2fx\n",
          format_bytes(bytes).c_str(),
          nccl_r.bw_gbps,
          l1,
          l2,
          best_name,
          speedup);
    }
    bootstrap->barrierAll();
  }
}

TEST_F(SendRecvTileBenchmarkFixture, IbSweep) {
  // Force IB by disabling P2P NVLink in topology discovery.
  MultiPeerTransportConfig ib_config{
      .nvlConfig =
          {
              .dataBufferSize = 8 * 1024 * 1024,
              .chunkSize = 8 * 1024 * 1024,
              .pipelineDepth = 2,
              .maxNumChannels = 32,
          },
      .ibConfig =
          {
              .cudaDevice = localRank,
              .dataBufferSize = 8 * 1024 * 1024,
              .maxGroups = 128,
              .sendRecv =
                  MultipeerIbTransportConfig::SendRecvConfig{
                      .maxGroups = 128,
                      .pipelineDepth = 2,
                  },
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

  const int blocks_a = 64;
  const int blocks_b = 96;

  char hdr_a[16], hdr_b[16];
  snprintf(hdr_a, sizeof(hdr_a), "%dblk", blocks_a);
  snprintf(hdr_b, sizeof(hdr_b), "%dblk", blocks_b);

  if (globalRank == 0) {
    XLOG(INFO) << "\n=== SendRecvTile IB Sweep ===\n"
               << worldSize << " GPUs (paired), " << kNIter << " iterations\n"
               << "Block counts: " << blocks_a << " vs " << blocks_b << "\n"
               << "Run with: NCCL_P2P_DISABLE=1 buck2 run ...\n";
    printf(
        "%-10s  %10s  %14s  %14s  %-10s %8s\n",
        "Size",
        "NCCL-IB",
        hdr_a,
        hdr_b,
        "Best",
        "vs NCCL");
    printf(
        "%-10s  %10s  %14s  %14s  %-10s %8s\n",
        "--------",
        "----------",
        "--------------",
        "--------------",
        "----------",
        "--------");
  }

  for (std::size_t bytes : sizes) {
    auto nccl_r = run_nccl(bytes);
    auto ta = run_tile(ib_handle, bytes, blocks_a);
    auto tb = run_tile(ib_handle, bytes, blocks_b);

    if (globalRank == 0) {
      float best_bw = tb.bw_gbps > ta.bw_gbps ? tb.bw_gbps : ta.bw_gbps;
      const char* best_name = tb.bw_gbps > ta.bw_gbps ? hdr_b : hdr_a;
      float speedup = nccl_r.bw_gbps > 0 ? best_bw / nccl_r.bw_gbps : 0;
      char l1[32], l2[32];
      snprintf(l1, sizeof(l1), "%.1f (%d)", ta.bw_gbps, blocks_a);
      snprintf(l2, sizeof(l2), "%.1f (%d)", tb.bw_gbps, blocks_b);
      printf(
          "%-10s  %10.2f  %14s  %14s  %-10s %7.2fx\n",
          format_bytes(bytes).c_str(),
          nccl_r.bw_gbps,
          l1,
          l2,
          best_name,
          speedup);
    }
    bootstrap->barrierAll();
  }
}

TEST_F(SendRecvTileBenchmarkFixture, IbSweepTwoWay) {
  // Bidirectional IB sweep: every paired rank sends to AND receives from
  // its partner at the same time (the kernel splits the grid half-send /
  // half-recv). Compares against NCCL grouped send+recv. BW is reported
  // per-direction (bytes / time).
  MultiPeerTransportConfig ib_config{
      .nvlConfig =
          {
              .dataBufferSize = 8 * 1024 * 1024,
              .chunkSize = 8 * 1024 * 1024,
              .pipelineDepth = 2,
              .maxNumChannels = 32,
          },
      .ibConfig =
          {
              .cudaDevice = localRank,
              .dataBufferSize = 8 * 1024 * 1024,
              .maxGroups = 128,
              .sendRecv =
                  MultipeerIbTransportConfig::SendRecvConfig{
                      .maxGroups = 128,
                      .pipelineDepth = 2,
                  },
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
                 << "\nSkipping IB two-way benchmark (requires multi-node "
                    "or IB loopback support)";
    }
    return;
  }
  auto ib_handle = ib_transport->get_device_handle();

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

  // Total grid blocks; split in half inside the kernel (so each direction
  // gets blocks_*/2). Both even.
  const int blocks_a = 64;
  const int blocks_b = 128;

  char hdr_a[16], hdr_b[16];
  snprintf(hdr_a, sizeof(hdr_a), "%dblk", blocks_a);
  snprintf(hdr_b, sizeof(hdr_b), "%dblk", blocks_b);

  if (globalRank == 0) {
    XLOG(INFO) << "\n=== SendRecvTile IB Two-Way Sweep ===\n"
               << worldSize << " GPUs (paired, bidirectional), " << kNIter
               << " iterations\n"
               << "Total block counts (split half send/half recv): " << blocks_a
               << " vs " << blocks_b << "\n"
               << "Per-direction BW. Run with: NCCL_P2P_DISABLE=1 buck2 "
                  "run ...\n";
    printf(
        "%-10s  %10s  %14s  %14s  %-10s %8s\n",
        "Size",
        "NCCL-IB",
        hdr_a,
        hdr_b,
        "Best",
        "vs NCCL");
    printf(
        "%-10s  %10s  %14s  %14s  %-10s %8s\n",
        "--------",
        "----------",
        "--------------",
        "--------------",
        "----------",
        "--------");
  }

  for (std::size_t bytes : sizes) {
    auto nccl_r = run_nccl_twoway(bytes);
    auto ta = run_tile_twoway(ib_handle, bytes, blocks_a);
    auto tb = run_tile_twoway(ib_handle, bytes, blocks_b);

    if (globalRank == 0) {
      float best_bw = tb.bw_gbps > ta.bw_gbps ? tb.bw_gbps : ta.bw_gbps;
      const char* best_name = tb.bw_gbps > ta.bw_gbps ? hdr_b : hdr_a;
      float speedup = nccl_r.bw_gbps > 0 ? best_bw / nccl_r.bw_gbps : 0;
      char l1[32], l2[32];
      snprintf(l1, sizeof(l1), "%.1f (%d)", ta.bw_gbps, blocks_a);
      snprintf(l2, sizeof(l2), "%.1f (%d)", tb.bw_gbps, blocks_b);
      printf(
          "%-10s  %10.2f  %14s  %14s  %-10s %7.2fx\n",
          format_bytes(bytes).c_str(),
          nccl_r.bw_gbps,
          l1,
          l2,
          best_name,
          speedup);
    }
    bootstrap->barrierAll();
  }
}

TEST_F(SendRecvTileBenchmarkFixture, IbSweepCompressed) {
  // ANS-compressed IB sweep. Mirrors AllToAllvTileBenchmark's
  // IbSweepCompressed config (128 MiB data buffer, 32 QPs/peer) and
  // unconditionally launches `sendrecv_tile_compressed_kernel`.
  //
  // Launch geometry overridable at runtime (no rebuild):
  //   PIPES_SENDRECV_BENCH_BLOCKS            grid blocks      (default 256)
  //   PIPES_SENDRECV_BENCH_MIN_BLOCKS_PER_SM launch_bounds m  (default 2; 8
  //                                          -> 100% occ at 256 threads)
  //   PIPES_SENDRECV_BENCH_NUM_SMS           green-ctx SM limit (default 0 =
  //                                          whole GPU; needs driver >= 12.4)
  //   PIPES_SENDRECV_BENCH_CLUSTER_DIM       blocks/cluster (default 0 = off).
  //                                          2..8 spreads clusters across the
  //                                          H100's 8 GPCs (spread policy).
  //   PIPES_SENDRECV_BENCH_PLAIN_PCT         integer percent [0,100] of
  //                                          blocks that use plain Memcpy
  //                                          instead of ANS (default 0 =
  //                                          all-ANS). Plain blocks fill NIC
  //                                          bandwidth the ANS blocks leave
  //                                          idle when compRatio >> bus gain.
  auto env_int = [](const char* name, int fallback) {
    const char* v = std::getenv(name);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : fallback;
  };
  const int kBenchBlocks = env_int("PIPES_SENDRECV_BENCH_BLOCKS", 256);
  const int kBenchMinBlocksPerSM =
      env_int("PIPES_SENDRECV_BENCH_MIN_BLOCKS_PER_SM", 2);
  const int kBenchNumSms = env_int("PIPES_SENDRECV_BENCH_NUM_SMS", 0);
  // Spread blocks across the H100's 8 GPCs via cluster launch (0 = off).
  const int kBenchClusterDim = env_int("PIPES_SENDRECV_BENCH_CLUSTER_DIM", 0);
  // Fraction of blocks that use plain Memcpy instead of ANS (see header).
  const int kBenchPlainPct =
      std::clamp(env_int("PIPES_SENDRECV_BENCH_PLAIN_PCT", 0), 0, 100);
  const float kBenchPlainFraction = kBenchPlainPct / 100.0f;

  // One-way path: every block is one direction's active block. Round the
  // IB staging buffer DOWN to a multiple of 512 * active_blocks so the
  // per-block slot (dataBufferSize / active_blocks) is 512-aligned; a
  // mixed plain/ANS launch then computes an identical perBlockSlot for
  // both block types (variable-size masks with ~511, fixed-size ~15), so
  // their staging offsets never overlap. See the transport slot math in
  // P2pIbgdaTransportDevice.cuh.
  const std::size_t kSlotQuantum =
      512ULL * static_cast<std::size_t>(std::max(kBenchBlocks, 1));
  std::size_t ibDataBufferSize = std::max<std::size_t>(
      128ULL * 1024 * 1024,
      static_cast<std::size_t>(kBenchBlocks) * 512ULL * 1024);
  ibDataBufferSize = (ibDataBufferSize / kSlotQuantum) * kSlotQuantum;
  ASSERT_GT(ibDataBufferSize, 0u);
  MultiPeerTransportConfig ib_config{
      .nvlConfig =
          {
              .dataBufferSize = 8 * 1024 * 1024,
              .chunkSize = 8 * 1024 * 1024,
              .pipelineDepth = 2,
              .maxNumChannels = 32,
          },
      .ibConfig =
          {
              .cudaDevice = localRank,
              .dataBufferSize = ibDataBufferSize,
              .maxGroups = std::max(kBenchBlocks, 256),
              .sendRecv =
                  MultipeerIbTransportConfig::SendRecvConfig{
                      // Cover the configured grid block count on the one-way
                      // path (active_blocks == grid_blocks there).
                      .maxGroups = std::max(kBenchBlocks, 256),
                      .pipelineDepth = 2,
                  },
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
                 << "\nSkipping IB compressed benchmark (requires "
                    "multi-node or IB loopback support)";
    }
    return;
  }
  auto ib_handle = ib_transport->get_device_handle();

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

  const int grid_blocks = kBenchBlocks;

  constexpr std::size_t kCompressChunkBytes = AnsCompressor::kMaxUncompBytes;
  constexpr int kNumWarpsPerBlock = 8; // blockDim.x / 32 = 256 / 32 = 8
  constexpr int kNumThreadsPerBlock = kNumWarpsPerBlock * 32;

  // Optionally confine the kernel to a green context of kBenchNumSms SMs
  // (e.g. 512 blocks on 64 SMs => 8 blocks/SM). nullptr stream = whole GPU.
  const GreenCtxStream gctx = makeGreenCtxStream(kBenchNumSms);
  const cudaStream_t benchStream = gctx.stream;

  const bool send = is_sender();
  const int peer = partner();

  if (globalRank == 0) {
    XLOG(INFO) << "\n=== SendRecvTile IB Compressed Sweep ===\n"
               << worldSize << " GPUs (paired), " << kNIter << " iterations\n"
               << grid_blocks << " blocks x " << kNumThreadsPerBlock
               << " threads, min_blocks_per_sm=" << kBenchMinBlocksPerSM
               << ", num_sms=" << kBenchNumSms
               << (kBenchNumSms > 0 ? " (green-ctx)" : " (full GPU)") << "\n"
               << "plain_block_pct=" << kBenchPlainPct
               << "% (Memcpy blocks), ibDataBufferSize="
               << format_bytes(ibDataBufferSize) << "\n"
               << "Sparsity sweep: 0% → 100% in 10% steps "
               << "(zero-byte fraction of input buffer)\n"
               << "Run with: NCCL_P2P_DISABLE=1 buck2 run ... "
               << "-- --gtest_filter='*IbSweepCompressed*'\n";
  }

  // Per-block 16-byte-aligned scratch for the ANS-compressed send path.
  DeviceBuffer aligned_aux_buf(
      static_cast<std::size_t>(grid_blocks) * kCompressChunkBytes);

  // Host-side non-compressible pattern (SplitMix64 hash per byte), tiled
  // into the send buffer; the leading N% is then zeroed to control
  // sparsity. See AllToAllvTileBenchmark.cc for the rationale.
  constexpr std::size_t kPatternBytes = 1 * 1024 * 1024;
  std::vector<uint8_t> host_pattern(kPatternBytes);
  for (std::size_t i = 0; i < kPatternBytes; ++i) {
    uint64_t x = static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL +
        0xDEADBEEFCAFEBABEULL;
    x = (x ^ (x >> 33)) * 0xff51afd7ed558ccdULL;
    x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53ULL;
    x = x ^ (x >> 33);
    host_pattern[i] = static_cast<uint8_t>(x & 0xff);
  }

  // Pre-cache NCCL results per size (NCCL doesn't depend on contents).
  std::vector<BenchResult> nccl_results;
  nccl_results.reserve(sizes.size());
  for (std::size_t bytes : sizes) {
    nccl_results.push_back(run_nccl(bytes));
  }

  for (int sparsityPct = 0; sparsityPct <= 100; sparsityPct += 10) {
    if (globalRank == 0) {
      XLOG(INFO) << "\n--- Sparsity " << sparsityPct
                 << "% (zero-byte fraction of input buffer) ---";
      printf(
          "%-10s  %10s  %14s  %-10s %8s  %12s\n",
          "Size",
          "NCCL-IB",
          "Tile",
          "Best",
          "vs NCCL",
          "compRatio");
      printf(
          "%-10s  %10s  %14s  %-10s %8s  %12s\n",
          "--------",
          "----------",
          "--------------",
          "----------",
          "--------",
          "------------");
    }

    std::size_t bppIdx = 0;
    for (std::size_t bytes : sizes) {
      auto nccl_r = nccl_results[bppIdx++];

      DeviceBuffer buf(bytes);
      if (send) {
        // Tile the host pattern, then zero the leading sparsityPct%.
        std::size_t off = 0;
        while (off < bytes) {
          const std::size_t copy = std::min(kPatternBytes, bytes - off);
          PIPES_CUDA_CHECK(cudaMemcpy(
              static_cast<char*>(buf.get()) + off,
              host_pattern.data(),
              copy,
              cudaMemcpyHostToDevice));
          off += copy;
        }
        const std::size_t zero_bytes =
            (bytes * static_cast<std::size_t>(sparsityPct)) / 100;
        if (zero_bytes > 0) {
          PIPES_CUDA_CHECK(cudaMemset(buf.get(), 0, zero_bytes));
        }
      } else {
        PIPES_CUDA_CHECK(cudaMemset(buf.get(), 0, bytes));
      }

      SendRecvTileArgs args{
          .handle = ib_handle,
          .is_send = send,
          .is_recv = !send,
          .send_peer = peer,
          .recv_peer = peer,
          .send_data = send ? static_cast<char*>(buf.get()) : nullptr,
          .send_count = send ? bytes : 0,
          .recv_data = send ? nullptr : static_cast<char*>(buf.get()),
          .recv_count = send ? 0 : bytes,
          .max_signal_bytes = 0,
          .aligned_aux_buf = static_cast<char*>(aligned_aux_buf.get()),
          .plain_block_fraction = kBenchPlainFraction,
      };

      // constexpr dim3 kH100ClusterDim{2, 1, 1};
      auto launch_one = [&](void** ka) {
        // m=8 (100% occupancy) is instantiated only for NumWarps=8; the
        // default callers use the m=2 instantiation.
        void* kfn = (kBenchMinBlocksPerSM == 8)
            ? reinterpret_cast<void*>(sendrecv_tile_compressed_kernel<
                                      AnsCompressor,
                                      kNumWarpsPerBlock,
                                      8>)
            : reinterpret_cast<void*>(sendrecv_tile_compressed_kernel<
                                      AnsCompressor,
                                      kNumWarpsPerBlock,
                                      2>);
        return comms::common::launchKernel(
            kfn,
            dim3(grid_blocks),
            dim3(kNumThreadsPerBlock),
            ka,
            /*stream=*/benchStream,
            clusterDimForGrid(kBenchClusterDim, grid_blocks));
      };

      bootstrap->barrierAll();
      for (int i = 0; i < kNWarmup; i++) {
        void* ka[] = {&args, &timeout_};
        const cudaError_t lerr = launch_one(ka);
        if (lerr != cudaSuccess) {
          XLOG(FATAL) << "[PIPES] FATAL: warmup launch err=" << lerr << " ("
                      << cudaGetErrorString(lerr) << ") bytes=" << bytes
                      << " iter=" << i;
        }
        PIPES_CUDA_CHECK(cudaDeviceSynchronize());
      }
      bootstrap->barrierAll();

      // Reset compression counters AFTER warmup so the ratio reflects
      // only timed iterations.
      (void)fetch_and_reset_sendrecv_ans_compress_stats();

      CudaEvent start, stop;
      PIPES_CUDA_CHECK(cudaEventRecord(start.get(), benchStream));
      for (int i = 0; i < kNIter; i++) {
        void* ka[] = {&args, &timeout_};
        const cudaError_t lerr = launch_one(ka);
        if (lerr != cudaSuccess) {
          XLOG(FATAL) << "[PIPES] FATAL: timed launch err=" << lerr << " ("
                      << cudaGetErrorString(lerr) << ") bytes=" << bytes
                      << " iter=" << i;
        }
      }
      PIPES_CUDA_CHECK(cudaEventRecord(stop.get(), benchStream));
      PIPES_CUDA_CHECK(cudaDeviceSynchronize());

      SendRecvAnsCompressStats compStats =
          fetch_and_reset_sendrecv_ans_compress_stats();

      float ms = 0;
      PIPES_CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), stop.get()));
      float avg = ms / kNIter;
      float bw = (bytes / 1e9f) / (avg / 1000.0f);
      bootstrap->barrierAll();
      BenchResult tile_r{
          bw,
          avg * 1000.0f,
          compStats.uncompressed_bytes,
          compStats.compressed_bytes};

      if (globalRank == 0) {
        const char* best_name =
            tile_r.bw_gbps > nccl_r.bw_gbps ? "tile" : "nccl";
        const float speedup =
            nccl_r.bw_gbps > 0 ? tile_r.bw_gbps / nccl_r.bw_gbps : 0;

        char l1[32], comp_ratio_buf[64];
        snprintf(l1, sizeof(l1), "%.1f (%d)", tile_r.bw_gbps, grid_blocks);
        if (tile_r.comp_comp_bytes > 0) {
          const double ratio = static_cast<double>(tile_r.comp_uncomp_bytes) /
              static_cast<double>(tile_r.comp_comp_bytes);
          snprintf(comp_ratio_buf, sizeof(comp_ratio_buf), "%.2fx", ratio);
        } else {
          snprintf(
              comp_ratio_buf,
              sizeof(comp_ratio_buf),
              "1(compression skipped due to small size)");
        }
        printf(
            "%-10s  %10.2f  %14s  %-10s %7.2fx  %12s\n",
            format_bytes(bytes).c_str(),
            nccl_r.bw_gbps,
            l1,
            best_name,
            speedup,
            comp_ratio_buf);
      }
      bootstrap->barrierAll();
    }
  }

  if (gctx.ctx != nullptr) {
    PIPES_CU_CHECK(cuStreamDestroy(reinterpret_cast<CUstream>(benchStream)));
    PIPES_CU_CHECK(cuGreenCtxDestroy(gctx.ctx));
  }
}

TEST_F(SendRecvTileBenchmarkFixture, IbSweepCompressedTwoWay) {
  // Bidirectional ANS-compressed IB sweep: every paired rank compresses+
  // sends to AND receives+decompresses from its partner at the same time
  // (grid split half-send / half-recv). Per-direction BW vs NCCL grouped
  // send+recv.
  //
  // Launch geometry overridable via PIPES_SENDRECV_BENCH_{BLOCKS,
  // MIN_BLOCKS_PER_SM,NUM_SMS,CLUSTER_DIM} (see IbSweepCompressed). Two-way
  // splits the grid in half, so per-direction active blocks = BLOCKS/2.
  // CLUSTER_DIM (2..8) spreads clusters across the H100's 8 GPCs; it applies
  // to each split-mode direction kernel and the two-way launch.
  //   PIPES_SENDRECV_BENCH_PLAIN_PCT  integer percent [0,100] of each
  //     direction's blocks that use plain Memcpy instead of ANS (default 0
  //     = all-ANS). Plain blocks fill NIC bandwidth the ANS blocks leave
  //     idle when compRatio >> realized bus gain.
  auto env_int = [](const char* name, int fallback) {
    const char* v = std::getenv(name);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : fallback;
  };
  const int kBenchBlocks = env_int("PIPES_SENDRECV_BENCH_BLOCKS", 256);
  const int kBenchMinBlocksPerSM =
      env_int("PIPES_SENDRECV_BENCH_MIN_BLOCKS_PER_SM", 2);
  const int kBenchNumSms = env_int("PIPES_SENDRECV_BENCH_NUM_SMS", 0);
  // When set (and NUM_SMS>0), run compress (send-only) and decompress
  // (recv-only) as two concurrent kernels on two DISJOINT green-context SM
  // partitions (each NUM_SMS/2) instead of one fused half-grid kernel.
  const int kBenchSplitSms = env_int("PIPES_SENDRECV_BENCH_SPLIT_SMS", 0);
  // Spread blocks across the H100's 8 GPCs via cluster launch (0 = off).
  const int kBenchClusterDim = env_int("PIPES_SENDRECV_BENCH_CLUSTER_DIM", 0);
  const bool kSplit = kBenchSplitSms != 0 && kBenchNumSms > 0;
  // Fraction of each direction's blocks that use plain Memcpy (see header).
  const int kBenchPlainPct =
      std::clamp(env_int("PIPES_SENDRECV_BENCH_PLAIN_PCT", 0), 0, 100);
  const float kBenchPlainFraction = kBenchPlainPct / 100.0f;
  // Override the IB staging buffer size (bytes; per direction) for the
  // buffer-vs-message-size study. 0 = formula default.
  const std::size_t kBenchDataBufBytes =
      static_cast<std::size_t>(
          env_int("PIPES_SENDRECV_BENCH_DATA_BUFFER_MB", 0)) *
      1024ULL * 1024ULL;
  // Per-sub-chunk signal hint (bytes). 0 = one signal per slot fill (couples
  // signal granularity to dataBufferSize/active_blocks); non-zero decouples it.
  // Must be <= the per-block slot or the kernel overruns it (launch failure).
  const std::size_t kBenchMaxSignal = static_cast<std::size_t>(
      env_int("PIPES_SENDRECV_BENCH_MAX_SIGNAL_BYTES", 0));

  // Per-direction active blocks (both split and fused two-way give each
  // direction BLOCKS/2). Round the IB staging buffer DOWN to a multiple of
  // 512 * active_blocks so the per-block slot is 512-aligned and a mixed
  // plain/ANS launch computes an identical perBlockSlot for both block
  // types (see one-way path / transport slot math).
  const int kDirBlocks = std::max(kBenchBlocks / 2, 1);
  const std::size_t kSlotQuantum =
      512ULL * static_cast<std::size_t>(kDirBlocks);
  std::size_t ibDataBufferSize = kBenchDataBufBytes > 0
      ? kBenchDataBufBytes
      : std::max<std::size_t>(
            128ULL * 1024 * 1024,
            static_cast<std::size_t>(kDirBlocks) * 512ULL * 1024);
  ibDataBufferSize = (ibDataBufferSize / kSlotQuantum) * kSlotQuantum;
  ASSERT_GT(ibDataBufferSize, 0u);
  MultiPeerTransportConfig ib_config{
      .nvlConfig =
          {
              .dataBufferSize = 8 * 1024 * 1024,
              .chunkSize = 8 * 1024 * 1024,
              .pipelineDepth = 2,
              .maxNumChannels = 32,
          },
      .ibConfig =
          {
              .cudaDevice = localRank,
              .dataBufferSize = ibDataBufferSize,
              .maxGroups = std::max(kBenchBlocks / 2, 256),
              .sendRecv =
                  MultipeerIbTransportConfig::SendRecvConfig{
                      // Two-way splits the grid: active blocks per direction
                      // == grid_blocks/2.
                      .maxGroups = std::max(kBenchBlocks / 2, 256),
                      .pipelineDepth = 2,
                  },
              // 256 groups => 256 QPs/peer/NIC, which exceeds the 128-QP eager
              // exchange wire format; materialize the peer's QPs lazily
              // instead. SendRecv uses getP2pTransportDevice() per peer, so it
              // is unaffected by lazy mode (unlike DeviceWindow/AllToAllv).
              .ibLazyConnect = true,
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
                 << "\nSkipping IB compressed two-way benchmark (requires "
                    "multi-node or IB loopback support)";
    }
    return;
  }
  // Lazy mode: materialize this peer's QPs and get a handle scoped to it.
  // Plain get_device_handle() throws under ibLazyConnect.
  auto ib_handle = ib_transport->get_device_handle({partner()});

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

  // Total grid blocks; split in half (each direction gets grid_blocks/2).
  const int grid_blocks = kBenchBlocks;

  constexpr std::size_t kCompressChunkBytes = AnsCompressor::kMaxUncompBytes;
  constexpr int kNumWarpsPerBlock = 8;
  constexpr int kNumThreadsPerBlock = kNumWarpsPerBlock * 32;

  // Single fused partition (kBenchNumSms SMs) unless split mode requests two
  // disjoint partitions for send vs recv.
  const GreenCtxStream gctx =
      kSplit ? GreenCtxStream{} : makeGreenCtxStream(kBenchNumSms);
  const SplitGreenCtx sgctx =
      kSplit ? makeSplitGreenCtxStreams(kBenchNumSms) : SplitGreenCtx{};
  const cudaStream_t benchStream = gctx.stream;
  const cudaStream_t sendStream = sgctx.streamA;
  const cudaStream_t recvStream = sgctx.streamB;

  const int peer = partner();

  if (globalRank == 0) {
    XLOG(INFO) << "\n=== SendRecvTile IB Compressed Two-Way Sweep ===\n"
               << worldSize << " GPUs (paired, bidirectional), " << kNIter
               << " iterations\n"
               << grid_blocks << " total blocks (split half send/half recv), "
               << kNumThreadsPerBlock
               << " threads, min_blocks_per_sm=" << kBenchMinBlocksPerSM
               << ", num_sms=" << kBenchNumSms
               << (kSplit ? " (green-ctx SPLIT send|recv)"
                          : (kBenchNumSms > 0 ? " (green-ctx fused)"
                                              : " (full GPU)"))
               << "\n"
               << "plain_block_pct=" << kBenchPlainPct
               << "% (Memcpy blocks), ibDataBufferSize="
               << format_bytes(ibDataBufferSize) << "\n"
               << "Sparsity sweep: 0% → 100% in 10% steps "
               << "(zero-byte fraction of input buffer)\n"
               << "Per-direction BW. Run with: NCCL_P2P_DISABLE=1 buck2 run "
               << "... -- --gtest_filter='*IbSweepCompressedTwoWay*'\n";
  }

  // Per-block 16-byte-aligned scratch for the ANS-compressed send path
  // (one slice per grid block, keyed on global blockIdx.x).
  DeviceBuffer aligned_aux_buf(
      static_cast<std::size_t>(grid_blocks) * kCompressChunkBytes);

  // Host-side non-compressible pattern (SplitMix64 hash per byte).
  constexpr std::size_t kPatternBytes = 1 * 1024 * 1024;
  std::vector<uint8_t> host_pattern(kPatternBytes);
  for (std::size_t i = 0; i < kPatternBytes; ++i) {
    uint64_t x = static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL +
        0xDEADBEEFCAFEBABEULL;
    x = (x ^ (x >> 33)) * 0xff51afd7ed558ccdULL;
    x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53ULL;
    x = x ^ (x >> 33);
    host_pattern[i] = static_cast<uint8_t>(x & 0xff);
  }

  std::vector<BenchResult> nccl_results;
  nccl_results.reserve(sizes.size());
  for (std::size_t bytes : sizes) {
    nccl_results.push_back(run_nccl_twoway(bytes));
  }

  for (int sparsityPct = 0; sparsityPct <= 100; sparsityPct += 10) {
    if (globalRank == 0) {
      XLOG(INFO) << "\n--- Sparsity " << sparsityPct
                 << "% (zero-byte fraction of input buffer) ---";
      printf(
          "%-10s  %10s  %14s  %-10s %8s  %12s\n",
          "Size",
          "NCCL-IB",
          "Tile",
          "Best",
          "vs NCCL",
          "compRatio");
      printf(
          "%-10s  %10s  %14s  %-10s %8s  %12s\n",
          "--------",
          "----------",
          "--------------",
          "----------",
          "--------",
          "------------");
    }

    std::size_t bppIdx = 0;
    for (std::size_t bytes : sizes) {
      auto nccl_r = nccl_results[bppIdx++];

      // Every rank both sends (compressible pattern, sparsified) and
      // receives (into a zeroed buffer).
      DeviceBuffer send_buf(bytes);
      DeviceBuffer recv_buf(bytes);
      {
        std::size_t off = 0;
        while (off < bytes) {
          const std::size_t copy = std::min(kPatternBytes, bytes - off);
          PIPES_CUDA_CHECK(cudaMemcpy(
              static_cast<char*>(send_buf.get()) + off,
              host_pattern.data(),
              copy,
              cudaMemcpyHostToDevice));
          off += copy;
        }
        const std::size_t zero_bytes =
            (bytes * static_cast<std::size_t>(sparsityPct)) / 100;
        if (zero_bytes > 0) {
          PIPES_CUDA_CHECK(cudaMemset(send_buf.get(), 0, zero_bytes));
        }
      }
      PIPES_CUDA_CHECK(cudaMemset(recv_buf.get(), 0, bytes));

      // Kernel function pointer for the configured launch_bounds m.
      void* kfn = (kBenchMinBlocksPerSM == 8)
          ? reinterpret_cast<void*>(sendrecv_tile_compressed_kernel<
                                    AnsCompressor,
                                    kNumWarpsPerBlock,
                                    8>)
          : reinterpret_cast<void*>(sendrecv_tile_compressed_kernel<
                                    AnsCompressor,
                                    kNumWarpsPerBlock,
                                    2>);

      float ms = 0;
      SendRecvAnsCompressStats compStats{};

      if (kSplit) {
        // Compress (send-only) on partition A, decompress (recv-only) on
        // partition B — two concurrent kernels on disjoint SM sets. Each
        // direction uses grid_blocks/2 blocks (matching the fused per-
        // direction count) so blocks/SM is unchanged.
        const int dirBlocks = std::max(grid_blocks / 2, 1);
        SendRecvTileArgs sArgs{
            .handle = ib_handle,
            .is_send = true,
            .is_recv = false,
            .send_peer = peer,
            .recv_peer = peer,
            .send_data = static_cast<char*>(send_buf.get()),
            .send_count = bytes,
            .recv_data = nullptr,
            .recv_count = 0,
            .max_signal_bytes = kBenchMaxSignal,
            .aligned_aux_buf = static_cast<char*>(aligned_aux_buf.get()),
            .plain_block_fraction = kBenchPlainFraction,
        };
        SendRecvTileArgs rArgs{
            .handle = ib_handle,
            .is_send = false,
            .is_recv = true,
            .send_peer = peer,
            .recv_peer = peer,
            .send_data = nullptr,
            .send_count = 0,
            .recv_data = static_cast<char*>(recv_buf.get()),
            .recv_count = bytes,
            .max_signal_bytes = kBenchMaxSignal,
            .aligned_aux_buf = static_cast<char*>(aligned_aux_buf.get()),
            .plain_block_fraction = kBenchPlainFraction,
        };
        auto launch_send = [&]() {
          void* ka[] = {&sArgs, &timeout_};
          return comms::common::launchKernel(
              kfn,
              dim3(dirBlocks),
              dim3(kNumThreadsPerBlock),
              ka,
              /*stream=*/sendStream,
              clusterDimForGrid(kBenchClusterDim, dirBlocks));
        };
        auto launch_recv = [&]() {
          void* ka[] = {&rArgs, &timeout_};
          return comms::common::launchKernel(
              kfn,
              dim3(dirBlocks),
              dim3(kNumThreadsPerBlock),
              ka,
              /*stream=*/recvStream,
              clusterDimForGrid(kBenchClusterDim, dirBlocks));
        };

        bootstrap->barrierAll();
        for (int i = 0; i < kNWarmup; i++) {
          PIPES_CUDA_CHECK(launch_send());
          PIPES_CUDA_CHECK(launch_recv());
          PIPES_CUDA_CHECK(cudaDeviceSynchronize());
        }
        bootstrap->barrierAll();

        (void)fetch_and_reset_sendrecv_ans_compress_stats();

        CudaEvent start, stopSend, stopRecv;
        PIPES_CUDA_CHECK(cudaEventRecord(start.get(), sendStream));
        // Both partitions begin at the same point.
        PIPES_CUDA_CHECK(cudaStreamWaitEvent(recvStream, start.get(), 0));
        for (int i = 0; i < kNIter; i++) {
          PIPES_CUDA_CHECK(launch_send());
          PIPES_CUDA_CHECK(launch_recv());
        }
        PIPES_CUDA_CHECK(cudaEventRecord(stopSend.get(), sendStream));
        PIPES_CUDA_CHECK(cudaEventRecord(stopRecv.get(), recvStream));
        PIPES_CUDA_CHECK(cudaDeviceSynchronize());
        compStats = fetch_and_reset_sendrecv_ans_compress_stats();

        float msSend = 0;
        float msRecv = 0;
        PIPES_CUDA_CHECK(
            cudaEventElapsedTime(&msSend, start.get(), stopSend.get()));
        PIPES_CUDA_CHECK(
            cudaEventElapsedTime(&msRecv, start.get(), stopRecv.get()));
        ms = std::max(msSend, msRecv);
      } else {
        SendRecvTileArgs args{
            .handle = ib_handle,
            .is_send = true,
            .is_recv = true,
            .send_peer = peer,
            .recv_peer = peer,
            .send_data = static_cast<char*>(send_buf.get()),
            .send_count = bytes,
            .recv_data = static_cast<char*>(recv_buf.get()),
            .recv_count = bytes,
            .max_signal_bytes = kBenchMaxSignal,
            .aligned_aux_buf = static_cast<char*>(aligned_aux_buf.get()),
            .plain_block_fraction = kBenchPlainFraction,
        };
        auto launch_one = [&](void** ka) {
          return comms::common::launchKernel(
              kfn,
              dim3(grid_blocks),
              dim3(kNumThreadsPerBlock),
              ka,
              /*stream=*/benchStream,
              clusterDimForGrid(kBenchClusterDim, grid_blocks));
        };

        bootstrap->barrierAll();
        for (int i = 0; i < kNWarmup; i++) {
          void* ka[] = {&args, &timeout_};
          const cudaError_t lerr = launch_one(ka);
          if (lerr != cudaSuccess) {
            XLOG(FATAL) << "[PIPES] FATAL: warmup launch err=" << lerr << " ("
                        << cudaGetErrorString(lerr) << ") bytes=" << bytes
                        << " iter=" << i;
          }
          PIPES_CUDA_CHECK(cudaDeviceSynchronize());
        }
        bootstrap->barrierAll();

        (void)fetch_and_reset_sendrecv_ans_compress_stats();

        CudaEvent start, stop;
        PIPES_CUDA_CHECK(cudaEventRecord(start.get(), benchStream));
        for (int i = 0; i < kNIter; i++) {
          void* ka[] = {&args, &timeout_};
          const cudaError_t lerr = launch_one(ka);
          if (lerr != cudaSuccess) {
            XLOG(FATAL) << "[PIPES] FATAL: timed launch err=" << lerr << " ("
                        << cudaGetErrorString(lerr) << ") bytes=" << bytes
                        << " iter=" << i;
          }
        }
        PIPES_CUDA_CHECK(cudaEventRecord(stop.get(), benchStream));
        PIPES_CUDA_CHECK(cudaDeviceSynchronize());
        compStats = fetch_and_reset_sendrecv_ans_compress_stats();
        PIPES_CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), stop.get()));
      }
      float avg = ms / kNIter;
      float bw = (bytes / 1e9f) / (avg / 1000.0f);
      bootstrap->barrierAll();
      BenchResult tile_r{
          bw,
          avg * 1000.0f,
          compStats.uncompressed_bytes,
          compStats.compressed_bytes};

      if (globalRank == 0) {
        const char* best_name =
            tile_r.bw_gbps > nccl_r.bw_gbps ? "tile" : "nccl";
        const float speedup =
            nccl_r.bw_gbps > 0 ? tile_r.bw_gbps / nccl_r.bw_gbps : 0;

        char l1[32], comp_ratio_buf[64];
        snprintf(l1, sizeof(l1), "%.1f (%d)", tile_r.bw_gbps, grid_blocks);
        if (tile_r.comp_comp_bytes > 0) {
          const double ratio = static_cast<double>(tile_r.comp_uncomp_bytes) /
              static_cast<double>(tile_r.comp_comp_bytes);
          snprintf(comp_ratio_buf, sizeof(comp_ratio_buf), "%.2fx", ratio);
        } else {
          snprintf(
              comp_ratio_buf,
              sizeof(comp_ratio_buf),
              "1(compression skipped due to small size)");
        }
        printf(
            "%-10s  %10.2f  %14s  %-10s %7.2fx  %12s\n",
            format_bytes(bytes).c_str(),
            nccl_r.bw_gbps,
            l1,
            best_name,
            speedup,
            comp_ratio_buf);
      }
      bootstrap->barrierAll();
    }
  }

  if (gctx.ctx != nullptr) {
    PIPES_CU_CHECK(cuStreamDestroy(reinterpret_cast<CUstream>(benchStream)));
    PIPES_CU_CHECK(cuGreenCtxDestroy(gctx.ctx));
  }
  if (sgctx.ctxA != nullptr) {
    PIPES_CU_CHECK(cuStreamDestroy(reinterpret_cast<CUstream>(sendStream)));
    PIPES_CU_CHECK(cuStreamDestroy(reinterpret_cast<CUstream>(recvStream)));
    PIPES_CU_CHECK(cuGreenCtxDestroy(sgctx.ctxA));
    PIPES_CU_CHECK(cuGreenCtxDestroy(sgctx.ctxB));
  }
}

} // namespace

} // namespace comms::prims::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::BenchmarkEnvironment());
  return RUN_ALL_TESTS();
}
