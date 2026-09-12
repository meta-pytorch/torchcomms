// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Benchmark for the tile-based point-to-point send/recv collective
// (`sendrecv_tile_kernel` / `sendrecv_tile_compressed_kernel`), modelled
// on AllToAllvTileBenchmark.cc. Ranks are paired (0<->1, 2<->3, ...);
// each even rank sends a contiguous buffer to its odd partner and the
// partner receives it. NCCL point-to-point (ncclSend/ncclRecv) is the
// baseline.

#include <folly/init/Init.h>
#include <nccl.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>
#include "comms/utils/logger/SpdlogLogger.h"

#include <cuda.h> // driver API: green contexts for SM partitioning

#include "comms/common/CudaWrap.h"
#include "comms/prims/core/AbortCheck.cuh"
#include "comms/prims/core/Checks.h"
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
#define PIPES_NCCL_CHECK(EXPR)                                           \
  do {                                                                   \
    ncclResult_t _pipes_nccl_rc = (EXPR);                                \
    if (_pipes_nccl_rc != ncclSuccess) {                                 \
      COMMS_LOG_STREAM(FATAL)                                            \
          << #EXPR << " failed: " << ncclGetErrorString(_pipes_nccl_rc); \
    }                                                                    \
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
    COMMS_LOG_STREAM_FIRST_N(WARN, 1)
        << "[PIPES] PIPES_SENDRECV_BENCH_CLUSTER_DIM=" << clusterDim
        << " exceeds the H100 portable cluster max of 8; using a standard launch";
    return std::nullopt;
  }
  if (gridBlocks % clusterDim != 0) {
    COMMS_LOG_STREAM_FIRST_N(WARN, 1)
        << "[PIPES] grid (" << gridBlocks << ") not divisible by cluster dim ("
        << clusterDim << "); using a standard launch";
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

// Render the requested vs actually-usable `max_signal_bytes` for a run banner.
//
// `PIPES_SENDRECV_BENCH_MAX_SIGNAL_BYTES` is a count of LOGICAL input bytes,
// but the transport slot has to hold the worst-case ENCODED ANS stride for that
// input -- roughly 1.3x expansion plus a per-chunk size-header table plus
// 16-byte alignment. So "request <= perBlockSlot" is not the safety condition,
// and a request that satisfies it can still exceed the slot once encoded.
//
// `default_max_signal_bytes_for_compress()` clamps such a request down to
// `CopyOp::max_safe_chunk_size_for_slot(perBlockSlot)` rather than letting the
// transport trap on it. That happens device-side, so a clamped run used to look
// identical to an honoured one in the output. Print enough to tell them apart.
//
// This is a `.cc`: `max_safe_chunk_size_for_slot` is `__host__ __device__` but
// lives behind `PIPES_ENABLE_ANS_COMPRESSION && __CUDACC__` in CopyOp.cuh, so
// it cannot be called from here. Report the slot instead -- the clamp is a
// function of it, so the reader can see when a request is in clamp territory.
std::string describe_max_signal(
    std::size_t requested,
    std::size_t perChannelSize,
    int pipelineDepth) {
  // The slot is perChannelSize / pipelineDepth, 512-byte aligned -- the same
  // expression `P2pIbgdaTransportDevice::pipeline_chunk()` uses. It is NOT
  // dataBufferSize / activeBlocks: an earlier version of this helper used that
  // and over-reported the slot by 16x, which is the divergence balajib flagged
  // on the dispatcher. Two places computing this differently is how a chunk
  // that looks safe here traps in the transport.
  const std::size_t perBlockSlot = pipelineDepth > 0
      ? ((perChannelSize / static_cast<std::size_t>(pipelineDepth)) & ~511ULL)
      : 0;
  std::ostringstream os;
  os << "max_signal_bytes=";
  if (requested == 0) {
    os << "0 (auto: largest chunk whose worst-case encoded form fits the slot)";
  } else {
    os << requested << " requested";
    if (requested > perBlockSlot) {
      os << " -- EXCEEDS the slot, will be CLAMPED";
    } else {
      os << " (clamped device-side if its encoded form exceeds the slot)";
    }
  }
  os << ", perBlockSlot=" << perBlockSlot << " (" << format_bytes(perBlockSlot)
     << ")\n";
  return os.str();
}

// Driver-API error check (runtime calls use PIPES_CUDA_CHECK).
#define PIPES_CU_CHECK(expr)                                                   \
  do {                                                                         \
    CUresult _res = (expr);                                                    \
    if (_res != CUDA_SUCCESS) {                                                \
      const char* _msg = nullptr;                                              \
      cuGetErrorString(_res, &_msg);                                           \
      COMMS_LOG_STREAM(FATAL)                                                  \
          << "[SendRecvTileVariantsBench] CUDA driver error: " #expr << " -> " \
          << (_msg ? _msg : "unknown");                                        \
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
    COMMS_LOG_STREAM(FATAL)
        << "[SendRecvTileVariantsBench] green-context SM partitioning "
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
    COMMS_LOG_STREAM(FATAL)
        << "[SendRecvTileVariantsBench] green-context split unavailable "
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
    COMMS_LOG_STREAM(FATAL)
        << "[SendRecvTileVariantsBench] could not split into 2 SM partitions "
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

class SendRecvTileVariantsBenchmarkFixture
    : public meta::comms::BenchmarkTestFixture {
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

    // Constrain NCCL to roughly our transport's IB shape. `setenv(..., 0)`
    // does NOT overwrite, so an operator-supplied value wins -- which is
    // intended, but it means the values below are requests, not guarantees.
    // Everything actually in effect is echoed back after this block so a run
    // can be compared against another one later.
    setenv("NCCL_NCHANNELS_PER_NET_PEER", "1", 0);
    setenv("NCCL_IB_QPS_PER_CONNECTION", "2", 0);
    setenv("NCCL_IB_SPLIT_DATA_ON_QPS", "1", 0);
    setenv("NCCL_BUFFSIZE", "8388608", 0);
    setenv("NCCL_P2P_NET_CHUNKSIZE", "524288", 0);

    if (globalRank == 0) {
      auto env_or = [](const char* k, const char* dflt) -> std::string {
        const char* v = std::getenv(k);
        return v != nullptr ? std::string(v) : std::string(dflt);
      };
      // `nccl_comm_` is built ONCE from process-global NCCL settings and then
      // reused by every sweep, so one of the two baselines is always
      // mis-labelled unless the operator picks. With NCCL_P2P_DISABLE unset,
      // NCCL may still ride NVLink in rows labelled `NCCL-IB`; with it set to
      // 1, `NvlSweep`'s NCCL baseline is forced onto the network. There is no
      // value that is right for both, so print what is in effect and let the
      // reader judge, and run the two sweeps in separate processes (separate
      // --gtest_filter invocations) when the comparison matters.
      const std::string p2pDisable = env_or("NCCL_P2P_DISABLE", "<unset>");
      COMMS_LOG_STREAM(INFO)
          << "\n=== NCCL baseline configuration (effective) ===\n"
          << "  NCCL_P2P_DISABLE           = " << p2pDisable << "\n"
          << "  NCCL_NCHANNELS_PER_NET_PEER= "
          << env_or("NCCL_NCHANNELS_PER_NET_PEER", "<unset>") << "\n"
          << "  NCCL_IB_QPS_PER_CONNECTION = "
          << env_or("NCCL_IB_QPS_PER_CONNECTION", "<unset>") << "\n"
          << "  NCCL_IB_SPLIT_DATA_ON_QPS  = "
          << env_or("NCCL_IB_SPLIT_DATA_ON_QPS", "<unset>") << "\n"
          << "  NCCL_BUFFSIZE              = "
          << env_or("NCCL_BUFFSIZE", "<unset>") << "\n"
          << "  NCCL_P2P_NET_CHUNKSIZE     = "
          << env_or("NCCL_P2P_NET_CHUNKSIZE", "<unset>") << "\n"
          << "\nThis NCCL baseline is DELIBERATELY CONSTRAINED and is not a\n"
          << "tuned NCCL number: one net channel per peer and two QPs, while\n"
          << "the Tile rows below provision far more IB groups. Read the\n"
          << "`vs NCCL` column as 'against NCCL held at this shape', not as\n"
          << "'against the best NCCL can do'.\n"
          << (p2pDisable == "1"
                  ? "NOTE: NCCL_P2P_DISABLE=1 -- NvlSweep's NCCL baseline is "
                    "forced onto the network too.\n"
                  : "NOTE: NCCL_P2P_DISABLE is not 1 -- rows labelled NCCL-IB "
                    "may still be using NVLink.\n");
    }

    // Value-initialized: only rank 0's ID is consumed below, but every rank
    // copies its own into the all-gather buffer, so leaving it indeterminate
    // feeds uninitialized bytes to the collective. Harmless in practice,
    // flagged by sanitizers, and free to avoid.
    ncclUniqueId id{};
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

    // Agree on the outcome across ALL ranks before acting on it.
    //
    // Acting on the local result alone diverges the ranks: the failing rank
    // skips every NCCL row while its peers still execute them, so the peers sit
    // in an `ncclAllReduce`/barrier that will never be matched and the run
    // hangs instead of reporting the setup failure. Whether the NCCL baseline
    // runs is therefore a property of the JOB, not of a rank.
    std::vector<int> nccl_ok(worldSize, 0);
    nccl_ok[globalRank] = (ncclRet == ncclSuccess) ? 1 : 0;
    bootstrap->allGather(nccl_ok.data(), sizeof(int), globalRank, worldSize)
        .get();
    const bool nccl_ok_everywhere = std::all_of(
        nccl_ok.begin(), nccl_ok.end(), [](int v) { return v != 0; });

    if (!nccl_ok_everywhere) {
      if (ncclRet == ncclSuccess && nccl_comm_ != nullptr) {
        // This rank initialized but a peer did not, so the communicator can
        // never complete a collective. Abort rather than destroy:
        // `ncclCommDestroy` is itself collective and would block on the ranks
        // that never got a communicator.
        ncclCommAbort(nccl_comm_);
      }
      nccl_comm_ = nullptr;
      COMMS_LOG_STREAM(WARNING)
          << "ncclCommInitRank failed on at least one rank (this rank rc="
          << ncclRet << "); every rank is skipping the NCCL baseline so the "
          << "sweep stays in lockstep";
    }

    MultiPeerTransportConfig transport_config{
        .nvlConfig =
            {
                .pipelineDepth = 2,
                .maxNumChannels = 32,
            },
    };
    transport_ = std::make_unique<MultiPeerTransport>(
        globalRank, worldSize, localRank, bootstrap, transport_config);
    transport_->exchange();
  }

  MultiPeerDeviceHandle make_handle() {
    return transport_->get_device_handle({partner()});
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

  // ---------------------------------------------------------------------
  // Untimed correctness pass
  //
  // The timed loop synchronizes and records a duration but never looks at the
  // bytes, so a wrong peer, a wrong offset, a truncated transfer or two
  // transfers overlapping all show up as a SPEEDUP rather than a failure. Each
  // sweep therefore runs one extra, untimed iteration through the exact same
  // kernel / stream / args it is about to time, and verifies it.
  //
  // Two details that make the check real rather than nominal:
  //
  //  - The payload is RANK-DISTINCT. It used to be one global hash of the byte
  //    offset, identical on every rank, so a transfer that delivered the
  //    receiver's own buffer, or a peer's instead of the intended one, was
  //    indistinguishable from a correct one.
  //  - The destination is POISONED with 0xA5, not zeroed. These sweeps zero the
  //    leading `sparsityPct%` of the payload on purpose, so a zeroed
  //    destination cannot distinguish "correctly received sparse data" from
  //    "never written at all".
  // ---------------------------------------------------------------------

  // The exact bytes rank `r` puts on the wire for this (size, sparsity).
  std::vector<uint8_t>
  expected_payload(int r, std::size_t bytes, int sparsityPct) const {
    std::vector<uint8_t> v(bytes);
    for (std::size_t i = 0; i < bytes; ++i) {
      uint64_t x = static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL +
          static_cast<uint64_t>(r) * 0x632BE59BD9B4E019ULL +
          0xDEADBEEFCAFEBABEULL;
      x = (x ^ (x >> 33)) * 0xff51afd7ed558ccdULL;
      x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53ULL;
      x = x ^ (x >> 33);
      v[i] = static_cast<uint8_t>(x & 0xff);
    }
    const std::size_t zero_bytes =
        (bytes * static_cast<std::size_t>(sparsityPct)) / 100;
    std::fill(
        v.begin(), v.begin() + static_cast<std::ptrdiff_t>(zero_bytes), 0);
    return v;
  }

  void fill_send_buffer(void* dst, std::size_t bytes, int sparsityPct) const {
    const std::vector<uint8_t> p =
        expected_payload(globalRank, bytes, sparsityPct);
    PIPES_CUDA_CHECK(cudaMemcpy(dst, p.data(), bytes, cudaMemcpyHostToDevice));
  }

  static void poison_buffer(void* dst, std::size_t bytes) {
    PIPES_CUDA_CHECK(cudaMemset(dst, 0xA5, bytes));
  }

  // Compare a received buffer against what `peer` should have sent. Reports at
  // most one failure, naming the first differing offset.
  void verify_received(
      const void* dst,
      int peer,
      std::size_t bytes,
      int sparsityPct,
      const char* what) const {
    std::vector<uint8_t> got(bytes);
    PIPES_CUDA_CHECK(
        cudaMemcpy(got.data(), dst, bytes, cudaMemcpyDeviceToHost));
    const std::vector<uint8_t> want =
        expected_payload(peer, bytes, sparsityPct);
    if (got == want) {
      return;
    }
    std::size_t at = 0;
    while (at < bytes && got[at] == want[at]) {
      ++at;
    }
    ADD_FAILURE() << "[" << what << "] rank " << globalRank
                  << ": payload from peer " << peer << " (" << bytes
                  << " bytes, sparsity " << sparsityPct << "%) differs at byte "
                  << at << " -- expected " << static_cast<int>(want[at])
                  << ", got " << static_cast<int>(got[at])
                  << (got[at] == 0xA5 ? " (still poison: nothing was written)"
                                      : "");
  }

  // Collectively decide whether the IB sweeps can run -- BEFORE any rank
  // constructs an IB transport.
  //
  // `MultiPeerTransport`'s constructor and `exchange()` are collective, so
  // catching a throw locally and returning is not recovery: the rank that threw
  // leaves the collective while its peers are still blocked inside it, and the
  // setup failure becomes a distributed hang instead of a message. Voting AFTER
  // the throw does not help for the same reason -- the peers never reach the
  // vote.
  //
  // Hence a preflight that touches nothing collective: every rank reports
  // whether it can see an IB device, the answers are all-gathered, and the
  // sweeps run only if EVERY rank said yes. A failure that still occurs after a
  // unanimous yes is genuinely exceptional and is deliberately left to
  // propagate -- a dead process is diagnosable, a hung job is not.
  bool ibUsableOnAllRanks() {
    int local = 0;
    std::error_code ec;
    std::filesystem::directory_iterator it("/sys/class/infiniband", ec);
    if (!ec && it != std::filesystem::directory_iterator{}) {
      local = 1;
    }
    std::vector<int> votes(worldSize, 0);
    votes[globalRank] = local;
    bootstrap->allGather(votes.data(), sizeof(int), globalRank, worldSize)
        .get();
    const bool all_yes =
        std::all_of(votes.begin(), votes.end(), [](int v) { return v != 0; });
    if (!all_yes && globalRank == 0) {
      COMMS_LOG_STREAM(WARN)
          << "Skipping IB sweep: no InfiniBand device visible on at least one "
             "rank (requires multi-node or IB loopback support). Every rank "
             "skips together so the sweep stays in lockstep.";
    }
    return all_yes;
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
    // Rank-distinct payload, poisoned destination: a constant fill (the old
    // `memset(send ? 1 : 0)`) cannot tell a correct transfer from one that
    // delivered the wrong peer's bytes or the wrong offset.
    if (send) {
      fill_send_buffer(buf.get(), bytes, /*sparsityPct=*/0);
    } else {
      poison_buffer(buf.get(), bytes);
    }

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

    auto launch_one_plain = [&]() {
      void* ka[] = {&args, &abortDevice_};
      return comms::common::launchKernel(
          (void*)sendrecv_tile_kernel,
          dim3(num_blocks),
          dim3(512),
          ka,
          nullptr,
          cluster_dim);
    };

    bootstrap->barrierAll();
    for (int i = 0; i < kNWarmup; i++) {
      (void)launch_one_plain();
      PIPES_CUDA_CHECK(cudaDeviceSynchronize());
    }
    bootstrap->barrierAll();

    // Untimed correctness iteration through the same kernel/grid/cluster and
    // the same max_signal_bytes the timed loop uses.
    {
      if (!send) {
        poison_buffer(buf.get(), bytes);
      }
      PIPES_CUDA_CHECK(cudaDeviceSynchronize());
      bootstrap->barrierAll();
      (void)launch_one_plain();
      PIPES_CUDA_CHECK(cudaDeviceSynchronize());
      bootstrap->barrierAll();
      if (!send) {
        verify_received(buf.get(), peer, bytes, /*sparsityPct=*/0, "run_tile");
      }
      bootstrap->barrierAll();
    }

    CudaEvent start, stop;
    PIPES_CUDA_CHECK(cudaEventRecord(start.get()));
    for (int i = 0; i < kNIter; i++) {
      (void)launch_one_plain();
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
    // Rank-distinct payload + poisoned destination; see run_tile.
    fill_send_buffer(send_buf.get(), bytes, /*sparsityPct=*/0);
    poison_buffer(recv_buf.get(), bytes);

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

    auto launch_one_plain = [&]() {
      void* ka[] = {&args, &abortDevice_};
      return comms::common::launchKernel(
          (void*)sendrecv_tile_kernel,
          dim3(num_blocks),
          dim3(512),
          ka,
          nullptr,
          cluster_dim);
    };

    bootstrap->barrierAll();
    for (int i = 0; i < kNWarmup; i++) {
      (void)launch_one_plain();
      PIPES_CUDA_CHECK(cudaDeviceSynchronize());
    }
    bootstrap->barrierAll();

    // Untimed correctness iteration. Both directions run here, so this also
    // covers the in-kernel half-send/half-recv grid split.
    {
      poison_buffer(recv_buf.get(), bytes);
      PIPES_CUDA_CHECK(cudaDeviceSynchronize());
      bootstrap->barrierAll();
      (void)launch_one_plain();
      PIPES_CUDA_CHECK(cudaDeviceSynchronize());
      bootstrap->barrierAll();
      verify_received(
          recv_buf.get(), peer, bytes, /*sparsityPct=*/0, "run_tile_twoway");
      bootstrap->barrierAll();
    }

    CudaEvent start, stop;
    PIPES_CUDA_CHECK(cudaEventRecord(start.get()));
    for (int i = 0; i < kNIter; i++) {
      (void)launch_one_plain();
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
  AbortDevice abortDevice_;
};

TEST_F(SendRecvTileVariantsBenchmarkFixture, NvlSweep) {
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
    COMMS_LOG_STREAM(INFO) << "\n=== SendRecvTile NVL Sweep ===\n"
                           << worldSize << " GPUs (paired), " << kNIter
                           << " iterations\n"
                           << "Block counts: " << blocks_a << " vs " << blocks_b
                           << "\n";
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

TEST_F(SendRecvTileVariantsBenchmarkFixture, IbSweep) {
  // Force IB by disabling P2P NVLink in topology discovery.
  MultiPeerTransportConfig ib_config{
      .nvlConfig =
          {
              .pipelineDepth = 2,
              .maxNumChannels = 32,
          },
      .ibConfig =
          {
              .cudaDevice = localRank,
              .perChannelSize = 8 * 1024 * 1024 / 128,
              .max_num_channels = 128,
              .pipelineDepth = 2,
          },
      .topoConfig =
          {
              .p2pDisable = true,
          },
  };

  // Preflight FIRST: every rank agrees on whether to run before anyone
  // enters a collective. See `ibUsableOnAllRanks()` for why a try/catch
  // around the construction cannot work here.
  if (!ibUsableOnAllRanks()) {
    return;
  }
  auto ib_transport = std::make_unique<MultiPeerTransport>(
      globalRank, worldSize, localRank, bootstrap, ib_config);
  ib_transport->exchange();
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

  const int blocks_a = 64;
  const int blocks_b = 96;

  char hdr_a[16], hdr_b[16];
  snprintf(hdr_a, sizeof(hdr_a), "%dblk", blocks_a);
  snprintf(hdr_b, sizeof(hdr_b), "%dblk", blocks_b);

  if (globalRank == 0) {
    COMMS_LOG_STREAM(INFO) << "\n=== SendRecvTile IB Sweep ===\n"
                           << worldSize << " GPUs (paired), " << kNIter
                           << " iterations\n"
                           << "Block counts: " << blocks_a << " vs " << blocks_b
                           << "\n"
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

TEST_F(SendRecvTileVariantsBenchmarkFixture, IbSweepTwoWay) {
  // Bidirectional IB sweep: every paired rank sends to AND receives from
  // its partner at the same time (the kernel splits the grid half-send /
  // half-recv). Compares against NCCL grouped send+recv. BW is reported
  // per-direction (bytes / time).
  MultiPeerTransportConfig ib_config{
      .nvlConfig =
          {
              .pipelineDepth = 2,
              .maxNumChannels = 32,
          },
      .ibConfig =
          {
              .cudaDevice = localRank,
              .perChannelSize = 8 * 1024 * 1024 / 128,
              .max_num_channels = 128,
              .pipelineDepth = 2,
          },
      .topoConfig =
          {
              .p2pDisable = true,
          },
  };

  // Preflight FIRST: every rank agrees on whether to run before anyone
  // enters a collective. See `ibUsableOnAllRanks()` for why a try/catch
  // around the construction cannot work here.
  if (!ibUsableOnAllRanks()) {
    return;
  }
  auto ib_transport = std::make_unique<MultiPeerTransport>(
      globalRank, worldSize, localRank, bootstrap, ib_config);
  ib_transport->exchange();
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

  // Total grid blocks; split in half inside the kernel (so each direction
  // gets blocks_*/2). Both even.
  const int blocks_a = 64;
  const int blocks_b = 128;

  char hdr_a[16], hdr_b[16];
  snprintf(hdr_a, sizeof(hdr_a), "%dblk", blocks_a);
  snprintf(hdr_b, sizeof(hdr_b), "%dblk", blocks_b);

  if (globalRank == 0) {
    COMMS_LOG_STREAM(INFO)
        << "\n=== SendRecvTile IB Two-Way Sweep ===\n"
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

TEST_F(SendRecvTileVariantsBenchmarkFixture, IbSweepCompressed) {
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
  //                                          idle when wireRatio >> bus gain.
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
  // Per-group slot = perChannelSize / pipelineDepth (see pipeline_chunk()); on
  // current stable that /pipelineDepth halves the effective slot vs the old
  // transport, so the historical 128 MiB floor left perBlockSlot (256 KiB) too
  // small to hold one worst-case-compressed sub-chunk (~1.3x MaxUncompBytes),
  // tripping the transport's `chunkStride > perBlockSlot` trap once compression
  // activates. Use a 512 MiB floor so perBlockSlot (>= 1 MiB here) fits it.
  std::size_t ibDataBufferSize = std::max<std::size_t>(
      512ULL * 1024 * 1024,
      static_cast<std::size_t>(kBenchBlocks) * 512ULL * 1024);
  ibDataBufferSize = (ibDataBufferSize / kSlotQuantum) * kSlotQuantum;
  ASSERT_GT(ibDataBufferSize, 0u);
  MultiPeerTransportConfig ib_config{
      .nvlConfig =
          {
              .pipelineDepth = 2,
              .maxNumChannels = 32,
          },
      .ibConfig =
          {
              .cudaDevice = localRank,
              // Cover the configured grid block count on the one-way path
              // (active_blocks == grid_blocks there). perChannelSize is the
              // per-group slot; the ctor derives dataBufferSize =
              // perChannelSize * max_num_channels and sets IbSendRecvState
              // .maxGroups from max_num_channels.
              .perChannelSize = ibDataBufferSize / std::max(kBenchBlocks, 256),
              .max_num_channels = std::max(kBenchBlocks, 256),
              .pipelineDepth = 2,
              // >= 256 groups => >= 256 QPs/peer/NIC, which exceeds the 128-QP
              // eager exchange wire format; materialize the peer's QPs lazily
              // instead. SendRecv uses getP2pTransportDevice() per peer, so it
              // is unaffected by lazy mode (unlike DeviceWindow/AllToAllv).
              .ibLazyConnect = true,
          },
      .topoConfig =
          {
              .p2pDisable = true,
          },
  };

  // Preflight FIRST: every rank agrees on whether to run before anyone
  // enters a collective. See `ibUsableOnAllRanks()` for why a try/catch
  // around the construction cannot work here.
  if (!ibUsableOnAllRanks()) {
    return;
  }
  auto ib_transport = std::make_unique<MultiPeerTransport>(
      globalRank, worldSize, localRank, bootstrap, ib_config);
  ib_transport->exchange();
  // Lazy mode: materialize this peer's QPs and get a handle scoped to it.
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
    COMMS_LOG_STREAM(INFO)
        << "\n=== SendRecvTile IB Compressed Sweep ===\n"
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
      COMMS_LOG_STREAM(INFO) << "\n--- Sparsity " << sparsityPct
                             << "% (zero-byte fraction of input buffer) ---";
      printf(
          "%-10s  %10s  %14s  %-10s %8s  %12s\n",
          "Size",
          "NCCL-IB",
          "Tile",
          "Best",
          "vs NCCL",
          "wireRatio");
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
      // One definition of the payload, shared with the verification pass:
      // rank-distinct hash, leading sparsityPct% zeroed. The receiver is
      // poisoned rather than zeroed so "never written" stays distinguishable
      // from "correctly received sparse data".
      if (send) {
        fill_send_buffer(buf.get(), bytes, sparsityPct);
      } else {
        poison_buffer(buf.get(), bytes);
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
      // Resolved once, through the public picker rather than by naming the
      // template: the kernel symbol is private now, and the picker is what
      // states which (threads_per_block, min_blocks_per_sm) pairs actually
      // exist. An unsupported pair returns nullptr here instead of failing at
      // device link.
      void* const kfn = pick_sendrecv_tile_compressed_kernel(
          kNumThreadsPerBlock, kBenchMinBlocksPerSM);
      ASSERT_NE(kfn, nullptr)
          << "no compressed kernel for threads_per_block="
          << kNumThreadsPerBlock
          << " min_blocks_per_sm=" << kBenchMinBlocksPerSM
          << "; set PIPES_SENDRECV_BENCH_MIN_BLOCKS_PER_SM to an instantiated "
             "value";

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
        void* ka[] = {&args, &abortDevice_};
        const cudaError_t lerr = launch_one(ka);
        if (lerr != cudaSuccess) {
          COMMS_LOG_STREAM(FATAL) << "[PIPES] FATAL: warmup launch err=" << lerr
                                  << " (" << cudaGetErrorString(lerr)
                                  << ") bytes=" << bytes << " iter=" << i;
        }
        PIPES_CUDA_CHECK(cudaDeviceSynchronize());
      }
      bootstrap->barrierAll();

      // Untimed correctness iteration, through the SAME kernel / stream / args
      // the timed loop is about to use -- including the green-context stream
      // and whatever max_signal_bytes is configured. See the helpers on the
      // fixture for why the payload is rank-distinct and the destination is
      // poisoned rather than zeroed.
      {
        if (send) {
          fill_send_buffer(buf.get(), bytes, sparsityPct);
        } else {
          poison_buffer(buf.get(), bytes);
        }
        PIPES_CUDA_CHECK(cudaDeviceSynchronize());
        bootstrap->barrierAll();
        void* ka[] = {&args, &abortDevice_};
        const cudaError_t lerr = launch_one(ka);
        ASSERT_EQ(lerr, cudaSuccess)
            << "verification launch failed: " << cudaGetErrorString(lerr);
        PIPES_CUDA_CHECK(cudaDeviceSynchronize());
        bootstrap->barrierAll();
        if (!send) {
          verify_received(
              buf.get(), peer, bytes, sparsityPct, "IbSweepCompressed");
        }
        // Restore the timed-loop contents: the sender's buffer doubles as the
        // compressibility input, so it has to hold the sweep's payload again.
        if (send) {
          fill_send_buffer(buf.get(), bytes, sparsityPct);
        }
        PIPES_CUDA_CHECK(cudaDeviceSynchronize());
        bootstrap->barrierAll();
      }

      // Reset compression counters AFTER warmup and after the verification
      // iteration, so the ratio reflects only timed iterations.
      (void)fetch_and_reset_sendrecv_ans_compress_stats();

      CudaEvent start, stop;
      PIPES_CUDA_CHECK(cudaEventRecord(start.get(), benchStream));
      for (int i = 0; i < kNIter; i++) {
        void* ka[] = {&args, &abortDevice_};
        const cudaError_t lerr = launch_one(ka);
        if (lerr != cudaSuccess) {
          COMMS_LOG_STREAM(FATAL) << "[PIPES] FATAL: timed launch err=" << lerr
                                  << " (" << cudaGetErrorString(lerr)
                                  << ") bytes=" << bytes << " iter=" << i;
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
        // WHOLE-TRANSFER ratio, not the ANS-payload-only one.
        //
        // The counters are fed only by blocks the compressor actually ran on,
        // so `uncomp / comp` describes the ANS payload in isolation and
        // overstates what the wire saw: a transfer that is 30% plain and
        // compresses the rest 4x reported "4.00x", while the wire carried
        // 30 + 70/4 = 47.5% of the input, i.e. about 2.1x. Plain bytes are
        // whatever the timed loop pushed that the compressor never saw.
        //
        // Still slightly optimistic: `compressed_bytes` is the ANS payload and
        // excludes the per-chunk size-header table and 16-byte alignment
        // padding, so the true on-wire figure is a little larger than this.
        const double totalLogical =
            static_cast<double>(bytes) * static_cast<double>(kNIter);
        const double ansUncomp = static_cast<double>(tile_r.comp_uncomp_bytes);
        const double ansComp = static_cast<double>(tile_r.comp_comp_bytes);
        const double plainLogical = std::max(0.0, totalLogical - ansUncomp);
        const double wireBytes = plainLogical + ansComp;
        if (ansUncomp <= 0.0) {
          // Nothing reached the compressor: every block took the plain Memcpy
          // path, either because plain_block_pct=100 or because the per-peer
          // byte count was under CopyOp::kActivationThreshold. Say so rather
          // than printing a number that looks measured.
          snprintf(comp_ratio_buf, sizeof(comp_ratio_buf), "n/a (all plain)");
        } else if (wireBytes > 0.0) {
          snprintf(
              comp_ratio_buf,
              sizeof(comp_ratio_buf),
              "%.2fx",
              totalLogical / wireBytes);
        } else {
          snprintf(comp_ratio_buf, sizeof(comp_ratio_buf), "n/a");
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

TEST_F(SendRecvTileVariantsBenchmarkFixture, IbSweepCompressedTwoWay) {
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
  //     idle when wireRatio >> realized bus gain.
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

  // Declared here rather than just before the launch: the slot-sizing block
  // below needs the warp count to ask the compressor for its worst-case
  // chunk stride.
  constexpr int kNumWarpsPerBlock = 8;
  constexpr int kNumThreadsPerBlock = kNumWarpsPerBlock * 32;

  // The channel count is CLAMPED UP to 256, so it does not shrink with
  // kBenchBlocks -- which is why lowering the block count does not enlarge the
  // slot.
  const int kTwoWayChannels = std::max(kBenchBlocks / 2, 256);

  // The per-block slot must hold one WORST-CASE encoded ANS chunk, not one
  // logical chunk: `perBlockSlot = perChannelSize / pipelineDepth`, and the
  // encoded form of a `kMaxUncompBytes` input is ~1.3x plus a size-header table
  // plus 16-byte alignment. When it does not fit,
  // `max_safe_chunk_size_for_slot()` finds room for ZERO chunks, falls back to
  // returning kMaxUncompBytes, and the transport traps -- surfacing only as a
  // bare "unspecified launch failure", with the diagnostic printf swallowed by
  // __trap().
  //
  // This sweep used to hit exactly that: 128 MB / 256 channels = 512 KiB per
  // channel, / 2 = a 256 KiB slot against a ~341 KiB worst case, so every size
  // at or above CopyOp::kActivationThreshold (4 MiB) aborted while everything
  // below it passed, because below the threshold nothing is compressed at all.
  // Raise the floor so the slot fits, and assert rather than trap if a
  // caller-supplied PIPES_SENDRECV_BENCH_DATA_BUF_BYTES is too small.
  // Per-chunk slot requirement, asked of the COMPRESSOR rather than guessed.
  //
  // nvCOMPDx reports the exact figure via `max_comp_chunk_size()`, which
  // `AnsCompress::worst_case_chunk_stride()` wraps with the size-header table
  // and the 512-byte NIC-burst alignment. Two earlier attempts here used closed
  // forms instead and both were wrong: the documented ~`1.3 * MaxUncomp + 1576`
  // (335 KiB) and `AnsCopyOpBench`'s `1.4 * MaxUncomp + 4096` (363 KiB) each
  // produced a slot that still trapped on this path.
  //
  // The helper cannot simply be called from here. It is `__host__ __device__`,
  // but the compressor descriptor embeds `SM<kAnsArch>`, and `kAnsArch` is
  // resolved from `__CUDA_ARCH__` -- undefined on the host pass, where it falls
  // back to 800. A host-side call would answer for sm_80 rather than for the
  // architecture the kernel runs on. So
  // `sendrecv_ans_worst_case_chunk_stride()` queries the device: one
  // single-thread kernel evaluating the helper under the real `__CUDA_ARCH__`.
  const std::size_t kWorstCaseChunk = sendrecv_ans_worst_case_chunk_stride(
      kNumWarpsPerBlock, AnsCompressor::kMaxUncompBytes);
  ASSERT_GT(kWorstCaseChunk, 0u)
      << "no compressed kernel family for num_warps=" << kNumWarpsPerBlock;
  // The floor has TWO requirements, and the second one is the subtle one.
  //
  // 1. The slot must hold one worst-case chunk (above).
  // 2. The slot must be 512-BYTE ALIGNED. `pipeline_chunk()` is masked with
  //    `& ~511` by the transport, so a slot whose unmasked value is not a
  //    multiple of 512 is silently reduced, and the reduced value is what the
  //    chunk-stride check actually sees.
  //
  // Requirement 2 alone caused a trap that looked like requirement 1: a run
  // with a 371456-byte chunk (not 512-aligned, masked down to 371200) trapped,
  // while both a LARGER 401408 and a SMALLER 342528 -- each exactly
  // 512-aligned -- passed. Sizing purely by "is the slot big enough" gets this
  // wrong non-monotonically, which is a miserable thing to debug.
  constexpr std::size_t kSlotAlign = 512ULL;
  const std::size_t kAlignedChunk =
      ((kWorstCaseChunk + kSlotAlign - 1ULL) / kSlotAlign) * kSlotAlign;
  const std::size_t kMinDataBuf =
      kAlignedChunk * 2 * static_cast<std::size_t>(kTwoWayChannels);
  if (kBenchDataBufBytes == 0 && ibDataBufferSize < kMinDataBuf) {
    ibDataBufferSize =
        ((kMinDataBuf + kSlotQuantum - 1) / kSlotQuantum) * kSlotQuantum;
  }
  {
    const std::size_t slot =
        (ibDataBufferSize / static_cast<std::size_t>(kTwoWayChannels)) / 2;
    if (globalRank == 0) {
      COMMS_LOG_STREAM(INFO)
          << "ANS worst-case chunk stride (device-queried, num_warps="
          << kNumWarpsPerBlock << "): " << kWorstCaseChunk << " bytes ("
          << format_bytes(kWorstCaseChunk) << "); per-block slot " << slot
          << " (" << format_bytes(slot) << ")";
    }
    EXPECT_EQ(slot % kSlotAlign, 0u)
        << "per-block slot " << slot << " is not 512-byte aligned; the "
        << "transport masks it with & ~511, so the effective slot is smaller "
        << "than this and the chunk-stride check can fail non-monotonically";
    ASSERT_GE(slot, kWorstCaseChunk)
        << "per-block slot " << slot << " cannot hold one worst-case encoded "
        << "ANS chunk bound (" << kWorstCaseChunk
        << "); the transport would trap "
        << "with a bare 'unspecified launch failure'. Raise "
        << "PIPES_SENDRECV_BENCH_DATA_BUF_BYTES to at least " << kMinDataBuf;
  }
  MultiPeerTransportConfig ib_config{
      .nvlConfig =
          {
              .pipelineDepth = 2,
              .maxNumChannels = 32,
          },
      .ibConfig =
          {
              .cudaDevice = localRank,
              // Two-way splits the grid: active blocks per direction ==
              // grid_blocks/2. perChannelSize is the per-group slot; the ctor
              // derives dataBufferSize = perChannelSize * max_num_channels and
              // sets IbSendRecvState.maxGroups from max_num_channels.
              .perChannelSize = ibDataBufferSize / kTwoWayChannels,
              .max_num_channels = kTwoWayChannels,
              .pipelineDepth = 2,
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

  // Preflight FIRST: every rank agrees on whether to run before anyone
  // enters a collective. See `ibUsableOnAllRanks()` for why a try/catch
  // around the construction cannot work here.
  if (!ibUsableOnAllRanks()) {
    return;
  }
  auto ib_transport = std::make_unique<MultiPeerTransport>(
      globalRank, worldSize, localRank, bootstrap, ib_config);
  ib_transport->exchange();
  // Lazy mode: materialize this peer's QPs and get a handle scoped to it.
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
    COMMS_LOG_STREAM(INFO)
        << "\n=== SendRecvTile IB Compressed Two-Way Sweep ===\n"
        << worldSize << " GPUs (paired, bidirectional), " << kNIter
        << " iterations\n"
        << grid_blocks << " total blocks (split half send/half recv), "
        << kNumThreadsPerBlock
        << " threads, min_blocks_per_sm=" << kBenchMinBlocksPerSM
        << ", num_sms=" << kBenchNumSms
        << (kSplit ? " (green-ctx SPLIT send|recv)"
                   : (kBenchNumSms > 0 ? " (green-ctx fused)" : " (full GPU)"))
        << "\n"
        << "plain_block_pct=" << kBenchPlainPct
        << "% (Memcpy blocks), ibDataBufferSize="
        << format_bytes(ibDataBufferSize) << "\n"
        << describe_max_signal(
               kBenchMaxSignal,
               ibDataBufferSize / kTwoWayChannels,
               /*pipelineDepth=*/2)
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
      COMMS_LOG_STREAM(INFO) << "\n--- Sparsity " << sparsityPct
                             << "% (zero-byte fraction of input buffer) ---";
      printf(
          "%-10s  %10s  %14s  %-10s %8s  %12s\n",
          "Size",
          "NCCL-IB",
          "Tile",
          "Best",
          "vs NCCL",
          "wireRatio");
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

      // Every rank both sends (rank-distinct compressible pattern, sparsified)
      // and receives. Same payload definition the verification pass uses.
      DeviceBuffer send_buf(bytes);
      DeviceBuffer recv_buf(bytes);
      fill_send_buffer(send_buf.get(), bytes, sparsityPct);
      poison_buffer(recv_buf.get(), bytes);

      // Kernel for the configured launch_bounds m, via the public picker; see
      // the one-way sweep for why this is not the raw template.
      void* const kfn = pick_sendrecv_tile_compressed_kernel(
          kNumThreadsPerBlock, kBenchMinBlocksPerSM);
      ASSERT_NE(kfn, nullptr) << "no compressed kernel for threads_per_block="
                              << kNumThreadsPerBlock
                              << " min_blocks_per_sm=" << kBenchMinBlocksPerSM;

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
          void* ka[] = {&sArgs, &abortDevice_};
          return comms::common::launchKernel(
              kfn,
              dim3(dirBlocks),
              dim3(kNumThreadsPerBlock),
              ka,
              /*stream=*/sendStream,
              clusterDimForGrid(kBenchClusterDim, dirBlocks));
        };
        auto launch_recv = [&]() {
          void* ka[] = {&rArgs, &abortDevice_};
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

        // Untimed correctness iteration over the SPLIT green-context streams,
        // which is the variant most likely to get the tiling wrong: send and
        // recv run concurrently on two SM partitions against one staging area.
        {
          fill_send_buffer(send_buf.get(), bytes, sparsityPct);
          poison_buffer(recv_buf.get(), bytes);
          PIPES_CUDA_CHECK(cudaDeviceSynchronize());
          bootstrap->barrierAll();
          PIPES_CUDA_CHECK(launch_send());
          PIPES_CUDA_CHECK(launch_recv());
          PIPES_CUDA_CHECK(cudaDeviceSynchronize());
          bootstrap->barrierAll();
          verify_received(
              recv_buf.get(),
              partner(),
              bytes,
              sparsityPct,
              "IbSweepCompressedTwoWay/split");
          fill_send_buffer(send_buf.get(), bytes, sparsityPct);
          PIPES_CUDA_CHECK(cudaDeviceSynchronize());
          bootstrap->barrierAll();
        }

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
          void* ka[] = {&args, &abortDevice_};
          const cudaError_t lerr = launch_one(ka);
          if (lerr != cudaSuccess) {
            COMMS_LOG_STREAM(FATAL)
                << "[PIPES] FATAL: warmup launch err=" << lerr << " ("
                << cudaGetErrorString(lerr) << ") bytes=" << bytes
                << " iter=" << i;
          }
          PIPES_CUDA_CHECK(cudaDeviceSynchronize());
        }
        bootstrap->barrierAll();

        // Untimed correctness iteration over the FUSED path (one stream, grid
        // split half send / half recv inside the kernel).
        {
          fill_send_buffer(send_buf.get(), bytes, sparsityPct);
          poison_buffer(recv_buf.get(), bytes);
          PIPES_CUDA_CHECK(cudaDeviceSynchronize());
          bootstrap->barrierAll();
          void* ka[] = {&args, &abortDevice_};
          const cudaError_t lerr = launch_one(ka);
          ASSERT_EQ(lerr, cudaSuccess)
              << "verification launch failed: " << cudaGetErrorString(lerr);
          PIPES_CUDA_CHECK(cudaDeviceSynchronize());
          bootstrap->barrierAll();
          verify_received(
              recv_buf.get(),
              partner(),
              bytes,
              sparsityPct,
              "IbSweepCompressedTwoWay/fused");
          fill_send_buffer(send_buf.get(), bytes, sparsityPct);
          PIPES_CUDA_CHECK(cudaDeviceSynchronize());
          bootstrap->barrierAll();
        }

        (void)fetch_and_reset_sendrecv_ans_compress_stats();

        CudaEvent start, stop;
        PIPES_CUDA_CHECK(cudaEventRecord(start.get(), benchStream));
        for (int i = 0; i < kNIter; i++) {
          void* ka[] = {&args, &abortDevice_};
          const cudaError_t lerr = launch_one(ka);
          if (lerr != cudaSuccess) {
            COMMS_LOG_STREAM(FATAL)
                << "[PIPES] FATAL: timed launch err=" << lerr << " ("
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
        // WHOLE-TRANSFER ratio, not the ANS-payload-only one.
        //
        // The counters are fed only by blocks the compressor actually ran on,
        // so `uncomp / comp` describes the ANS payload in isolation and
        // overstates what the wire saw: a transfer that is 30% plain and
        // compresses the rest 4x reported "4.00x", while the wire carried
        // 30 + 70/4 = 47.5% of the input, i.e. about 2.1x. Plain bytes are
        // whatever the timed loop pushed that the compressor never saw.
        //
        // Still slightly optimistic: `compressed_bytes` is the ANS payload and
        // excludes the per-chunk size-header table and 16-byte alignment
        // padding, so the true on-wire figure is a little larger than this.
        const double totalLogical =
            static_cast<double>(bytes) * static_cast<double>(kNIter);
        const double ansUncomp = static_cast<double>(tile_r.comp_uncomp_bytes);
        const double ansComp = static_cast<double>(tile_r.comp_comp_bytes);
        const double plainLogical = std::max(0.0, totalLogical - ansUncomp);
        const double wireBytes = plainLogical + ansComp;
        if (ansUncomp <= 0.0) {
          // Nothing reached the compressor: every block took the plain Memcpy
          // path, either because plain_block_pct=100 or because the per-peer
          // byte count was under CopyOp::kActivationThreshold. Say so rather
          // than printing a number that looks measured.
          snprintf(comp_ratio_buf, sizeof(comp_ratio_buf), "n/a (all plain)");
        } else if (wireBytes > 0.0) {
          snprintf(
              comp_ratio_buf,
              sizeof(comp_ratio_buf),
              "%.2fx",
              totalLogical / wireBytes);
        } else {
          snprintf(comp_ratio_buf, sizeof(comp_ratio_buf), "n/a");
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
