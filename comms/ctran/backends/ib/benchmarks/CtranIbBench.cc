// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <unistd.h>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <folly/init/Init.h>

#include "comms/ctran/backends/ib/BootstrapExternal.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/Exception.h"

using namespace ctran;

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

constexpr int kDummyRank = 0;

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------

// Define the config to pass to the benchmark
CtranIbConfig config_256k{
    .numQps = 16,
    .qpScalingTh = 262144,
    .qpMsgs = 128,
};

CtranIbConfig config_512k{
    .numQps = 16,
    .qpScalingTh = 524288,
    .qpMsgs = 128,
};

//------------------------------------------------------------------------------
// CtranIb Benchmarks
//------------------------------------------------------------------------------

struct BenchmarkContext {
  std::unique_ptr<CtranIb> senderIb;
  std::unique_ptr<CtranIb> receiverIb;
  void* sendBuffer{};
  void* recvBuffer{};
  void* senderRegHdl{};
  void* receiverRegHdl{};
  CtranIbRemoteAccessKey ibSendKey;
  CtranIbRemoteAccessKey ibReceiveKey;
  size_t bufferSize;

  // Constructor
  BenchmarkContext(
      std::unique_ptr<CtranIb> senderIb_,
      std::unique_ptr<CtranIb> receiverIb_,
      void* sendBuffer_,
      void* recvBuffer_,
      void* senderRegHdl_,
      void* receiverRegHdl_,
      CtranIbRemoteAccessKey ibSendKey_,
      CtranIbRemoteAccessKey ibReceiveKey_,
      size_t bufferSize_)
      : senderIb(std::move(senderIb_)),
        receiverIb(std::move(receiverIb_)),
        sendBuffer(sendBuffer_),
        recvBuffer(recvBuffer_),
        senderRegHdl(senderRegHdl_),
        receiverRegHdl(receiverRegHdl_),
        ibSendKey(ibSendKey_),
        ibReceiveKey(ibReceiveKey_),
        bufferSize(bufferSize_) {}

  // Disable copy and move constructors
  BenchmarkContext(const BenchmarkContext&) = delete;
  BenchmarkContext& operator=(const BenchmarkContext&) = delete;
  BenchmarkContext(BenchmarkContext&&) = delete;
  BenchmarkContext& operator=(BenchmarkContext&&) = delete;
};

static BenchmarkContext setupBenchmarkContext(size_t bufferSize) {
  const int cudaDev0 = 0;
  const int cudaDev1 = 1;

  // GB200 (aarch64) uses 64KB pages; dma-buf registration via
  // cuMemGetHandleForAddressRange requires a page-aligned length (sub-page
  // lengths fail with CUDA_ERROR_INVALID_VALUE). Register a page-aligned region
  // while still transferring `bufferSize` bytes.
  const size_t pageSize = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const size_t regLen = ((bufferSize + pageSize - 1) / pageSize) * pageSize;

  // Initialize senderIb and receiverIb
  auto senderIb = std::make_unique<CtranIb>(
      kDummyRank,
      cudaDev0,
      -1 /* commHash */,
      "RDMA-Transport",
      true /* enableLocalFlush */,
      CtranIb::BootstrapMode::kExternal);
  auto receiverIb = std::make_unique<CtranIb>(
      kDummyRank,
      cudaDev1,
      -1 /* commHash */,
      "RDMA-Transport",
      true /* enableLocalFlush */,
      CtranIb::BootstrapMode::kExternal);

  // Connect senderIb and receiverIb
  auto senderVcIdentifier =
      senderIb->externalBootstrap()->getLocalVcId(kDummyRank);
  auto receiverVcIdentifier =
      receiverIb->externalBootstrap()->getLocalVcId(kDummyRank);
  CHECK_EQ(
      senderIb->externalBootstrap()->connectVc(
          receiverVcIdentifier, kDummyRank),
      commSuccess);
  CHECK_EQ(
      receiverIb->externalBootstrap()->connectVc(
          senderVcIdentifier, kDummyRank),
      commSuccess);

  // Allocate RDMA-registerable device buffers with commCudaMalloc, the same
  // auto-selecting allocator CTRAN algos use: a GPUDirect-RDMA-capable CUDA VMM
  // allocation when cuMem is supported (matching production and ncclMemAlloc,
  // both POSIX|FABRIC on GB200), falling back to cudaMalloc otherwise. The
  // dma-buf fd export used by regMem (cuMemGetHandleForAddressRange) operates
  // on the VMM address range and is independent of the POSIX/FABRIC handle
  // type. Allocate/register the page-aligned regLen while still transferring
  // bufferSize bytes.
  CHECK_EQ(cudaSetDevice(cudaDev0), cudaSuccess);
  void* sendBuffer = nullptr;
  CHECK_EQ(
      ctran::utils::commCudaMalloc(
          reinterpret_cast<char**>(&sendBuffer),
          regLen,
          /*logMetaData=*/nullptr,
          "CtranIbBench"),
      commSuccess);
  void* senderRegHdl = nullptr;
  if (CtranIb::regMem(sendBuffer, regLen, cudaDev0, &senderRegHdl) !=
      commSuccess) {
    throw ctran::utils::Exception(
        "regMem failed for sendBuffer", commSystemError);
  }

  // Allocate memory on the receiver side (see sender note above).
  CHECK_EQ(cudaSetDevice(cudaDev1), cudaSuccess);
  void* recvBuffer = nullptr;
  CHECK_EQ(
      ctran::utils::commCudaMalloc(
          reinterpret_cast<char**>(&recvBuffer),
          regLen,
          /*logMetaData=*/nullptr,
          "CtranIbBench"),
      commSuccess);
  void* receiverRegHdl = nullptr;
  if (CtranIb::regMem(recvBuffer, regLen, cudaDev1, &receiverRegHdl) !=
      commSuccess) {
    throw ctran::utils::Exception(
        "regMem failed for recvBuffer", commSystemError);
  }

  // Check connection
  if (senderIb->getVc(kDummyRank) == nullptr) {
    throw ctran::utils::Exception("senderIb not connected", commInternalError);
  }
  if (receiverIb->getVc(kDummyRank) == nullptr) {
    throw ctran::utils::Exception(
        "receiverIb not connected", commInternalError);
  }

  auto ibSendKey = CtranIb::getRemoteAccessKey(senderRegHdl);
  auto ibReceiveKey = CtranIb::getRemoteAccessKey(receiverRegHdl);

  return BenchmarkContext(
      std::move(senderIb),
      std::move(receiverIb),
      sendBuffer,
      recvBuffer,
      senderRegHdl,
      receiverRegHdl,
      ibSendKey,
      ibReceiveKey,
      bufferSize);
}

static void cleanupBenchmarkContext(BenchmarkContext& ctx) {
  if (CtranIb::deregMem(ctx.senderRegHdl) != commSuccess) {
    XLOGF(ERR, "deregMem failed for senderRegHdl");
  }

  if (CtranIb::deregMem(ctx.receiverRegHdl) != commSuccess) {
    XLOGF(ERR, "deregMem failed for receiverRegHdl");
  }

  // Free each buffer on the device it was allocated on (CUDA VMM unmap is
  // context-sensitive): sendBuffer on cudaDev0, recvBuffer on cudaDev1.
  CHECK_EQ(cudaSetDevice(0), cudaSuccess);
  CHECK_EQ(ctran::utils::commCudaFree(ctx.sendBuffer), commSuccess);
  CHECK_EQ(cudaSetDevice(1), cudaSuccess);
  CHECK_EQ(ctran::utils::commCudaFree(ctx.recvBuffer), commSuccess);
}

static void
benchmarkIput(benchmark::State& state, CtranIbConfig config, bool withNotify) {
  const size_t bufferSize = state.range(0);
  auto ctx = setupBenchmarkContext(bufferSize);

  // Benchmark the iput operation
  for (auto _ : state) {
    CtranIbRequest ibReq;
    if (ctx.senderIb->iput(
            ctx.sendBuffer, /* sbuf */
            ctx.recvBuffer, /* dbuf */
            ctx.bufferSize, /* len */
            kDummyRank, /* peerRank */
            ctx.senderRegHdl, /* ibRegElem */
            ctx.ibReceiveKey, /* remoteAccessKey */
            withNotify, /* notify */
            &config, /* config */
            &ibReq, /* req */
            false /* fast */
            ) != commSuccess) {
      throw ctran::utils::Exception("iput failed", commSystemError);
    }

    do {
      if (ctx.senderIb->progress() != commSuccess) {
        throw ctran::utils::Exception("progress failed", commSystemError);
      }
    } while (!ibReq.isComplete());

    if (withNotify) {
      if (ctx.receiverIb->waitNotify(kDummyRank, 1) != commSuccess) {
        throw ctran::utils::Exception("waitNotify failed", commSystemError);
      }
    }
  }

  // Calculate and report bandwidth using custom counters
  double totalBytes = static_cast<double>(state.iterations()) * bufferSize;
  state.counters["BW_GBps"] =
      benchmark::Counter(totalBytes / 1e9, benchmark::Counter::kIsRate);
  cleanupBenchmarkContext(ctx);
}

static void
benchmarkIget(benchmark::State& state, CtranIbConfig config, bool withNotify) {
  const size_t bufferSize = state.range(0);
  auto ctx = setupBenchmarkContext(bufferSize);

  // Benchmark the iput operation
  for (auto _ : state) {
    CtranIbRequest ibReq;
    if (ctx.receiverIb->iget(
            ctx.sendBuffer, /* sbuf */
            ctx.recvBuffer, /* dbuf */
            ctx.bufferSize, /* len */
            kDummyRank, /* peerRank */
            ctx.receiverRegHdl, /* ibRegElem */
            ctx.ibSendKey, /* remoteAccessKey */
            &config, /* config */
            &ibReq, /* req */
            false /* fast */
            ) != commSuccess) {
      throw ctran::utils::Exception("iget failed", commSystemError);
    }

    do {
      if (ctx.receiverIb->progress() != commSuccess) {
        throw ctran::utils::Exception("progress failed", commSystemError);
      }
    } while (!ibReq.isComplete());
  }

  // Calculate and report bandwidth using custom counters
  double totalBytes = static_cast<double>(state.iterations()) * bufferSize;
  state.counters["BW_GBps"] =
      benchmark::Counter(totalBytes / 1e9, benchmark::Counter::kIsRate);
  cleanupBenchmarkContext(ctx);
}

/**
 * Benchmark CtranIb Iput operation latency with configurable CtranIbConfig
 */
static void BM_CtranIb_IputWithoutNotify(
    benchmark::State& state,
    CtranIbConfig config) {
  benchmarkIput(state, config, false);
}

static void BM_CtranIb_IputWithNotifySpray(
    benchmark::State& state,
    CtranIbConfig config) {
  config.vcMode = NCCL_CTRAN_IB_VC_MODE::spray;
  benchmarkIput(state, config, true);
}

static void BM_CtranIb_IputWithNotifyDqplb(
    benchmark::State& state,
    CtranIbConfig config) {
  config.vcMode = NCCL_CTRAN_IB_VC_MODE::dqplb;
  benchmarkIput(state, config, true);
}

static void BM_CtranIb_IGet(benchmark::State& state, CtranIbConfig config) {
  benchmarkIget(state, config, false);
}

//------------------------------------------------------------------------------
// Multi-put per-arrival benchmark (interleave on vs off)
//
// Issues `numPuts` concurrent CtranIb::iput ops (each `chunkSize` bytes, to
// distinct offsets of the receive buffer) on the single per-peer VC, then
// records the elapsed time at which each put's notify arrives at the receiver
// (notify1_us = first arrival ... notifyN_us = last). Uses only the high-level
// CtranIb iput/progress/checkNotify API on the simple kExternal setup -- no
// multi-VC / control-message transport.
//
// Purpose: expose NCCL_CTRAN_IB_QP_INTERLEAVE_DEVICES_ENABLE, which only has an
// effect when the VC spans >1 NIC (DEVICES_PER_RANK=2, default on GB200). With
// K = MAX_QPS/devices QPs per NIC and a chunk whose QP-scaling sub-chunks
// number <= K, interleave OFF packs each put onto a single NIC (consecutive
// small puts can pile onto the same NIC, leaving the other idle), while
// interleave ON spreads every put across both NICs. Aggregate BW can look
// identical; the per-put arrival times do not.
//------------------------------------------------------------------------------

static void benchmarkMultiPut(benchmark::State& state, int numPuts) {
  const size_t chunkSize = state.range(0);
  // Reuse the kExternal single-VC setup; one contiguous region holds all
  // chunks (setupBenchmarkContext page-aligns and registers the buffers).
  auto ctx = setupBenchmarkContext(numPuts * chunkSize);

  CtranIbConfig config{
      .numQps = 16,
      .qpScalingTh = 524288,
      .qpMsgs = 128,
  };

  std::vector<double> sumDeltaUs(numPuts, 0.0);

  for (auto _ : state) {
    CHECK_EQ(cudaSetDevice(0), cudaSuccess);
    std::vector<CtranIbRequest> putReq(numPuts);
    const auto t0 = std::chrono::steady_clock::now();

    // Issue all puts back-to-back; each notifies the receiver on completion.
    for (int i = 0; i < numPuts; ++i) {
      const void* sbuf =
          static_cast<const char*>(ctx.sendBuffer) + i * chunkSize;
      void* dbuf = static_cast<char*>(ctx.recvBuffer) + i * chunkSize;
      if (ctx.senderIb->iput(
              sbuf,
              dbuf,
              chunkSize,
              kDummyRank,
              ctx.senderRegHdl,
              ctx.ibReceiveKey,
              true /* notify */,
              &config,
              &putReq[i],
              false /* fast */) != commSuccess) {
        throw ctran::utils::Exception("iput failed", commSystemError);
      }
    }

    // Drive sender progress + receiver notify polling; timestamp each arrival.
    int seen = 0;
    std::vector<double> deltaUs(numPuts, 0.0);
    while (seen < numPuts) {
      if (ctx.senderIb->progress() != commSuccess) {
        throw ctran::utils::Exception("progress failed", commSystemError);
      }
      bool notified = false;
      if (ctx.receiverIb->checkNotify(kDummyRank, &notified) != commSuccess) {
        throw ctran::utils::Exception("checkNotify failed", commSystemError);
      }
      if (notified) {
        deltaUs[seen++] = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
      }
    }

    // Drain the sender's outstanding iputs before putReq leaves scope.
    bool allComplete = false;
    while (!allComplete) {
      if (ctx.senderIb->progress() != commSuccess) {
        throw ctran::utils::Exception("progress failed", commSystemError);
      }
      allComplete = true;
      for (int i = 0; i < numPuts; ++i) {
        if (!putReq[i].isComplete()) {
          allComplete = false;
          break;
        }
      }
    }

    for (int i = 0; i < numPuts; ++i) {
      sumDeltaUs[i] += deltaUs[i];
    }
  }

  const double iters = static_cast<double>(state.iterations());
  for (int i = 0; i < numPuts; ++i) {
    state.counters["notify" + std::to_string(i + 1) + "_us"] =
        sumDeltaUs[i] / iters;
  }
  const double totalBytes = iters * numPuts * static_cast<double>(chunkSize);
  state.counters["BW_GBps"] =
      benchmark::Counter(totalBytes / 1e9, benchmark::Counter::kIsRate);
  // Self-document the resolved config in the output row.
  state.counters["interleave"] =
      NCCL_CTRAN_IB_QP_INTERLEAVE_DEVICES_ENABLE ? 1 : 0;
  state.counters["devs"] = NCCL_CTRAN_IB_DEVICES_PER_RANK;

  cleanupBenchmarkContext(ctx);
}

static void BM_CtranIb_MultiPut2(benchmark::State& state) {
  benchmarkMultiPut(state, 2);
}

static void BM_CtranIb_MultiPut4(benchmark::State& state) {
  benchmarkMultiPut(state, 4);
}

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

const size_t kMinBufferSize = 8 * 1024; // 8 KB
const size_t kMaxBufferSize = 256 * 1024 * 1024; // 256 MB

// Register the benchmark with both qpScalingTh configs for comparison
static auto* registered_iput_no_notify_256k =
    benchmark::RegisterBenchmark(
        "BM_CtranIb_IputWithoutNotify_256k",
        BM_CtranIb_IputWithoutNotify,
        config_256k)
        ->RangeMultiplier(2)
        ->Range(kMinBufferSize, kMaxBufferSize)
        ->UseRealTime()
        ->Unit(benchmark::kMicrosecond);

static auto* registered_iput_no_notify_512k =
    benchmark::RegisterBenchmark(
        "BM_CtranIb_IputWithoutNotify_512k",
        BM_CtranIb_IputWithoutNotify,
        config_512k)
        ->RangeMultiplier(2)
        ->Range(kMinBufferSize, kMaxBufferSize)
        ->UseRealTime()
        ->Unit(benchmark::kMicrosecond);

static auto* registered_iput_spray_256k =
    benchmark::RegisterBenchmark(
        "BM_CtranIb_IputWithNotifySpray_256k",
        BM_CtranIb_IputWithNotifySpray,
        config_256k)
        ->RangeMultiplier(2)
        ->Range(kMinBufferSize, kMaxBufferSize)
        ->UseRealTime()
        ->Unit(benchmark::kMicrosecond);

static auto* registered_iput_spray_512k =
    benchmark::RegisterBenchmark(
        "BM_CtranIb_IputWithNotifySpray_512k",
        BM_CtranIb_IputWithNotifySpray,
        config_512k)
        ->RangeMultiplier(2)
        ->Range(kMinBufferSize, kMaxBufferSize)
        ->UseRealTime()
        ->Unit(benchmark::kMicrosecond);

static auto* registered_iput_dqplb_256k =
    benchmark::RegisterBenchmark(
        "BM_CtranIb_IputWithNotifyDqplb_256k",
        BM_CtranIb_IputWithNotifyDqplb,
        config_256k)
        ->RangeMultiplier(2)
        ->Range(kMinBufferSize, kMaxBufferSize)
        ->UseRealTime()
        ->Unit(benchmark::kMicrosecond);

static auto* registered_iput_dqplb_512k =
    benchmark::RegisterBenchmark(
        "BM_CtranIb_IputWithNotifyDqplb_512k",
        BM_CtranIb_IputWithNotifyDqplb,
        config_512k)
        ->RangeMultiplier(2)
        ->Range(kMinBufferSize, kMaxBufferSize)
        ->UseRealTime()
        ->Unit(benchmark::kMicrosecond);

static auto* registered_iget_256k = benchmark::RegisterBenchmark(
                                        "BM_CtranIb_Iget_256k",
                                        BM_CtranIb_IGet,
                                        config_256k)
                                        ->RangeMultiplier(2)
                                        ->Range(kMinBufferSize, kMaxBufferSize)
                                        ->UseRealTime()
                                        ->Unit(benchmark::kMicrosecond);

static auto* registered_iget_512k = benchmark::RegisterBenchmark(
                                        "BM_CtranIb_Iget_512k",
                                        BM_CtranIb_IGet,
                                        config_512k)
                                        ->RangeMultiplier(2)
                                        ->Range(kMinBufferSize, kMaxBufferSize)
                                        ->UseRealTime()
                                        ->Unit(benchmark::kMicrosecond);

// Multi-put per-arrival: 2 and 4 concurrent puts across the
// interleave-sensitive chunk-size range. Run twice -- with
// NCCL_CTRAN_IB_QP_INTERLEAVE_DEVICES_ENABLE 0 then 1 (and
// NCCL_CTRAN_IB_DEVICES_PER_RANK=2) -- and compare the notify*_us columns.
const size_t kMultiPut32K = 32 * 1024;
const size_t kMultiPut64K = 64 * 1024;
const size_t kMultiPut128K = 128 * 1024;
const size_t kMultiPut256K = 256 * 1024;
const size_t kMultiPut512K = 512 * 1024;
const size_t kMultiPut1M = 1 * 1024 * 1024;
const size_t kMultiPut2M = 2 * 1024 * 1024;
const size_t kMultiPut4M = 4 * 1024 * 1024;

static auto* registered_multiput2 =
    benchmark::RegisterBenchmark("BM_CtranIb_MultiPut2", BM_CtranIb_MultiPut2)
        ->Arg(kMultiPut32K)
        ->Arg(kMultiPut64K)
        ->Arg(kMultiPut128K)
        ->Arg(kMultiPut256K)
        ->Arg(kMultiPut512K)
        ->Arg(kMultiPut1M)
        ->Arg(kMultiPut2M)
        ->Arg(kMultiPut4M)
        ->UseRealTime()
        ->Unit(benchmark::kMicrosecond);

static auto* registered_multiput4 =
    benchmark::RegisterBenchmark("BM_CtranIb_MultiPut4", BM_CtranIb_MultiPut4)
        ->Arg(kMultiPut32K)
        ->Arg(kMultiPut64K)
        ->Arg(kMultiPut128K)
        ->Arg(kMultiPut256K)
        ->Arg(kMultiPut512K)
        ->Arg(kMultiPut1M)
        ->Arg(kMultiPut2M)
        ->Arg(kMultiPut4M)
        ->UseRealTime()
        ->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------
// Main Function
//------------------------------------------------------------------------------

// Custom main function to handle initialization
int main(int argc, char** argv) {
  ncclCvarInit();
  ctran::utils::commCudaLibraryInit();

  // Check if we have multiple CUDA devices for transport benchmarks
  int deviceCount;
  if (cudaGetDeviceCount(&deviceCount) == cudaSuccess) {
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    if (deviceCount < 2) {
      std::cout
          << "Warning: Transport benchmarks require at least 2 CUDA devices"
          << std::endl;
    }
  }

  // Initialize and run benchmark
  ::benchmark::Initialize(&argc, argv);
  folly::init(&argc, &argv);
  ::benchmark::RunSpecifiedBenchmarks();

  // Cleanup
  CHECK_EQ(cudaDeviceReset(), cudaSuccess);

  return 0;
}
