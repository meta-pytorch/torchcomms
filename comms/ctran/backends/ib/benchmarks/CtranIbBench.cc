// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <memory>

#include <folly/init/Init.h>

#include "comms/ctran/backends/ib/CtranIb.h"
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
CtranIbConfig config{
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

  // Initialize senderIb and receiverIb
  auto senderIb = std::make_unique<CtranIb>(
      kDummyRank,
      cudaDev0,
      -1 /* commHash */,
      "RDMA-Transport",
      nullptr /* ctrlManager */,
      true /* enableLocalFlush */,
      CtranIb::BootstrapMode::kExternal);
  auto receiverIb = std::make_unique<CtranIb>(
      kDummyRank,
      cudaDev1,
      -1 /* commHash */,
      "RDMA-Transport",
      nullptr /* ctrlManager */,
      true /* enableLocalFlush */,
      CtranIb::BootstrapMode::kExternal);

  // Connect senderIb and receiverIb
  auto senderVcIdentifier = senderIb->getLocalVcIdentifier(kDummyRank);
  auto receiverVcIdentifier = receiverIb->getLocalVcIdentifier(kDummyRank);
  CHECK_EQ(
      senderIb->connectVcDirect(receiverVcIdentifier, kDummyRank), commSuccess);
  CHECK_EQ(
      receiverIb->connectVcDirect(senderVcIdentifier, kDummyRank), commSuccess);

  // Allocate memory on the sender side
  CHECK_EQ(cudaSetDevice(cudaDev0), cudaSuccess);
  void* sendBuffer = nullptr;
  CHECK_EQ(cudaMalloc(&sendBuffer, bufferSize), cudaSuccess);
  void* senderRegHdl = nullptr;
  if (CtranIb::regMem(sendBuffer, bufferSize, cudaDev0, &senderRegHdl) !=
      commSuccess) {
    throw ctran::utils::Exception(
        "regMem failed for sendBuffer", commSystemError);
  }

  // Allocate memory on the receiver side
  CHECK_EQ(cudaSetDevice(cudaDev1), cudaSuccess);
  void* recvBuffer = nullptr;
  CHECK_EQ(cudaMalloc(&recvBuffer, bufferSize), cudaSuccess);
  void* receiverRegHdl = nullptr;
  if (CtranIb::regMem(recvBuffer, bufferSize, cudaDev1, &receiverRegHdl) !=
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

  CHECK_EQ(cudaFree(ctx.sendBuffer), cudaSuccess);
  CHECK_EQ(cudaFree(ctx.recvBuffer), cudaSuccess);
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
// Benchmark Registration
//------------------------------------------------------------------------------

const size_t kMinBufferSize = 8 * 1024; // 8 KB
const size_t kMaxBufferSize = 256 * 1024 * 1024; // 256 MB

// Register the benchmark with the config as a parameter
static auto* registered_benchmark = benchmark::RegisterBenchmark(
                                        "BM_CtranIb_IputWithoutNotify",
                                        BM_CtranIb_IputWithoutNotify,
                                        config)
                                        ->RangeMultiplier(2)
                                        ->Range(kMinBufferSize, kMaxBufferSize)
                                        ->UseRealTime()
                                        ->Unit(benchmark::kMicrosecond);

static auto* registered_benchmark_with_notify_spray =
    benchmark::RegisterBenchmark(
        "BM_CtranIb_IputWithNotifySpray",
        BM_CtranIb_IputWithNotifySpray,
        config)
        ->RangeMultiplier(2)
        ->Range(kMinBufferSize, kMaxBufferSize)
        ->UseRealTime()
        ->Unit(benchmark::kMicrosecond);

static auto* registered_benchmark_with_notify_dqplb =
    benchmark::RegisterBenchmark(
        "BM_CtranIb_IputWithNotifyDqplb",
        BM_CtranIb_IputWithNotifyDqplb,
        config)
        ->RangeMultiplier(2)
        ->Range(kMinBufferSize, kMaxBufferSize)
        ->UseRealTime()
        ->Unit(benchmark::kMicrosecond);

static auto* registered_benchmark_iget =
    benchmark::RegisterBenchmark("BM_CtranIb_Iget", BM_CtranIb_IGet, config)
        ->RangeMultiplier(2)
        ->Range(kMinBufferSize, kMaxBufferSize)
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
