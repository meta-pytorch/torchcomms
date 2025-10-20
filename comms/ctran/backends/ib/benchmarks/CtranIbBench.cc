// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <memory>

#include "comms/ctran/backends/ib/CtranIb.h"

using namespace ctran;

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
      CtranIbRemoteAccessKey ibReceiveKey_,
      size_t bufferSize_)
      : senderIb(std::move(senderIb_)),
        receiverIb(std::move(receiverIb_)),
        sendBuffer(sendBuffer_),
        recvBuffer(recvBuffer_),
        senderRegHdl(senderRegHdl_),
        receiverRegHdl(receiverRegHdl_),
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
    throw std::runtime_error("regMem failed for sendBuffer");
  }

  // Allocate memory on the receiver side
  CHECK_EQ(cudaSetDevice(cudaDev1), cudaSuccess);
  void* recvBuffer = nullptr;
  CHECK_EQ(cudaMalloc(&recvBuffer, bufferSize), cudaSuccess);
  void* receiverRegHdl = nullptr;
  if (CtranIb::regMem(recvBuffer, bufferSize, cudaDev1, &receiverRegHdl) !=
      commSuccess) {
    throw std::runtime_error("regMem failed for recvBuffer");
  }

  // Check connection
  if (senderIb->getVc(kDummyRank) == nullptr) {
    throw std::runtime_error("senderIb not connected");
  }
  if (receiverIb->getVc(kDummyRank) == nullptr) {
    throw std::runtime_error("receiverIb not connected");
  }

  auto ibReceiveKey = CtranIb::getRemoteAccessKey(receiverRegHdl);

  return BenchmarkContext(
      std::move(senderIb),
      std::move(receiverIb),
      sendBuffer,
      recvBuffer,
      senderRegHdl,
      receiverRegHdl,
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
      throw std::runtime_error("iput failed");
    }

    do {
      if (ctx.senderIb->progress() != commSuccess) {
        throw std::runtime_error("progress failed");
      }
    } while (!ibReq.isComplete());

    if (withNotify) {
      if (ctx.receiverIb->waitNotify(kDummyRank, 1) != commSuccess) {
        throw std::runtime_error("waitNotify failed");
      }
    }
  }

  state.SetBytesProcessed(state.iterations() * bufferSize);
  cleanupBenchmarkContext(ctx);
}

/**
 * Benchmark CtranIb Iput operation latency with configurable CtranIbConfig
 */
static void BM_CtranIb_Iput(benchmark::State& state, CtranIbConfig config) {
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

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

const size_t kMinBufferSize = 8 * 1024; // 8 KB
const size_t kMaxBufferSize = 256 * 1024 * 1024; // 256 MB

// Register the benchmark with the config as a parameter
static auto* registered_benchmark =
    benchmark::RegisterBenchmark("BM_CtranIb_Iput", BM_CtranIb_Iput, config)
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
  ::benchmark::RunSpecifiedBenchmarks();

  // Cleanup
  CHECK_EQ(cudaDeviceReset(), cudaSuccess);

  return 0;
}
