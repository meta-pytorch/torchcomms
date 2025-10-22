// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <memory>

#include <folly/futures/Future.h>
#include <folly/io/async/EventBase.h>
#include <folly/io/async/ScopedEventBaseThread.h>

#include "comms/torchcomms/transport/RdmaTransport.h"

using namespace torch::comms;

//------------------------------------------------------------------------------
// RdmaMemory Benchmarks
//------------------------------------------------------------------------------

/**
 * Benchmark RdmaMemory creation with different buffer sizes
 */
static void BM_RdmaMemory_Register(benchmark::State& state) {
  const size_t bufferSize = state.range(0);
  const int cudaDev = 0;
  CHECK_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  void* buffer = nullptr;
  CHECK_EQ(cudaMalloc(&buffer, bufferSize), cudaSuccess);

  for (auto _ : state) {
    auto memory = std::make_unique<RdmaMemory>(buffer, bufferSize, cudaDev);
    benchmark::DoNotOptimize(memory->localKey());
    state.PauseTiming();
    memory.reset(); // Destroy the memory object
    state.ResumeTiming();
  }

  cudaFree(buffer);
}

/**
 * Benchmark RdmaMemory destruction time
 */
static void BM_RdmaMemory_Deregister(benchmark::State& state) {
  const size_t bufferSize = state.range(0);
  const int cudaDev = 0;
  CHECK_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  void* buffer = nullptr;
  CHECK_EQ(cudaMalloc(&buffer, bufferSize), cudaSuccess);

  for (auto _ : state) {
    state.PauseTiming();
    auto memory = std::make_unique<RdmaMemory>(buffer, bufferSize, cudaDev);
    state.ResumeTiming();
    benchmark::DoNotOptimize(memory->localKey());
    memory.reset();
  }

  cudaFree(buffer);
}

//------------------------------------------------------------------------------
// RdmaTransport Benchmarks
//------------------------------------------------------------------------------

/**
 * Benchmark RdmaTransport write operation latency
 */
static void BM_RdmaTransport_Write(benchmark::State& state) {
  const size_t bufferSize = state.range(0);
  const int cudaDev0 = 0;
  const int cudaDev1 = 1;

  // Setup event base thread
  auto evbThread = std::make_unique<folly::ScopedEventBaseThread>();
  auto evb = evbThread->getEventBase();

  // Setup P2P transport
  auto sender = std::make_unique<RdmaTransport>(cudaDev0, evb);
  auto receiver = std::make_unique<RdmaTransport>(cudaDev1, evb);
  const auto senderUrl = sender->bind();
  const auto receiverUrl = receiver->bind();
  sender->connect(receiverUrl);
  receiver->connect(senderUrl);

  // Allocate memory on the sender side
  CHECK_EQ(cudaSetDevice(cudaDev0), cudaSuccess);
  void* sendBuffer = nullptr;
  CHECK_EQ(cudaMalloc(&sendBuffer, bufferSize), cudaSuccess);
  auto sendMemory =
      std::make_unique<RdmaMemory>(sendBuffer, bufferSize, cudaDev0);

  // Allocate memory on the receiver side
  CHECK_EQ(cudaSetDevice(cudaDev1), cudaSuccess);
  void* recvBuffer = nullptr;
  CHECK_EQ(cudaMalloc(&recvBuffer, bufferSize), cudaSuccess);
  auto recvMemory =
      std::make_unique<RdmaMemory>(recvBuffer, bufferSize, cudaDev1);
  auto remoteBuffer =
      RdmaRemoteBuffer{.ptr = recvBuffer, .accessKey = recvMemory->remoteKey()};

  //
  // Benchmark the write operation
  //
  for (auto _ : state) {
    sender
        ->write(
            sendMemory->createView(sendBuffer, bufferSize), remoteBuffer, false)
        .get();
  }

  state.SetBytesProcessed(state.iterations() * bufferSize);

  sendMemory.reset();
  recvMemory.reset();
  cudaFree(sendBuffer);
  cudaFree(recvBuffer);
}

//------------------------------------------------------------------------------
// Benchmarks
//------------------------------------------------------------------------------

const size_t kMinBufferSize = 8 * 1024; // 8 KB
const size_t kMaxBufferSizeMemory = 16 * 1024; // 16 KB
const size_t kMaxBufferSizeWrite = 256 * 1024 * 1024; // 256 MB

BENCHMARK(BM_RdmaMemory_Register)
    ->RangeMultiplier(2)
    ->Range(kMinBufferSize, kMaxBufferSizeMemory)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_RdmaMemory_Deregister)
    ->RangeMultiplier(2)
    ->Range(kMinBufferSize, kMaxBufferSizeMemory)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_RdmaTransport_Write)
    ->RangeMultiplier(2)
    ->Range(kMinBufferSize, kMaxBufferSizeWrite)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

// Custom main function to handle initialization
int main(int argc, char** argv) {
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
  cudaDeviceReset();

  return 0;
}
