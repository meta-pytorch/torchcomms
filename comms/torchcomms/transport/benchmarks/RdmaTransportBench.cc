// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <iostream>
#include <memory>

#include <folly/Benchmark.h>
#include <folly/futures/Future.h>
#include <folly/init/Init.h>
#include <folly/io/async/EventBase.h>
#include <folly/io/async/ScopedEventBaseThread.h>
#include <folly/stop_watch.h>

#include "comms/torchcomms/transport/RdmaTransport.h"

using namespace torch::comms;

//------------------------------------------------------------------------------
// RdmaMemory Benchmarks
//------------------------------------------------------------------------------

/**
 * Benchmark RdmaMemory creation with different buffer sizes
 */
static void RdmaMemory_Register_Deregister(uint32_t iters, size_t bufferSize) {
  const int cudaDev = 0;
  void* buffer = nullptr;

  BENCHMARK_SUSPEND {
    CHECK_EQ(cudaSetDevice(cudaDev), cudaSuccess);
    CHECK_EQ(cudaMalloc(&buffer, bufferSize), cudaSuccess);
  }

  for (uint32_t i = 0; i < iters; ++i) {
    auto memory = std::make_unique<RdmaMemory>(buffer, bufferSize, cudaDev);
    folly::doNotOptimizeAway(memory->localKey());
    memory.reset(); // Destroy the memory object
  }

  BENCHMARK_SUSPEND {
    cudaFree(buffer);
  }
}
BENCHMARK_PARAM(RdmaMemory_Register_Deregister, 8192);
// BENCHMARK_PARAM(RdmaMemory_Register_Deregister, 1024 * 1024);

/**
 * Benchmark RdmaTransport write operation latency
 */
static void RdmaTransport_Write(
    uint32_t iters,
    size_t bufferSize,
    folly::UserCounters& counters) {
  const int cudaDev0 = 0;
  const int cudaDev1 = 1;
  std::unique_ptr<RdmaTransport> sender, receiver;
  std::unique_ptr<folly::ScopedEventBaseThread> evbThread;
  void* sendBuffer = nullptr;
  void* recvBuffer = nullptr;
  std::unique_ptr<RdmaMemory> sendMemory, recvMemory;

  BENCHMARK_SUSPEND {
    // Setup event base thread
    evbThread = std::make_unique<folly::ScopedEventBaseThread>();
    auto evb = evbThread->getEventBase();

    // Setup P2P transport
    sender = std::make_unique<RdmaTransport>(cudaDev0, evb);
    receiver = std::make_unique<RdmaTransport>(cudaDev1, evb);
    const auto senderUrl = sender->bind();
    const auto receiverUrl = receiver->bind();
    sender->connect(receiverUrl);
    receiver->connect(senderUrl);

    // Allocate memory on the sender side
    CHECK_EQ(cudaSetDevice(cudaDev0), cudaSuccess);
    CHECK_EQ(cudaMalloc(&sendBuffer, bufferSize), cudaSuccess);
    sendMemory = std::make_unique<RdmaMemory>(sendBuffer, bufferSize, cudaDev0);

    // Allocate memory on the receiver side
    CHECK_EQ(cudaSetDevice(cudaDev1), cudaSuccess);
    CHECK_EQ(cudaMalloc(&recvBuffer, bufferSize), cudaSuccess);
    recvMemory = std::make_unique<RdmaMemory>(recvBuffer, bufferSize, cudaDev1);
  }
  auto remoteBuffer =
      RdmaRemoteBuffer{.ptr = recvBuffer, .accessKey = recvMemory->remoteKey()};
  folly::stop_watch<std::chrono::microseconds> timer;

  //
  // Benchmark the write operation
  //
  for (uint32_t i = 0; i < iters; ++i) {
    sender
        ->write(
            sendMemory->createView(sendBuffer, bufferSize), remoteBuffer, false)
        .get();
  }

  BENCHMARK_SUSPEND {
    size_t bytesPerSec =
        (iters * bufferSize) * 1000 * 1000 / timer.elapsed().count();
    counters["bytes_per_second"] =
        folly::UserMetric(bytesPerSec, folly::UserMetric::Type::METRIC);
    counters["message_size"] =
        folly::UserMetric(bufferSize, folly::UserMetric::Type::METRIC);
    sendMemory.reset();
    recvMemory.reset();
    cudaFree(sendBuffer);
    cudaFree(recvBuffer);
    sender.reset();
    receiver.reset();
  }
}

#define BENCHMARK_PARAM_COUNTERS(name, param)                     \
  BENCHMARK_IMPL_COUNTERS(                                        \
      FB_CONCATENATE(name, FB_CONCATENATE(_, param)),             \
      FOLLY_PP_STRINGIZE(name) "(" FOLLY_PP_STRINGIZE(param) ")", \
      counters,                                                   \
      iters,                                                      \
      unsigned,                                                   \
      iters) {                                                    \
    name(iters, param, counters);                                 \
  }

BENCHMARK_PARAM_COUNTERS(RdmaTransport_Write, 8192);
BENCHMARK_PARAM_COUNTERS(RdmaTransport_Write, 16384);
BENCHMARK_PARAM_COUNTERS(RdmaTransport_Write, 32768);
BENCHMARK_PARAM_COUNTERS(RdmaTransport_Write, 65536);
BENCHMARK_PARAM_COUNTERS(RdmaTransport_Write, 131072);
BENCHMARK_PARAM_COUNTERS(RdmaTransport_Write, 262144);
BENCHMARK_PARAM_COUNTERS(RdmaTransport_Write, 524288);
BENCHMARK_PARAM_COUNTERS(RdmaTransport_Write, 1048576);
BENCHMARK_PARAM_COUNTERS(RdmaTransport_Write, 2097152);
BENCHMARK_PARAM_COUNTERS(RdmaTransport_Write, 4194304);
BENCHMARK_PARAM_COUNTERS(RdmaTransport_Write, 8388608);
BENCHMARK_PARAM_COUNTERS(RdmaTransport_Write, 16777216);
BENCHMARK_PARAM_COUNTERS(RdmaTransport_Write, 33554432);
BENCHMARK_PARAM_COUNTERS(RdmaTransport_Write, 67108864);
BENCHMARK_PARAM_COUNTERS(RdmaTransport_Write, 134217728);
BENCHMARK_PARAM_COUNTERS(RdmaTransport_Write, 268435456);

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
  folly::Init init(&argc, &argv);
  folly::runBenchmarks();

  // Cleanup
  cudaDeviceReset();

  return 0;
}
