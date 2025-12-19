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

#include "comms/testinfra/BenchUtils.h"
#include "comms/torchcomms/transport/RdmaTransport.h"

using namespace torch::comms;

//------------------------------------------------------------------------------
// RdmaMemory Benchmarks
//------------------------------------------------------------------------------

enum class RdmaMemoryOperation { READ, WRITE };

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

static void RdmaTransport_run_benchmark(
    uint32_t iters,
    size_t bufferSize,
    folly::UserCounters& counters,
    RdmaMemoryOperation op) {
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

  folly::stop_watch<std::chrono::microseconds> timer;
  // Create remote buffers
  auto resvRemoteBuffer =
      RdmaRemoteBuffer{.ptr = recvBuffer, .accessKey = recvMemory->remoteKey()};
  auto senderRemoteBuffer =
      RdmaRemoteBuffer{.ptr = sendBuffer, .accessKey = sendMemory->remoteKey()};

  // Create memory views
  auto sendMemoryView = sendMemory->createView(sendBuffer, bufferSize);
  auto recvMemoryView = recvMemory->createMutableView(recvBuffer, bufferSize);

  //
  // Benchmark the operations
  //
  if (op == RdmaMemoryOperation::WRITE) {
    for (uint32_t i = 0; i < iters; ++i) {
      sender->write(sendMemoryView, resvRemoteBuffer, false).get();
    }
  } else if (op == RdmaMemoryOperation::READ) {
    for (uint32_t i = 0; i < iters; ++i) {
      receiver->read(recvMemoryView, senderRemoteBuffer).get();
    }
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
/**
 * Benchmark RdmaTransport write operation latency
 */
static void RdmaTransport_Write(
    uint32_t iters,
    size_t bufferSize,
    folly::UserCounters& counters) {
  RdmaTransport_run_benchmark(
      iters, bufferSize, counters, RdmaMemoryOperation::WRITE);
}

/**
 * Benchmark RdmaTransport read operation latency
 */
static void RdmaTransport_Read(
    uint32_t iters,
    size_t bufferSize,
    folly::UserCounters& counters) {
  RdmaTransport_run_benchmark(
      iters, bufferSize, counters, RdmaMemoryOperation::WRITE);
}

BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Write, 8192);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Write, 16384);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Write, 32768);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Write, 65536);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Write, 131072);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Write, 262144);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Write, 524288);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Write, 1048576);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Write, 2097152);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Write, 4194304);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Write, 8388608);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Write, 16777216);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Write, 33554432);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Write, 67108864);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Write, 134217728);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Write, 268435456);

BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Read, 8192);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Read, 16384);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Read, 32768);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Read, 65536);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Read, 131072);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Read, 262144);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Read, 524288);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Read, 1048576);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Read, 2097152);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Read, 4194304);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Read, 8388608);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Read, 16777216);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Read, 33554432);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Read, 67108864);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Read, 134217728);
BENCHMARK_SINGLE_PARAM_COUNTERS(RdmaTransport_Read, 268435456);

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
