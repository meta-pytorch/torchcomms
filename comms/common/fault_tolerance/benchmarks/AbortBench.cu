// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/common/fault_tolerance/benchmarks/AbortBench.cuh"

#include <cuda/atomic>

#include <cstdint>

namespace comms::fault_tolerance::benchmark {
namespace {

using SystemAtomicInt = cuda::atomic_ref<int, cuda::thread_scope_system>;

__device__ bool
waitForCounterAtLeast(SystemAtomicInt& counter, int expected, int maxPolls) {
  for (int polls = 0; polls < maxPolls; ++polls) {
    if (counter.load(cuda::memory_order_acquire) >= expected) {
      return true;
    }
    __nanosleep(64);
  }
  return false;
}

__device__ void recordFirstObserved(int* observed, int value) {
  auto observedValue = SystemAtomicInt{*observed};
  int expected = 0;
  observedValue.compare_exchange_strong(
      expected, value, cuda::memory_order_acq_rel, cuda::memory_order_acquire);
}

__global__ void deviceLoadLoopKernel(int* flag, int* sink, int iterations) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int local = 0;
  auto atomicFlag = SystemAtomicInt{*flag};
  for (int i = 0; i < iterations; ++i) {
    local += atomicFlag.load(cuda::memory_order_acquire);
  }
  *sink = local;
}

__global__ void
manyBlockDeviceLoadLoopKernel(int* flag, int* sink, int iterations) {
  int local = 0;
  auto atomicFlag = SystemAtomicInt{*flag};
  for (int i = 0; i < iterations; ++i) {
    local += atomicFlag.load(cuda::memory_order_acquire);
  }
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  sink[index] = local;
}

__global__ void deviceStoreLoopKernel(int* flag, int iterations) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  auto atomicFlag = SystemAtomicInt{*flag};
  for (int i = 0; i < iterations; ++i) {
    atomicFlag.store(i + 1, cuda::memory_order_release);
  }
}

__global__ void deviceToHostRoundTripKernel(
    int* request,
    int* response,
    int* ready,
    int iterations) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  auto requestCounter = SystemAtomicInt{*request};
  auto responseCounter = SystemAtomicInt{*response};
  SystemAtomicInt{*ready}.store(1, cuda::memory_order_release);
  for (int expected = 1; expected <= iterations; ++expected) {
    while (requestCounter.load(cuda::memory_order_acquire) < expected) {
      __nanosleep(64);
    }
    responseCounter.store(expected, cuda::memory_order_release);
  }
}

__global__ void deviceToDevicePingPongKernel(
    int* request,
    int* response,
    int* ready,
    int* start,
    int* observed,
    int iterations,
    int maxPolls) {
  if (threadIdx.x != 0 || blockIdx.x > 1) {
    return;
  }

  auto requestCounter = SystemAtomicInt{*request};
  auto responseCounter = SystemAtomicInt{*response};
  auto startCounter = SystemAtomicInt{*start};
  SystemAtomicInt{*ready}.fetch_add(1, cuda::memory_order_release);
  if (!waitForCounterAtLeast(startCounter, 1, maxPolls)) {
    recordFirstObserved(observed, -(iterations + 1));
    return;
  }

  if (blockIdx.x == 0) {
    for (int expected = 1; expected <= iterations; ++expected) {
      requestCounter.store(expected, cuda::memory_order_release);
      if (!waitForCounterAtLeast(responseCounter, expected, maxPolls)) {
        recordFirstObserved(observed, -expected);
        return;
      }
    }
    recordFirstObserved(observed, iterations);
    return;
  }

  for (int expected = 1; expected <= iterations; ++expected) {
    if (!waitForCounterAtLeast(requestCounter, expected, maxPolls)) {
      recordFirstObserved(observed, -expected);
      return;
    }
    responseCounter.store(expected, cuda::memory_order_release);
  }
}

__global__ void abortDeviceDefaultTimeoutLoadLoopKernel(
    AbortDevice abort,
    int64_t* sink,
    int iterations) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int64_t local = 0;
  for (int i = 0; i < iterations; ++i) {
    local += abort.getTimeoutMs();
  }
  *sink = local;
}

} // namespace

cudaError_t launchDeviceLoadLoop(
    int* flag,
    int* sink,
    int iterations,
    cudaStream_t stream) {
  deviceLoadLoopKernel<<<1, 1, 0, stream>>>(flag, sink, iterations);
  return cudaGetLastError();
}

cudaError_t launchManyBlockDeviceLoadLoop(
    int* flag,
    int* sink,
    int blocks,
    int threads,
    int iterations,
    cudaStream_t stream) {
  manyBlockDeviceLoadLoopKernel<<<blocks, threads, 0, stream>>>(
      flag, sink, iterations);
  return cudaGetLastError();
}

cudaError_t
launchDeviceStoreLoop(int* flag, int iterations, cudaStream_t stream) {
  deviceStoreLoopKernel<<<1, 1, 0, stream>>>(flag, iterations);
  return cudaGetLastError();
}

cudaError_t launchDeviceToHostRoundTrip(
    int* request,
    int* response,
    int* ready,
    int iterations,
    cudaStream_t stream) {
  deviceToHostRoundTripKernel<<<1, 1, 0, stream>>>(
      request, response, ready, iterations);
  return cudaGetLastError();
}

cudaError_t launchDeviceToDevicePingPong(
    int* request,
    int* response,
    int* ready,
    int* start,
    int* observed,
    int iterations,
    int maxPolls,
    cudaStream_t stream) {
  deviceToDevicePingPongKernel<<<2, 1, 0, stream>>>(
      request, response, ready, start, observed, iterations, maxPolls);
  return cudaGetLastError();
}

cudaError_t launchAbortDeviceDefaultTimeoutLoadLoop(
    AbortDevice abort,
    int64_t* sink,
    int iterations,
    cudaStream_t stream) {
  abortDeviceDefaultTimeoutLoadLoopKernel<<<1, 1, 0, stream>>>(
      abort, sink, iterations);
  return cudaGetLastError();
}

} // namespace comms::fault_tolerance::benchmark
