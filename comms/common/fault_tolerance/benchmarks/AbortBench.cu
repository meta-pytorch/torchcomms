// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/common/fault_tolerance/benchmarks/AbortBench.cuh"

#include <cuda/atomic>

#include <cassert>
#include <cstdint>

namespace comms::fault_tolerance::benchmark {
namespace {

using SystemAtomicInt = cuda::atomic_ref<int, cuda::thread_scope_system>;

__device__ bool waitForCounterAtLeast(
    SystemAtomicInt& counter,
    int expected,
    uint64_t deadlineCycles) {
  while (detail::deviceClock() < deadlineCycles) {
    if (counter.load(cuda::memory_order_acquire) >= expected) {
      return true;
    }
    __nanosleep(64);
  }
  return false;
}

__device__ void recordFirstObserved(int* observed, int value) {
  assert(value != 0);
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
    int* observed,
    int iterations,
    uint64_t maxWaitCycles) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  auto requestCounter = SystemAtomicInt{*request};
  auto responseCounter = SystemAtomicInt{*response};
  const auto deadlineCycles = detail::deviceClock() + maxWaitCycles;
  SystemAtomicInt{*ready}.store(1, cuda::memory_order_release);
  for (int expected = 1; expected <= iterations; ++expected) {
    if (!waitForCounterAtLeast(requestCounter, expected, deadlineCycles)) {
      recordFirstObserved(observed, -expected);
      return;
    }
    responseCounter.store(expected, cuda::memory_order_release);
  }
  recordFirstObserved(observed, iterations);
}

__global__ void deviceToDevicePingPongKernel(
    int* request,
    int* response,
    int* ready,
    int* start,
    int* observed,
    int iterations,
    uint64_t maxWaitCycles) {
  if (threadIdx.x != 0 || blockIdx.x >= kPingPongBlocks) {
    return;
  }

  // Each block owns one slot. Sharing a single slot let a late timeout in the
  // responder lose the CAS to the requester's success and disappear: the wait
  // helper checks the deadline before the counter, so the responder can report
  // failure on its final wait even though the exchange had already completed.
  int* status = observed + blockIdx.x;

  auto requestCounter = SystemAtomicInt{*request};
  auto responseCounter = SystemAtomicInt{*response};
  auto startCounter = SystemAtomicInt{*start};
  const auto deadlineCycles = detail::deviceClock() + maxWaitCycles;
  SystemAtomicInt{*ready}.fetch_add(1, cuda::memory_order_release);
  if (!waitForCounterAtLeast(startCounter, 1, deadlineCycles)) {
    recordFirstObserved(status, -(iterations + 1));
    return;
  }

  if (blockIdx.x == 0) {
    for (int expected = 1; expected <= iterations; ++expected) {
      requestCounter.store(expected, cuda::memory_order_release);
      if (!waitForCounterAtLeast(responseCounter, expected, deadlineCycles)) {
        recordFirstObserved(status, -expected);
        return;
      }
    }
    recordFirstObserved(status, iterations);
    return;
  }

  for (int expected = 1; expected <= iterations; ++expected) {
    if (!waitForCounterAtLeast(requestCounter, expected, deadlineCycles)) {
      recordFirstObserved(status, -expected);
      return;
    }
    responseCounter.store(expected, cuda::memory_order_release);
  }
  recordFirstObserved(status, iterations);
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

__global__ void abortDeviceIsAbortedLoadLoopKernel(
    AbortDevice abort,
    int* sink,
    int iterations,
    bool startTimeout) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (startTimeout) {
    abort.startTimeout();
  }

  int local = 0;
  for (int i = 0; i < iterations; ++i) {
    local += abort.isAborted() ? 1 : 0;
  }
  *sink = local;
}

__global__ void abortDeviceArmOnlyKernel(AbortDevice abort, uint64_t* sink) {
  // Deliberately NOT leader-gated: the collectives arm on every thread (see
  // `abortDevice.start()` at the top of the AllReduce tree/ring kernels), and
  // reproducing that is the point of this benchmark.
  abort.startTimeout();
  // Consume the result so the arm cannot be optimized away, without adding a
  // store on the measured path: the deadline is a clock value and is never 1.
  if (abort.deadlineCycles() == 1) {
    *sink = 1;
  }
}

} // namespace

cudaError_t launchAbortDeviceArmOnly(
    AbortDevice abort,
    uint64_t* sink,
    int blocks,
    int threads,
    cudaStream_t stream) {
  abortDeviceArmOnlyKernel<<<blocks, threads, 0, stream>>>(abort, sink);
  return cudaGetLastError();
}

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
    int* observed,
    int iterations,
    uint64_t maxWaitCycles,
    cudaStream_t stream) {
  deviceToHostRoundTripKernel<<<1, 1, 0, stream>>>(
      request, response, ready, observed, iterations, maxWaitCycles);
  return cudaGetLastError();
}

cudaError_t launchDeviceToDevicePingPong(
    int* request,
    int* response,
    int* ready,
    int* start,
    int* observed,
    int iterations,
    uint64_t maxWaitCycles,
    cudaStream_t stream) {
  deviceToDevicePingPongKernel<<<kPingPongBlocks, 1, 0, stream>>>(
      request, response, ready, start, observed, iterations, maxWaitCycles);
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

cudaError_t launchAbortDeviceIsAbortedLoadLoop(
    AbortDevice abort,
    int* sink,
    int iterations,
    bool startTimeout,
    cudaStream_t stream) {
  abortDeviceIsAbortedLoadLoopKernel<<<1, 1, 0, stream>>>(
      abort, sink, iterations, startTimeout);
  return cudaGetLastError();
}

} // namespace comms::fault_tolerance::benchmark
