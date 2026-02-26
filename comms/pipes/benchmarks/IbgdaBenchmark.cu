// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/benchmarks/IbgdaBenchmark.cuh"

#include <cuda_runtime.h>

#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes::benchmark {

// Single-shot kernel implementations for correctness verification.
// Each kernel does exactly one put_signal + wait_local, no warmup, no loop.

__global__ void ibgdaPutSignalWaitLocalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    auto work =
        transport->put_signal(localBuf, remoteBuf, nbytes, signalId, signalVal);
    transport->wait_local(work);
  }
}

__global__ void ibgdaPutSignalNonAdaptiveWaitLocalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    auto work = transport->put_signal_non_adaptive(
        localBuf, remoteBuf, nbytes, signalId, signalVal);
    transport->wait_local(work);
  }
}

// Batched kernel implementations - these run multiple iterations in a single
// kernel launch to exclude kernel launch overhead and use GPU cycle counters
// for accurate timing.

__global__ void ibgdaPutWaitLocalBatchKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    // Warmup - do a few iterations to warm up the path
    for (int i = 0; i < 10; i++) {
      auto work = transport->put(localBuf, remoteBuf, nbytes);
      transport->wait_local(work);
    }

    // Timed iterations using GPU cycle counter
    unsigned long long startCycle = clock64();

    for (int i = 0; i < numIters; i++) {
      auto work = transport->put(localBuf, remoteBuf, nbytes);
      transport->wait_local(work);
    }

    unsigned long long endCycle = clock64();
    *totalCycles = endCycle - startCycle;
  }
}

__global__ void ibgdaPutSignalWaitLocalBatchKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    // Warmup - do a few iterations to warm up the path
    for (int i = 0; i < 10; i++) {
      auto work =
          transport->put_signal(localBuf, remoteBuf, nbytes, signalId, 1);
      transport->wait_local(work);
    }

    // Timed iterations using GPU cycle counter
    unsigned long long startCycle = clock64();

    for (int i = 0; i < numIters; i++) {
      auto work =
          transport->put_signal(localBuf, remoteBuf, nbytes, signalId, 1);
      transport->wait_local(work);
    }

    unsigned long long endCycle = clock64();
    *totalCycles = endCycle - startCycle;
  }
}

__global__ void ibgdaPutSignalNonAdaptiveWaitLocalBatchKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    // Warmup - do a few iterations to warm up the path
    for (int i = 0; i < 10; i++) {
      auto work = transport->put_signal_non_adaptive(
          localBuf, remoteBuf, nbytes, signalId, 1);
      transport->wait_local(work);
    }

    // Timed iterations using GPU cycle counter
    unsigned long long startCycle = clock64();

    for (int i = 0; i < numIters; i++) {
      auto work = transport->put_signal_non_adaptive(
          localBuf, remoteBuf, nbytes, signalId, 1);
      transport->wait_local(work);
    }

    unsigned long long endCycle = clock64();
    *totalCycles = endCycle - startCycle;
  }
}

__global__ void ibgdaSignalOnlyBatchKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    int numIters,
    unsigned long long* totalCycles) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    // Warmup - do a few iterations to warm up the path
    for (int i = 0; i < 10; i++) {
      auto work = transport->signal(signalId, 1);
      transport->wait_local(work);
    }

    // Timed iterations using GPU cycle counter
    unsigned long long startCycle = clock64();

    for (int i = 0; i < numIters; i++) {
      auto work = transport->signal(signalId, 1);
      transport->wait_local(work);
    }

    unsigned long long endCycle = clock64();
    *totalCycles = endCycle - startCycle;
  }
}

// Launch wrapper implementations

// Single-shot launchers for correctness verification (exactly 1 put_signal)

void launchIbgdaPutSignalSingle(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    cudaStream_t stream) {
  ibgdaPutSignalWaitLocalKernel<<<1, 32, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytes, signalId, 1);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

void launchIbgdaPutSignalNonAdaptiveSingle(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    cudaStream_t stream) {
  ibgdaPutSignalNonAdaptiveWaitLocalKernel<<<1, 32, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytes, signalId, 1);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Kernel launch failed: ") + cudaGetErrorString(err));
  }
}

// Batched launchers for performance measurement

void launchIbgdaPutWaitLocalBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream) {
  ibgdaPutWaitLocalBatchKernel<<<1, 32, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytes, numIters, totalCycles);
}

void launchIbgdaPutSignalWaitLocalBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream) {
  ibgdaPutSignalWaitLocalBatchKernel<<<1, 32, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytes, signalId, numIters, totalCycles);
}

void launchIbgdaPutSignalNonAdaptiveWaitLocalBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream) {
  ibgdaPutSignalNonAdaptiveWaitLocalBatchKernel<<<1, 32, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytes, signalId, numIters, totalCycles);
}

void launchIbgdaSignalOnlyBatch(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    int numIters,
    unsigned long long* totalCycles,
    cudaStream_t stream) {
  ibgdaSignalOnlyBatchKernel<<<1, 32, 0, stream>>>(
      transport, signalId, numIters, totalCycles);
}

} // namespace comms::pipes::benchmark
