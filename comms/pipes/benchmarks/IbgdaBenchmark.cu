// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/benchmarks/IbgdaBenchmark.cuh"

#include <cuda_runtime.h>

#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes::benchmark {

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

__global__ void ibgdaWaitSignalKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    IbgdaCmpOp cmpOp,
    uint64_t expectedSignal) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    transport->wait_signal(signalId, cmpOp, expectedSignal);
  }
}

__global__ void ibgdaSignalOnlyKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    uint64_t signalVal) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    auto work = transport->signal(signalId, signalVal);
    transport->wait_local(work);
  }
}

__global__ void ibgdaResetSignalKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    // reset_signal is now synchronous (includes fences and wait internally)
    transport->reset_signal(signalId);
  }
}

__global__ void ibgdaPutWaitLocalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes) {
  auto group = make_block_group();
  if (group.is_global_leader()) {
    auto work = transport->put(localBuf, remoteBuf, nbytes);
    transport->wait_local(work);
  }
}

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

void launchIbgdaPutSignalWaitLocal(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int numThreads,
    cudaStream_t stream) {
  ibgdaPutSignalWaitLocalKernel<<<numBlocks, numThreads, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytes, signalId, signalVal);
}

void launchIbgdaWaitSignal(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    IbgdaCmpOp cmpOp,
    uint64_t expectedSignal,
    int numBlocks,
    int numThreads,
    cudaStream_t stream) {
  ibgdaWaitSignalKernel<<<numBlocks, numThreads, 0, stream>>>(
      transport, signalId, cmpOp, expectedSignal);
}

void launchIbgdaSignalOnly(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int numThreads,
    cudaStream_t stream) {
  ibgdaSignalOnlyKernel<<<numBlocks, numThreads, 0, stream>>>(
      transport, signalId, signalVal);
}

void launchIbgdaResetSignal(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    cudaStream_t stream) {
  ibgdaResetSignalKernel<<<1, 1, 0, stream>>>(transport, signalId);
}

void launchIbgdaPutWaitLocal(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int numBlocks,
    int numThreads,
    cudaStream_t stream) {
  ibgdaPutWaitLocalKernel<<<numBlocks, numThreads, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytes);
}

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
