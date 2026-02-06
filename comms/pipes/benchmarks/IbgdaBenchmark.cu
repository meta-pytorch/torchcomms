// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/benchmarks/IbgdaBenchmark.cuh"

#include <cuda_runtime.h>

namespace comms::pipes::benchmark {

__global__ void ibgdaPutSignalWaitLocalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal) {
  // Only first thread performs the operation
  if (threadIdx.x == 0 && blockIdx.x == 0) {
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
  // Only first thread performs the operation
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto work = transport->put_signal_non_adaptive(
        localBuf, remoteBuf, nbytes, signalId, signalVal);
    transport->wait_local(work);
  }
}

__global__ void ibgdaWaitSignalKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    IbgdaCmpOp cmpOp,
    uint64_t expectedSignal) {
  // Only first thread waits for signal
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    transport->wait_signal(signalId, cmpOp, expectedSignal);
  }
}

__global__ void ibgdaSignalOnlyKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    uint64_t signalVal) {
  // Only first thread sends signal
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto work = transport->signal(signalId, signalVal);
    transport->wait_local(work);
  }
}

__global__ void ibgdaResetSignalKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto work = transport->reset_signal(signalId);
    transport->wait_local(work);
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

void launchIbgdaPutSignalNonAdaptiveWaitLocal(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int numThreads,
    cudaStream_t stream) {
  ibgdaPutSignalNonAdaptiveWaitLocalKernel<<<
      numBlocks,
      numThreads,
      0,
      stream>>>(transport, localBuf, remoteBuf, nbytes, signalId, signalVal);
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

} // namespace comms::pipes::benchmark
