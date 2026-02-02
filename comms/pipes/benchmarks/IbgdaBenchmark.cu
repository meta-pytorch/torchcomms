// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/benchmarks/IbgdaBenchmark.cuh"

#include <cuda_runtime.h>

namespace comms::pipes::benchmark {

__global__ void ibgdaPutSignalWaitLocalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    uint64_t signalVal) {
  // Only first thread performs the operation
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto work = transport->put_signal(localBuf, remoteBuf, nbytes, signalVal);
    transport->wait_local(work);
  }
}

__global__ void ibgdaWaitSignalKernel(
    P2pIbgdaTransportDevice* transport,
    uint64_t expectedSignal) {
  // Only first thread waits for signal
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    transport->wait_signal(expectedSignal);
  }
}

__global__ void ibgdaSignalOnlyKernel(
    P2pIbgdaTransportDevice* transport,
    uint64_t signalVal) {
  // Only first thread sends signal
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto work = transport->signal(signalVal);
    transport->wait_local(work);
  }
}

__global__ void ibgdaResetSignalKernel(P2pIbgdaTransportDevice* transport) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    transport->reset_signal();
  }
}

// Launch wrapper implementations

void launchIbgdaPutSignalWaitLocal(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    uint64_t signalVal,
    int numBlocks,
    int numThreads,
    cudaStream_t stream) {
  ibgdaPutSignalWaitLocalKernel<<<numBlocks, numThreads, 0, stream>>>(
      transport, localBuf, remoteBuf, nbytes, signalVal);
}

void launchIbgdaWaitSignal(
    P2pIbgdaTransportDevice* transport,
    uint64_t expectedSignal,
    int numBlocks,
    int numThreads,
    cudaStream_t stream) {
  ibgdaWaitSignalKernel<<<numBlocks, numThreads, 0, stream>>>(
      transport, expectedSignal);
}

void launchIbgdaSignalOnly(
    P2pIbgdaTransportDevice* transport,
    uint64_t signalVal,
    int numBlocks,
    int numThreads,
    cudaStream_t stream) {
  ibgdaSignalOnlyKernel<<<numBlocks, numThreads, 0, stream>>>(
      transport, signalVal);
}

void launchIbgdaResetSignal(
    P2pIbgdaTransportDevice* transport,
    cudaStream_t stream) {
  ibgdaResetSignalKernel<<<1, 1, 0, stream>>>(transport);
}

} // namespace comms::pipes::benchmark
