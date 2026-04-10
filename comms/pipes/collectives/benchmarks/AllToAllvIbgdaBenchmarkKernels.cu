// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/collectives/benchmarks/AllToAllvIbgdaBenchmarkKernels.cuh"

#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"

namespace comms::pipes::benchmark {

__global__ void ibgdaSendBenchmarkKernel(
    P2pIbgdaTransportDevice* transport,
    void* srcPtr,
    std::size_t nbytes,
    int numIters) {
  // Each block handles a disjoint data slice using channelId = blockIdx.x.
  // When gridDim.x == 1 (default), behavior is identical to the original.
  uint32_t channelId = blockIdx.x;
  uint32_t numBlocks = gridDim.x;

  size_t blockBytes = nbytes / numBlocks;
  size_t blockOffset = channelId * blockBytes;
  size_t myBytes =
      (channelId == numBlocks - 1) ? (nbytes - blockOffset) : blockBytes;
  char* myPtr = static_cast<char*>(srcPtr) + blockOffset;

  // Single-group ThreadGroup: this block handles exactly one channel.
  ThreadGroup group = {
      .thread_id_in_group = threadIdx.x,
      .group_size = blockDim.x,
      .group_id = 0,
      .total_groups = 1,
      .scope = SyncScope::MULTIWARP,
      .barrier_id = 0};

  for (int i = 0; i < numIters; i++) {
    transport->send(group, myPtr, myBytes, Timeout{}, channelId);
  }
}

__global__ void ibgdaRecvBenchmarkKernel(
    P2pIbgdaTransportDevice* transport,
    void* dstPtr,
    std::size_t nbytes,
    int numIters) {
  uint32_t channelId = blockIdx.x;
  uint32_t numBlocks = gridDim.x;

  size_t blockBytes = nbytes / numBlocks;
  size_t blockOffset = channelId * blockBytes;
  size_t myBytes =
      (channelId == numBlocks - 1) ? (nbytes - blockOffset) : blockBytes;
  char* myPtr = static_cast<char*>(dstPtr) + blockOffset;

  ThreadGroup group = {
      .thread_id_in_group = threadIdx.x,
      .group_size = blockDim.x,
      .group_id = 0,
      .total_groups = 1,
      .scope = SyncScope::MULTIWARP,
      .barrier_id = 0};

  for (int i = 0; i < numIters; i++) {
    transport->recv(group, myPtr, myBytes, Timeout{}, channelId);
  }
}

void launchIbgdaSendBench(
    P2pIbgdaTransportDevice* transport,
    void* sendBuf,
    std::size_t nbytes,
    int numIters,
    cudaStream_t stream,
    int numBlocks,
    int numThreads) {
  ibgdaSendBenchmarkKernel<<<numBlocks, numThreads, 0, stream>>>(
      transport, sendBuf, nbytes, numIters);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Send kernel launch failed: ") + cudaGetErrorString(err));
  }
}

void launchIbgdaRecvBench(
    P2pIbgdaTransportDevice* transport,
    void* recvBuf,
    std::size_t nbytes,
    int numIters,
    cudaStream_t stream,
    int numBlocks,
    int numThreads) {
  ibgdaRecvBenchmarkKernel<<<numBlocks, numThreads, 0, stream>>>(
      transport, recvBuf, nbytes, numIters);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("Recv kernel launch failed: ") + cudaGetErrorString(err));
  }
}

} // namespace comms::pipes::benchmark
