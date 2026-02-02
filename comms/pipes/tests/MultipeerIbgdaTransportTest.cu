// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/MultipeerIbgdaTransportTest.cuh"

#include <cuda_runtime.h>

namespace comms::pipes::test {

// =============================================================================
// Kernel: Put with signal
// =============================================================================

__global__ void putSignalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    uint64_t signalVal) {
  // Only thread 0 performs the put_signal
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto work = transport->put_signal(localBuf, remoteBuf, nbytes, signalVal);
    transport->wait_local(work);
  }
}

void testPutSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    uint64_t signalVal,
    int numBlocks,
    int blockSize) {
  putSignalKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr, localBuf, remoteBuf, nbytes, signalVal);
}

// =============================================================================
// Kernel: Wait for signal
// =============================================================================

__global__ void waitSignalKernel(
    P2pIbgdaTransportDevice* transport,
    uint64_t expectedSignal) {
  // Only thread 0 waits for the signal
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    transport->wait_signal(expectedSignal);
  }
}

void testWaitSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    uint64_t expectedSignal,
    int numBlocks,
    int blockSize) {
  waitSignalKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr, expectedSignal);
}

// =============================================================================
// Kernel: Multiple put_signal operations
// =============================================================================

__global__ void multiplePutSignalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t bytesPerPut,
    int numPuts) {
  // Only thread 0 performs the puts
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int i = 0; i < numPuts; i++) {
      IbgdaLocalBuffer srcBuf = localBuf.subBuffer(i * bytesPerPut);
      IbgdaRemoteBuffer dstBuf = remoteBuf.subBuffer(i * bytesPerPut);

      // Signal value is i+1 (cumulative count)
      auto work = transport->put_signal(srcBuf, dstBuf, bytesPerPut, 1);
      transport->wait_local(work);
    }
  }
}

void testMultiplePutSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t bytesPerPut,
    int numPuts,
    int numBlocks,
    int blockSize) {
  multiplePutSignalKernel<<<numBlocks, blockSize>>>(
      deviceTransportPtr, localBuf, remoteBuf, bytesPerPut, numPuts);
}

// =============================================================================
// Kernel: Fill buffer with pattern
// =============================================================================

__global__ void
fillPatternKernel(uint8_t* buffer, std::size_t nbytes, uint8_t baseValue) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t i = idx; i < nbytes; i += stride) {
    buffer[i] = static_cast<uint8_t>(baseValue + (i % 256));
  }
}

void fillBufferWithPattern(
    void* buffer,
    std::size_t nbytes,
    uint8_t baseValue,
    int numBlocks,
    int blockSize) {
  fillPatternKernel<<<numBlocks, blockSize>>>(
      static_cast<uint8_t*>(buffer), nbytes, baseValue);
}

// =============================================================================
// Kernel: Verify buffer pattern
// =============================================================================

__global__ void verifyPatternKernel(
    const uint8_t* buffer,
    std::size_t nbytes,
    uint8_t expectedBaseValue,
    int* errorCount) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t i = idx; i < nbytes; i += stride) {
    uint8_t expected = static_cast<uint8_t>(expectedBaseValue + (i % 256));
    if (buffer[i] != expected) {
      atomicAdd(errorCount, 1);
    }
  }
}

void verifyBufferPattern(
    const void* buffer,
    std::size_t nbytes,
    uint8_t expectedBaseValue,
    int* errorCount,
    int numBlocks,
    int blockSize) {
  verifyPatternKernel<<<numBlocks, blockSize>>>(
      static_cast<const uint8_t*>(buffer),
      nbytes,
      expectedBaseValue,
      errorCount);
}

} // namespace comms::pipes::test
