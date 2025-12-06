// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/pipes/tests/P2pNvlTransportTest.cuh"

namespace comms::pipes::test {

__global__ void fillBufferKernel(int* buffer, int value, size_t numElements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    buffer[idx] = value;
  }
}

__global__ void verifyBufferKernel(
    const int* buffer,
    int expectedValue,
    size_t numElements,
    int* errorCount) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    if (buffer[idx] != expectedValue) {
      atomicAdd(errorCount, 1);
    }
  }
}

void fillBuffer(int* deviceBuffer, int value, size_t numElements) {
  const int blockSize = 256;
  const int numBlocks = (numElements + blockSize - 1) / blockSize;
  fillBufferKernel<<<numBlocks, blockSize>>>(deviceBuffer, value, numElements);
}

void verifyBuffer(
    const int* deviceBuffer,
    int expectedValue,
    size_t numElements,
    int* deviceErrorCount) {
  const int blockSize = 256;
  const int numBlocks = (numElements + blockSize - 1) / blockSize;
  verifyBufferKernel<<<numBlocks, blockSize>>>(
      deviceBuffer, expectedValue, numElements, deviceErrorCount);
}

} // namespace comms::pipes::test
