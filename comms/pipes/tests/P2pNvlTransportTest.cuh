// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace comms::pipes::test {

// Kernel to fill buffer with a specific value
// Each thread writes one integer
__global__ void fillBufferKernel(int* buffer, int value, size_t numElements);

// Kernel to verify buffer contents
// Each thread checks one integer and atomically increments error counter if
// mismatch
__global__ void verifyBufferKernel(
    const int* buffer,
    int expectedValue,
    size_t numElements,
    int* errorCount);

// Host wrapper functions
void fillBuffer(int* deviceBuffer, int value, size_t numElements);
void verifyBuffer(
    const int* deviceBuffer,
    int expectedValue,
    size_t numElements,
    int* deviceErrorCount);

} // namespace comms::pipes::test
