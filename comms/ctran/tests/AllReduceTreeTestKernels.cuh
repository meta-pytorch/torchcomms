// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Initialize data buffer: each element = 1.0f / (1.0f + (rep + rank + i) % 256)
// Produces different values per element AND per rank, avoiding uniform fill.
void launchInitDataKernel(
    float* buf,
    size_t count,
    int rank,
    int rep,
    cudaStream_t stream);

// Initialize expected buffer: expected[i] = sum over all ranks of
// initData(rank, i) This is exactly what AllReduce(sum) should produce.
void launchInitExpectedKernel(
    float* buf,
    size_t count,
    int nranks,
    int rep,
    cudaStream_t stream);

// Compute max |actual[i] - expected[i]| across all elements.
// Returns the result synchronously (blocks on stream).
double computeMaxDelta(
    const float* actual,
    const float* expected,
    size_t count,
    cudaStream_t stream);

// Compute a deterministic checksum of the raw FP32 output bits using a single
// CUDA block. Returns synchronously after copying one uint64_t to host.
uint64_t
computeRawBitsChecksum(const float* data, size_t count, cudaStream_t stream);
