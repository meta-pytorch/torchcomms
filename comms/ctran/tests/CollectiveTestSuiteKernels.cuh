// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "comms/utils/commSpecs.h"

// Shared algo-agnostic per-element kernels for the collective TestSuite family:
// deterministic initData, AR(sum) initExpected, float/int max-delta, and a
// raw-bits checksum. Collective-specific expected builders (e.g. AG gather / RS
// scatter shards) live with their collective's traits, not here.

// -----------------------------------------------------------------------------
// initData: each element = deterministic per-(rank, rep, idx) value.
// -----------------------------------------------------------------------------
void launchInitDataKernel(
    float* buf,
    size_t count,
    int rank,
    int rep,
    cudaStream_t stream);
void launchInitDataKernel(
    __half* buf,
    size_t count,
    int rank,
    int rep,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// initExpected: expected[i] = sum over all ranks of initData(rank, rep, i).
// This is the AR(sum) expected buffer.
// -----------------------------------------------------------------------------
void launchInitExpectedKernel(
    float* buf,
    size_t count,
    int nranks,
    int rep,
    cudaStream_t stream);
void launchInitExpectedKernel(
    __half* buf,
    size_t count,
    int nranks,
    int rep,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// delta: max |actual[i] - expected[i]|. Synchronous.
// -----------------------------------------------------------------------------
double computeMaxDelta(
    const float* actual,
    const float* expected,
    size_t count,
    cudaStream_t stream);
double computeMaxDelta(
    const __half* actual,
    const __half* expected,
    size_t count,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// raw-bits checksum: deterministic per-buffer digest for determinism checks.
// -----------------------------------------------------------------------------
uint64_t
computeRawBitsChecksum(const float* data, size_t count, cudaStream_t stream);
uint64_t
computeRawBitsChecksum(const __half* data, size_t count, cudaStream_t stream);
