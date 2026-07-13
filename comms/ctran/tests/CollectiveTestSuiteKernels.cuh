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
// initScatterExpected (ReduceScatter): expected[i] = sum over all ranks of
// initData(rank, rep, baseIdx + i). Each rank's expected recv buffer is the
// reduction shard starting at baseIdx = myRank * recvcount.
// -----------------------------------------------------------------------------
void launchInitScatterExpectedKernel(
    float* buf,
    size_t recvcount,
    int nranks,
    int rep,
    size_t baseIdx,
    cudaStream_t stream);
void launchInitScatterExpectedKernel(
    __half* buf,
    size_t recvcount,
    int nranks,
    int rep,
    size_t baseIdx,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// initGatherExpected (AllGather): expected[r*sendcount + i] = initData(r, rep,
// i) for every r in [0, nranks). Pure host composition of launchInitDataKernel
// over each rank's shard.
// -----------------------------------------------------------------------------
template <typename T>
inline void launchInitGatherExpectedTyped(
    T* dst,
    size_t sendcount,
    int nranks,
    int rep,
    cudaStream_t stream) {
  for (int r = 0; r < nranks; ++r) {
    launchInitDataKernel(
        dst + static_cast<size_t>(r) * sendcount, sendcount, r, rep, stream);
  }
}

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
