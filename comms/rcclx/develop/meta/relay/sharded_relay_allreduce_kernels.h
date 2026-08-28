/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <cstddef>
#include <cstdint>

#include "meta/relay/sharded_relay_oneshot.h"
#include "nccl.h"

/*
 * Host-callable launchers for the sharded-relay GPU kernels.
 *
 * These templates are DEFINED in
 * `sharded_relay_allreduce_kernels.cu`, which is compiled as a
 * monolithic (non-RDC) HIP translation unit so that the host stub
 * for the `<<<...>>>` launch and the matching `__global__` kernel
 * body live in the same TU.  We forward-declare every
 * instantiation that the host TU's dispatch macros may reference,
 * so the host TU never tries to instantiate them itself.
 */

template <typename T>
void launchIncrementalAddKernel(
    void* output,
    const void* input,
    size_t count,
    cudaStream_t stream);

template <typename T>
void launchScaleKernel(
    void* data,
    size_t count,
    int divisor,
    cudaStream_t stream);

template <typename T>
void launchIncrementalAddAndScaleKernel(
    void* output,
    const void* input,
    size_t count,
    int divisor,
    cudaStream_t stream);

template <typename T>
void launchFusedReduceKernel(
    void* output,
    const void* inputA,
    const void* inputB,
    size_t count,
    int divisor,
    cudaStream_t stream);

template <typename T>
void launchMultiReduceKernel(
    void* dst,
    const void* contribs,
    int numContribs,
    size_t count,
    int divisor,
    cudaStream_t stream);

/**
 * One-shot reduce-scatter over A active ranks: transfer AND reduction in a
 * single launch, replacing an ncclGroup plus a reduce kernel.
 *
 * sendBuff holds A*rc elements. The rank whose active index is mySlot produces
 *   out[i] = (sum over sources s of sendBuff_s[mySlot*rc + i]) / divisor
 *
 * Step 1  store each peer's block into that peer's staging slot mySlot.
 * Step 2  fence, then raise every peer's flag for this block.
 * Step 3  spin until every source raised MY flag for this block.
 * Step 4  out = own block + every staged block.
 *
 * Blocks handshake pairwise, so there is no global barrier and no co-residency
 * requirement: every block writes and flags before it waits.
 *
 * In-place (out aliasing sendBuff + mySlot*rc) is supported: the reduce reads
 * and writes the same element index.
 */
template <typename T>
void launchOneShotReduceScatterKernel(
    void* out,
    const void* sendBuff,
    const rcclx::relay::OneShotPeerTable& table,
    const rcclx::relay::OneShotRanks& ranks,
    int nActive,
    int myRank,
    int mySlot,
    size_t rc,
    size_t slotBytes,
    uint32_t epoch,
    int divisor,
    cudaStream_t stream);

template <typename T>
void launchSeededMultiReduceKernel(
    void* dst,
    const void* seed,
    const void* contribs,
    int numContribs,
    size_t count,
    int divisor,
    cudaStream_t stream);

// Suppress instantiation in the host TU; the actual instantiations live in
// sharded_relay_allreduce_kernels.cu.
#define RCCLX_DECLARE_RELAY_KERNEL_INSTANTIATIONS(T)                       \
  extern template void launchIncrementalAddKernel<T>(                      \
      void* output, const void* input, size_t count, cudaStream_t stream); \
  extern template void launchScaleKernel<T>(                               \
      void* data, size_t count, int divisor, cudaStream_t stream);         \
  extern template void launchIncrementalAddAndScaleKernel<T>(              \
      void* output,                                                        \
      const void* input,                                                   \
      size_t count,                                                        \
      int divisor,                                                         \
      cudaStream_t stream);                                                \
  extern template void launchFusedReduceKernel<T>(                         \
      void* output,                                                        \
      const void* inputA,                                                  \
      const void* inputB,                                                  \
      size_t count,                                                        \
      int divisor,                                                         \
      cudaStream_t stream);                                                \
  extern template void launchMultiReduceKernel<T>(                         \
      void* dst,                                                           \
      const void* contribs,                                                \
      int numContribs,                                                     \
      size_t count,                                                        \
      int divisor,                                                         \
      cudaStream_t stream);                                                \
  extern template void launchSeededMultiReduceKernel<T>(                   \
      void* dst,                                                           \
      const void* seed,                                                    \
      const void* contribs,                                                \
      int numContribs,                                                     \
      size_t count,                                                        \
      int divisor,                                                         \
      cudaStream_t stream);                                                \
  extern template void launchOneShotReduceScatterKernel<T>(                \
      void* out,                                                           \
      const void* sendBuff,                                                \
      const rcclx::relay::OneShotPeerTable& table,                         \
      const rcclx::relay::OneShotRanks& ranks,                             \
      int nActive,                                                         \
      int myRank,                                                          \
      int mySlot,                                                          \
      size_t rc,                                                           \
      size_t slotBytes,                                                    \
      uint32_t epoch,                                                      \
      int divisor,                                                         \
      cudaStream_t stream);

RCCLX_DECLARE_RELAY_KERNEL_INSTANTIATIONS(int8_t)
RCCLX_DECLARE_RELAY_KERNEL_INSTANTIATIONS(uint8_t)
RCCLX_DECLARE_RELAY_KERNEL_INSTANTIATIONS(int32_t)
RCCLX_DECLARE_RELAY_KERNEL_INSTANTIATIONS(uint32_t)
RCCLX_DECLARE_RELAY_KERNEL_INSTANTIATIONS(int64_t)
RCCLX_DECLARE_RELAY_KERNEL_INSTANTIATIONS(uint64_t)
RCCLX_DECLARE_RELAY_KERNEL_INSTANTIATIONS(__half)
RCCLX_DECLARE_RELAY_KERNEL_INSTANTIATIONS(float)
RCCLX_DECLARE_RELAY_KERNEL_INSTANTIATIONS(double)
RCCLX_DECLARE_RELAY_KERNEL_INSTANTIATIONS(__nv_bfloat16)

#undef RCCLX_DECLARE_RELAY_KERNEL_INSTANTIATIONS
