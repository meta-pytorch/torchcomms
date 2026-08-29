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
 * One-shot 2-rank reduce-scatter: transfer AND reduction in a single launch.
 *
 * sendBuff holds 2*rc elements. Rank m must produce
 *   out[i] = (sendBuff[m*rc + i] + peerSendBuff[m*rc + i]) / divisor
 *
 * Step 1  store my foreign block into the PEER's staging slot mySlot.
 * Step 2  fence, then raise the peer's flag for this block.
 * Step 3  spin until the peer raised MY flag for this block.
 * Step 4  out = own block + what the peer staged.
 *
 * Blocks handshake pairwise, so no global barrier and no co-residency
 * requirement: every block writes and flags before it waits.
 *
 * `inPlace` means out aliases sendBuff + mySlot*rc; the reduce reads that as
 * its own contribution and writes it back, which is safe because it is the same
 * element index.
 */
template <typename T>
void launchOneShotReduceScatter2Kernel(
    void* out,
    const void* sendBuff,
    const rcclx::relay::OneShotPeerTable& table,
    int myRank,
    int peerRank,
    int mySlot,
    int peerSlot,
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
  extern template void launchOneShotReduceScatter2Kernel<T>(               \
      void* out,                                                           \
      const void* sendBuff,                                                \
      const rcclx::relay::OneShotPeerTable& table,                         \
      int myRank,                                                          \
      int peerRank,                                                        \
      int mySlot,                                                          \
      int peerSlot,                                                        \
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
