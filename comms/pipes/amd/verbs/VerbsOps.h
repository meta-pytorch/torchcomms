/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// Modifications: (c) Meta Platforms, Inc. and affiliates.

// =============================================================================
// VerbsOps - GPU-initiated RDMA one-sided verbs operations for AMD/HIP
// =============================================================================
//
// Provides template helper functions that wrap NIC backend calls into a
// higher-level API for GPU-initiated RDMA operations (put, signal, fence).
//
// All functions are templated on NicBackend for compile-time NIC selection.
// =============================================================================

#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>

#include "verbs/VerbsDev.h" // @manual

namespace pipes_gda {

// =============================================================================
// Low-level WQE operations
// =============================================================================

template <typename NicBackend>
__device__ __forceinline__ uint64_t pipes_gda_gpu_dev_verbs_reserve_wq_slots(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_qp* qp,
    uint32_t numSlots) {
  return nic.reserveWqSlots(qp, numSlots);
}

template <typename NicBackend>
__device__ __forceinline__ pipes_gda_gpu_dev_verbs_wqe*
pipes_gda_gpu_dev_verbs_get_wqe_ptr(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_qp* qp,
    uint64_t wqeIdx) {
  return nic.getWqePtr(qp, wqeIdx);
}

template <typename NicBackend>
__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_wqe_prepare_write(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_qp* qp,
    pipes_gda_gpu_dev_verbs_wqe* wqe,
    uint64_t wqeIdx,
    uint8_t ctrlFlags,
    uint64_t remoteAddr,
    uint32_t remoteKey,
    uint64_t localAddr,
    uint32_t localKey,
    uint32_t size) {
  nic.prepareRdmaWriteWqe(
      qp,
      wqe,
      wqeIdx,
      ctrlFlags,
      remoteAddr,
      remoteKey,
      localAddr,
      localKey,
      size);
}

template <typename NicBackend>
__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_wqe_prepare_atomic(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_qp* qp,
    pipes_gda_gpu_dev_verbs_wqe* wqe,
    uint64_t wqeIdx,
    uint8_t ctrlFlags,
    uint64_t remoteAddr,
    uint32_t remoteKey,
    uint64_t localAddr,
    uint32_t localKey,
    uint32_t size,
    uint64_t addVal,
    uint64_t compareVal) {
  (void)size;
  (void)compareVal;
  nic.prepareAtomicFaWqe(
      qp,
      wqe,
      wqeIdx,
      ctrlFlags,
      remoteAddr,
      remoteKey,
      localAddr,
      localKey,
      addVal);
}

template <typename NicBackend>
__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_wqe_prepare_nop(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_qp* qp,
    pipes_gda_gpu_dev_verbs_wqe* wqe,
    uint64_t wqeIdx) {
  nic.prepareNopWqe(qp, wqe, wqeIdx);
}

template <typename NicBackend>
__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_mark_wqes_ready(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_qp* qp,
    uint64_t firstIdx,
    uint64_t lastIdx) {
  nic.markWqesReady(qp, firstIdx, lastIdx);
}

template <typename NicBackend>
__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_submit(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_qp* qp,
    uint64_t nextWqeIdx) {
  nic.ringDoorbell(qp, nextWqeIdx);
}

template <typename NicBackend>
__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_wait(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_qp* qp,
    uint64_t ticket) {
  nic.pollCqAt(qp, &qp->cq_sq, ticket);
}

// =============================================================================
// High-level composite operations
// =============================================================================

/**
 * pipes_gda_gpu_dev_verbs_put - RDMA Write (handles multi-chunk transfers)
 *
 * Reserves WQE slots, prepares RDMA WRITE WQEs (splitting into chunks
 * if size > MAX_TRANSFER_SIZE), marks ready, and submits.
 */
template <typename NicBackend>
__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_put(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_qp* qp,
    pipes_gda_gpu_dev_verbs_addr raddr,
    pipes_gda_gpu_dev_verbs_addr laddr,
    std::size_t size,
    uint64_t* out_ticket) {
  uint32_t numChunks = static_cast<uint32_t>(
      (size + PIPES_GDA_VERBS_MAX_TRANSFER_SIZE - 1) >>
      PIPES_GDA_VERBS_MAX_TRANSFER_SIZE_SHIFT);
  if (numChunks == 0)
    numChunks = 1;

  uint64_t baseIdx =
      pipes_gda_gpu_dev_verbs_reserve_wq_slots(nic, qp, numChunks);
  std::size_t remaining = size;

  for (uint32_t i = 0; i < numChunks; i++) {
    uint64_t wqeIdx = baseIdx + i;
    std::size_t chunkSize = remaining > PIPES_GDA_VERBS_MAX_TRANSFER_SIZE
        ? PIPES_GDA_VERBS_MAX_TRANSFER_SIZE
        : remaining;

    auto* wqe = pipes_gda_gpu_dev_verbs_get_wqe_ptr(nic, qp, wqeIdx);
    pipes_gda_gpu_dev_verbs_wqe_prepare_write(
        nic,
        qp,
        wqe,
        wqeIdx,
        PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE,
        raddr.addr + i * PIPES_GDA_VERBS_MAX_TRANSFER_SIZE,
        raddr.key,
        laddr.addr + i * PIPES_GDA_VERBS_MAX_TRANSFER_SIZE,
        laddr.key,
        static_cast<uint32_t>(chunkSize));
    remaining -= chunkSize;
  }

  uint64_t lastIdx = baseIdx + numChunks - 1;
  pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, qp, baseIdx, lastIdx);
  pipes_gda_gpu_dev_verbs_submit(nic, qp, lastIdx + 1);

  *out_ticket = lastIdx;
}

/**
 * pipes_gda_gpu_dev_verbs_signal - Atomic fetch-add signal
 *
 * Posts an atomic fetch-add WQE to the remote signal buffer.
 */
template <typename NicBackend>
__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_signal(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_qp* qp,
    pipes_gda_gpu_dev_verbs_addr sig_raddr,
    pipes_gda_gpu_dev_verbs_addr sig_laddr,
    uint64_t sig_val,
    uint64_t* out_ticket) {
  uint64_t wqeIdx = pipes_gda_gpu_dev_verbs_reserve_wq_slots(nic, qp, 1);
  auto* wqe = pipes_gda_gpu_dev_verbs_get_wqe_ptr(nic, qp, wqeIdx);

  pipes_gda_gpu_dev_verbs_wqe_prepare_atomic(
      nic,
      qp,
      wqe,
      wqeIdx,
      PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE,
      sig_raddr.addr,
      sig_raddr.key,
      sig_laddr.addr,
      sig_laddr.key,
      sizeof(uint64_t),
      sig_val,
      0);

  pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, qp, wqeIdx, wqeIdx);
  pipes_gda_gpu_dev_verbs_submit(nic, qp, wqeIdx + 1);

  *out_ticket = wqeIdx;
}

/**
 * pipes_gda_gpu_dev_verbs_put_signal - RDMA Write + atomic signal
 * (non-adaptive)
 *
 * Posts data WQEs followed by an atomic signal WQE without NIC fence.
 */
template <typename NicBackend>
__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_put_signal(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_qp* qp,
    pipes_gda_gpu_dev_verbs_addr raddr,
    pipes_gda_gpu_dev_verbs_addr laddr,
    std::size_t size,
    pipes_gda_gpu_dev_verbs_addr sig_raddr,
    pipes_gda_gpu_dev_verbs_addr sig_laddr,
    uint64_t sig_val,
    uint64_t* out_ticket) {
  uint32_t numChunks = static_cast<uint32_t>(
      (size + PIPES_GDA_VERBS_MAX_TRANSFER_SIZE - 1) >>
      PIPES_GDA_VERBS_MAX_TRANSFER_SIZE_SHIFT);
  if (numChunks == 0)
    numChunks = 1;

  uint64_t baseIdx =
      pipes_gda_gpu_dev_verbs_reserve_wq_slots(nic, qp, numChunks + 1);
  std::size_t remaining = size;

  for (uint32_t i = 0; i < numChunks; i++) {
    uint64_t wqeIdx = baseIdx + i;
    std::size_t chunkSize = remaining > PIPES_GDA_VERBS_MAX_TRANSFER_SIZE
        ? PIPES_GDA_VERBS_MAX_TRANSFER_SIZE
        : remaining;

    auto* wqe = pipes_gda_gpu_dev_verbs_get_wqe_ptr(nic, qp, wqeIdx);
    pipes_gda_gpu_dev_verbs_wqe_prepare_write(
        nic,
        qp,
        wqe,
        wqeIdx,
        PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE,
        raddr.addr + i * PIPES_GDA_VERBS_MAX_TRANSFER_SIZE,
        raddr.key,
        laddr.addr + i * PIPES_GDA_VERBS_MAX_TRANSFER_SIZE,
        laddr.key,
        static_cast<uint32_t>(chunkSize));
    remaining -= chunkSize;
  }

  uint64_t sigIdx = baseIdx + numChunks;
  auto* sigWqe = pipes_gda_gpu_dev_verbs_get_wqe_ptr(nic, qp, sigIdx);
  pipes_gda_gpu_dev_verbs_wqe_prepare_atomic(
      nic,
      qp,
      sigWqe,
      sigIdx,
      PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE,
      sig_raddr.addr,
      sig_raddr.key,
      sig_laddr.addr,
      sig_laddr.key,
      sizeof(uint64_t),
      sig_val,
      0);

  pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, qp, baseIdx, sigIdx);
  pipes_gda_gpu_dev_verbs_submit(nic, qp, sigIdx + 1);

  *out_ticket = sigIdx;
}

// =============================================================================
// Utility functions
// =============================================================================

/**
 * pipes_gda_fence - Wait for all pending RDMA operations to complete
 *
 * Issues a NOP WQE and waits for it to complete. Since WQEs are processed
 * in order, when the NOP completes, all prior WQEs have been processed.
 */
template <typename NicBackend>
__device__ __forceinline__ void pipes_gda_fence(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_qp* qp) {
  uint64_t wqeIdx = pipes_gda_gpu_dev_verbs_reserve_wq_slots(nic, qp, 1);
  auto* wqe = pipes_gda_gpu_dev_verbs_get_wqe_ptr(nic, qp, wqeIdx);

  pipes_gda_gpu_dev_verbs_wqe_prepare_nop(nic, qp, wqe, wqeIdx);
  pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, qp, wqeIdx, wqeIdx);
  pipes_gda_gpu_dev_verbs_submit(nic, qp, wqeIdx + 1);
  pipes_gda_gpu_dev_verbs_wait(nic, qp, wqeIdx);
}

/**
 * pipes_gda_put_fenced - Fenced RDMA Write with completion
 *
 * Issues a fence, then performs an RDMA Write and waits for completion,
 * then issues another fence.
 */
template <typename NicBackend>
__device__ __forceinline__ void pipes_gda_put_fenced(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_qp* qp,
    pipes_gda_gpu_dev_verbs_addr raddr,
    pipes_gda_gpu_dev_verbs_addr laddr,
    std::size_t size) {
  pipes_gda_fence(nic, qp);

  uint64_t ticket;
  pipes_gda_gpu_dev_verbs_put(nic, qp, raddr, laddr, size, &ticket);
  pipes_gda_gpu_dev_verbs_wait(nic, qp, ticket);

  pipes_gda_fence(nic, qp);
}

// =============================================================================
// Additional primitives
// =============================================================================

/**
 * pipes_gda_gpu_dev_verbs_p<T> - Inline RDMA write of a scalar value
 *
 * Writes a scalar value to a remote address using an inline RDMA Write WQE.
 * No local memory region needed — data is embedded in the WQE.
 * Used by reset_signal() to write zero to remote signal buffer.
 */
template <typename T, typename NicBackend>
__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_p(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_qp* qp,
    pipes_gda_gpu_dev_verbs_addr raddr,
    T value,
    uint64_t* out_ticket) {
  uint64_t wqeIdx = pipes_gda_gpu_dev_verbs_reserve_wq_slots(nic, qp, 1);
  auto* wqe = pipes_gda_gpu_dev_verbs_get_wqe_ptr(nic, qp, wqeIdx);

  nic.prepareInlineWriteWqe(
      qp,
      wqe,
      wqeIdx,
      PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE,
      raddr.addr,
      raddr.key,
      value);

  pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, qp, wqeIdx, wqeIdx);
  pipes_gda_gpu_dev_verbs_submit(nic, qp, wqeIdx + 1);

  *out_ticket = wqeIdx;
}

/**
 * pipes_gda_gpu_dev_verbs_poll_one_cq_at - Non-blocking CQ poll wrapper
 *
 * Returns EBUSY if not yet complete, 0 on success.
 * Used by wait_local() with timeout.
 */
template <typename NicBackend>
__device__ __forceinline__ int pipes_gda_gpu_dev_verbs_poll_one_cq_at(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_cq* cq,
    uint64_t consIndex) {
  return nic.pollOneCqAt(cq, consIndex);
}

/**
 * pipes_gda_gpu_dev_verbs_qp_get_cq_sq - Get pointer to QP's SQ CQ
 */
__device__ __forceinline__ pipes_gda_gpu_dev_verbs_cq*
pipes_gda_gpu_dev_verbs_qp_get_cq_sq(pipes_gda_gpu_dev_verbs_qp* qp) {
  return &qp->cq_sq;
}

/**
 * pipes_gda_gpu_dev_verbs_put_signal_counter - Data write + remote signal +
 * local counter via companion QP
 *
 * Compound operation:
 * 1. Main QP: RDMA Write data
 * 2. Main QP: Fenced atomic fetch-add to remote signal buffer
 * 3. Companion QP: WAIT on main QP signal completion
 * 4. Companion QP: Atomic fetch-add to local counter buffer
 */
template <typename NicBackend>
__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_put_signal_counter(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_qp* mainQp,
    pipes_gda_gpu_dev_verbs_addr raddr,
    pipes_gda_gpu_dev_verbs_addr laddr,
    std::size_t size,
    pipes_gda_gpu_dev_verbs_addr sigRemoteAddr,
    pipes_gda_gpu_dev_verbs_addr sigSinkAddr,
    uint64_t sigVal,
    pipes_gda_gpu_dev_verbs_qp* companionQp,
    pipes_gda_gpu_dev_verbs_addr counterRemoteAddr,
    pipes_gda_gpu_dev_verbs_addr counterSinkAddr,
    uint64_t counterVal) {
  uint32_t numChunks = static_cast<uint32_t>(
      (size + PIPES_GDA_VERBS_MAX_TRANSFER_SIZE - 1) >>
      PIPES_GDA_VERBS_MAX_TRANSFER_SIZE_SHIFT);
  if (numChunks == 0)
    numChunks = 1;

  uint64_t mainBase =
      pipes_gda_gpu_dev_verbs_reserve_wq_slots(nic, mainQp, numChunks + 1);
  std::size_t remaining = size;

  for (uint32_t i = 0; i < numChunks; i++) {
    uint64_t wqeIdx = mainBase + i;
    std::size_t chunkSize = remaining > PIPES_GDA_VERBS_MAX_TRANSFER_SIZE
        ? PIPES_GDA_VERBS_MAX_TRANSFER_SIZE
        : remaining;

    auto* wqe = pipes_gda_gpu_dev_verbs_get_wqe_ptr(nic, mainQp, wqeIdx);
    pipes_gda_gpu_dev_verbs_wqe_prepare_write(
        nic,
        mainQp,
        wqe,
        wqeIdx,
        PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE,
        raddr.addr + i * PIPES_GDA_VERBS_MAX_TRANSFER_SIZE,
        raddr.key,
        laddr.addr + i * PIPES_GDA_VERBS_MAX_TRANSFER_SIZE,
        laddr.key,
        static_cast<uint32_t>(chunkSize));
    remaining -= chunkSize;
  }

  uint64_t sigIdx = mainBase + numChunks;
  auto* sigWqe = pipes_gda_gpu_dev_verbs_get_wqe_ptr(nic, mainQp, sigIdx);
  pipes_gda_gpu_dev_verbs_wqe_prepare_atomic(
      nic,
      mainQp,
      sigWqe,
      sigIdx,
      static_cast<uint8_t>(
          PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE |
          PIPES_GDA_IB_MLX5_WQE_CTRL_FENCE),
      sigRemoteAddr.addr,
      sigRemoteAddr.key,
      sigSinkAddr.addr,
      sigSinkAddr.key,
      sizeof(uint64_t),
      sigVal,
      0);

  pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, mainQp, mainBase, sigIdx);
  pipes_gda_gpu_dev_verbs_submit(nic, mainQp, sigIdx + 1);

  // Companion QP: WAIT + counter atomic
  uint64_t compBase =
      pipes_gda_gpu_dev_verbs_reserve_wq_slots(nic, companionQp, 2);

  uint64_t waitIdx = compBase;
  auto* waitWqe =
      pipes_gda_gpu_dev_verbs_get_wqe_ptr(nic, companionQp, waitIdx);
  nic.prepareWaitWqe(
      companionQp,
      waitWqe,
      waitIdx,
      PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE,
      mainQp->cq_sq.cq_num,
      sigIdx);

  uint64_t cntIdx = compBase + 1;
  auto* cntWqe = pipes_gda_gpu_dev_verbs_get_wqe_ptr(nic, companionQp, cntIdx);
  pipes_gda_gpu_dev_verbs_wqe_prepare_atomic(
      nic,
      companionQp,
      cntWqe,
      cntIdx,
      PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE,
      counterRemoteAddr.addr,
      counterRemoteAddr.key,
      counterSinkAddr.addr,
      counterSinkAddr.key,
      sizeof(uint64_t),
      counterVal,
      0);

  pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, companionQp, compBase, cntIdx);
  pipes_gda_gpu_dev_verbs_submit(nic, companionQp, cntIdx + 1);
}

/**
 * pipes_gda_gpu_dev_verbs_signal_counter - Remote signal + local counter
 * (no data write)
 *
 * Same as put_signal_counter but without the data write.
 */
template <typename NicBackend>
__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_signal_counter(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_qp* mainQp,
    pipes_gda_gpu_dev_verbs_addr sigRemoteAddr,
    pipes_gda_gpu_dev_verbs_addr sigSinkAddr,
    uint64_t sigVal,
    pipes_gda_gpu_dev_verbs_qp* companionQp,
    pipes_gda_gpu_dev_verbs_addr counterRemoteAddr,
    pipes_gda_gpu_dev_verbs_addr counterSinkAddr,
    uint64_t counterVal) {
  // Main QP: signal atomic
  uint64_t sigIdx = pipes_gda_gpu_dev_verbs_reserve_wq_slots(nic, mainQp, 1);
  auto* sigWqe = pipes_gda_gpu_dev_verbs_get_wqe_ptr(nic, mainQp, sigIdx);
  pipes_gda_gpu_dev_verbs_wqe_prepare_atomic(
      nic,
      mainQp,
      sigWqe,
      sigIdx,
      PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE,
      sigRemoteAddr.addr,
      sigRemoteAddr.key,
      sigSinkAddr.addr,
      sigSinkAddr.key,
      sizeof(uint64_t),
      sigVal,
      0);

  pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, mainQp, sigIdx, sigIdx);
  pipes_gda_gpu_dev_verbs_submit(nic, mainQp, sigIdx + 1);

  // Companion QP: WAIT + counter atomic
  uint64_t compBase =
      pipes_gda_gpu_dev_verbs_reserve_wq_slots(nic, companionQp, 2);

  uint64_t waitIdx = compBase;
  auto* waitWqe =
      pipes_gda_gpu_dev_verbs_get_wqe_ptr(nic, companionQp, waitIdx);
  nic.prepareWaitWqe(
      companionQp,
      waitWqe,
      waitIdx,
      PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE,
      mainQp->cq_sq.cq_num,
      sigIdx);

  uint64_t cntIdx = compBase + 1;
  auto* cntWqe = pipes_gda_gpu_dev_verbs_get_wqe_ptr(nic, companionQp, cntIdx);
  pipes_gda_gpu_dev_verbs_wqe_prepare_atomic(
      nic,
      companionQp,
      cntWqe,
      cntIdx,
      PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE,
      counterRemoteAddr.addr,
      counterRemoteAddr.key,
      counterSinkAddr.addr,
      counterSinkAddr.key,
      sizeof(uint64_t),
      counterVal,
      0);

  pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, companionQp, compBase, cntIdx);
  pipes_gda_gpu_dev_verbs_submit(nic, companionQp, cntIdx + 1);
}

} // namespace pipes_gda
