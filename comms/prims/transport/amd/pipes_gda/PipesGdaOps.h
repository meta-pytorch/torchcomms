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
// Provides helper functions that wrap NIC backend calls into a higher-level
// API for GPU-initiated RDMA operations (put, signal, fence).
//
// =============================================================================
// Two coexisting overload families
// =============================================================================
//
// Each `pipes_gda_*` primitive is provided in TWO overloads with the same
// name, distinguished by parameter list. They achieve identical
// functionality; pick whichever is more convenient at the call site.
//
// 1. **NicBackend-explicit form** (legacy, used by AMD-only callers in
//    `comms/prims/transport/amd/{collectives,benchmarks,tests}/`)
//    - First parameter is `NicBackend& nic`.
//    - Templated on `<typename NicBackend>` for compile-time NIC selection.
//    - Existing AMD-only call sites use this form.
//
// 2. **DOCA-aligned form** (used by `comms/prims/DocaCompat.h`)
//    - **Signature is identical to the same-suffix `doca_*` API in NVIDIA's
//      `<device/doca_gpunetio_dev_verbs_*.cuh>`** — same parameter list,
//      same parameter order, same template parameters
//      (`<MODE>`/`<SCOPE>`/`<HANDLER>`), same return type. DOCA-only
//      params that AMD doesn't need (`opcode`, `lkey_id`) are accepted
//      and `(void)`-cast inside the body.
//    - No `nic` parameter; the function creates a stack-local
//      `ActiveNicBackend nic{}` (stateless empty struct → zero-cost) and
//      forwards to the NicBackend-explicit overload.
//    - Lets `comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh` use one
//    set of call
//      shapes that resolve to DOCA on NVIDIA and to these wrappers on AMD,
//      via a thin name-prefix shim in `DocaCompat.h`.
//
// Functional-equivalence map (DOCA suffix ↔ pipes_gda suffix):
//   reserve_wq_slots / get_wqe_ptr / wqe_prepare_write /
//   wqe_prepare_atomic / mark_wqes_ready / submit / wait / put /
//   signal_counter / poll_one_cq_at / qp_get_cq_sq
//
// And `pipes_gda_fence` (no `gpu_dev_verbs_` infix) is the AMD
// equivalent of `doca_fence` (defined in
// `comms/prims/platform/DocaVerbsUtils.cuh`).
// =============================================================================

#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>

#include "nic/NicSelector.h" // @manual
#include "pipes_gda/PipesGdaDev.h" // @manual

namespace pipes_gda {

#if defined(__HIP_PLATFORM_AMD__)
namespace {
// File-local helper: spin on the non-blocking `pollOneCqAt` until the
// CQE arrives, then trap on NIC error. Intentionally does NOT advance
// `cq_sq.cqe_ci` or ring the CQ doorbell — callers manage `cqe_ci`
// locally because ringing the CQ doorbell on the BNXT counter-only
// paths flushes the QP into LOC_QP_OP error state (see callers).
//
// Used by both the BNXT chunked-put paths (`#ifdef NIC_BNXT`) and the
// AMD counter/signal fast paths (`#if defined(__HIP_PLATFORM_AMD__)`),
// so it must be visible for every AMD build (bnxt AND mlx5), not just
// NIC_BNXT. NVIDIA never compiles this file (it uses the real DOCA
// `doca_gpu_dev_verbs_*` headers), so this is AMD-only.
//
// Not exposed in the `pipes_gda_*` API surface: this is a CQE-drain
// idiom with no DOCA equivalent, kept internal to preserve the 1:1
// mirror with NVIDIA's `doca_gpu_dev_verbs_*` names.
template <typename NicBackend>
__device__ __forceinline__ void spin_poll_or_trap(
    NicBackend& nic,
    pipes_gda_gpu_dev_verbs_cq* cq,
    uint64_t consIndex) {
  int rc;
  while ((rc = nic.pollOneCqAt(cq, consIndex)) == EBUSY) {
  }
  // `pollOneCqAt` already prints the BNXT CQE status on error; trap to
  // surface the failure to the host instead of silently advancing
  // `cqe_ci` past a failed WQE (which would let the GPU treat a
  // NIC-error completion as success and produce misleading numbers).
  // `__trap()` is a device-only intrinsic; HIP parses `__device__`
  // function bodies during the host pass too, so gate it on the
  // device-compile guard.
  if (rc != 0) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __trap();
#endif
  }
}
} // namespace
#endif

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

#ifdef NIC_BNXT
    // BNXT: per-WQE doorbell + per-chunk CQE drain. The mlx5 batched
    // doorbell pattern desyncs the NIC, and overlapping chunks trigger
    // LOC_QP_OP. The caller drains the last chunk's CQE.
    pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, qp, wqeIdx, wqeIdx);
    pipes_gda_gpu_dev_verbs_submit(nic, qp, wqeIdx + 1);
    if (i + 1 < numChunks) {
      spin_poll_or_trap(nic, &qp->cq_sq, wqeIdx);
      qp->cq_sq.cqe_ci = wqeIdx + 1;
    }
#endif
  }

  uint64_t lastIdx = baseIdx + numChunks - 1;
#ifdef NIC_MLX5
  // Mlx5: batched doorbell — single ring after all chunks prepared.
  // Uses fast submit with ctrl segment captured on GPU stack (avoids
  // PCIe round-trip re-read from SQ buffer).
  pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, qp, baseIdx, lastIdx);
  pipes_gda_gpu_dev_verbs_submit(nic, qp, lastIdx + 1);
#endif

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

#ifdef NIC_BNXT
    // BNXT: per-WQE doorbell + per-chunk CQE drain.
    pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, qp, wqeIdx, wqeIdx);
    pipes_gda_gpu_dev_verbs_submit(nic, qp, wqeIdx + 1);
    spin_poll_or_trap(nic, &qp->cq_sq, wqeIdx);
    qp->cq_sq.cqe_ci = wqeIdx + 1;
#endif
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

#ifdef NIC_BNXT
  // BNXT: per-WQE doorbell for the signal WQE too.
  pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, qp, sigIdx, sigIdx);
  pipes_gda_gpu_dev_verbs_submit(nic, qp, sigIdx + 1);
#else
  pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, qp, baseIdx, sigIdx);
  pipes_gda_gpu_dev_verbs_submit(nic, qp, sigIdx + 1);
#endif

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

  // PutWaitLocal benchmark passes IbgdaRemoteBuffer{} for sig (addr==0)
  // — that means "no signal, just data write + counter". Skipping the
  // atomic-FA WQE on this path drops per-iter latency from ~700us
  // (BNXT atomic-FA RTT) to ~10-30us (RDMA-WRITE RTT).
  bool hasSignal = (sigRemoteAddr.addr != 0);
  uint32_t numWqes = numChunks + (hasSignal ? 1 : 0);

  uint64_t mainBase =
      pipes_gda_gpu_dev_verbs_reserve_wq_slots(nic, mainQp, numWqes);
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

#ifdef NIC_BNXT
    // BNXT: per-WQE doorbell + per-chunk CQE drain.
    pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, mainQp, wqeIdx, wqeIdx);
    pipes_gda_gpu_dev_verbs_submit(nic, mainQp, wqeIdx + 1);
    spin_poll_or_trap(nic, &mainQp->cq_sq, wqeIdx);
    mainQp->cq_sq.cqe_ci = wqeIdx + 1;
#endif
  }

  uint64_t lastIdx;
  if (hasSignal) {
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
    lastIdx = sigIdx;
#ifdef NIC_BNXT
    pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, mainQp, sigIdx, sigIdx);
    pipes_gda_gpu_dev_verbs_submit(nic, mainQp, sigIdx + 1);
#endif
  } else {
    lastIdx = mainBase + numChunks - 1;
  }

#ifdef NIC_MLX5
  pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, mainQp, mainBase, lastIdx);
  pipes_gda_gpu_dev_verbs_submit(nic, mainQp, lastIdx + 1);
#endif

#if defined(__HIP_PLATFORM_AMD__)
  // AMD: spin-poll main QP CQ + GPU atomic instead of companion-QP/WAIT.
  // BNXT: no inter-QP wait WQE primitive (must use this path).
  // mlx5+AMD: WAIT WQE / companion-QP loopback observed not to advance
  // cross-host on the available MI300X test hardware — matches the legacy
  // AMD `wait_local(work)` direct main-QP CQ-poll pattern, which was the
  // validated cross-host surface.
  // (BNXT-specific note: ncqe=1 + phase compression means the NIC
  // overwrites the single CQE slot per WQE completion; ringing the CQ
  // doorbell here would cause the NIC to flush the QP into LOC_QP_OP,
  // so just track cqe_ci locally.)
  spin_poll_or_trap(nic, &mainQp->cq_sq, lastIdx);
  mainQp->cq_sq.cqe_ci = lastIdx + 1;
  __atomic_fetch_add(
      reinterpret_cast<unsigned long long*>(counterRemoteAddr.addr),
      static_cast<unsigned long long>(counterVal),
      __ATOMIC_RELEASE);
  (void)companionQp;
  (void)counterSinkAddr;
#else
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
      lastIdx);

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
#endif
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
#if defined(__HIP_PLATFORM_AMD__)
  // AMD counter-only fast path: when sigVal == 0 (P2pIbgdaTransportDevice
  // routes counter-only puts through signal_counter with sigVal=0 against
  // a discardSignalSlot — atomic FA on the slot is purely there to give
  // NVIDIA mlx5's companion-QP something to wait on; on AMD we don't need
  // it), skip posting the slow atomic-FA WQE entirely. The prior put's
  // WRITE was posted with CQ_UPDATE, so a CQE is on its way. Spin-poll
  // until we observe the new CQE and bump the local counter via GPU atomic.
  //
  // Required on AMD across both BNXT (no inter-QP wait WQE primitive) and
  // mlx5 (WAIT WQE / companion-QP loopback path observed to never advance
  // cross-host on the available MI300X test hardware) — matches the legacy
  // `wait_local(work)` direct main-QP CQ-poll pattern, which was the
  // validated cross-host surface.
  if (sigVal == 0) {
    // Wait for the last reserved WQE (the prior put() may have chunked
    // into multiple WRITE WQEs when size > PIPES_GDA_VERBS_MAX_TRANSFER_SIZE).
    uint64_t lastWqeIdx = mainQp->sq_rsvd_index - 1;
    spin_poll_or_trap(nic, &mainQp->cq_sq, lastWqeIdx);
    mainQp->cq_sq.cqe_ci = lastWqeIdx + 1;
    __atomic_fetch_add(
        reinterpret_cast<unsigned long long*>(counterRemoteAddr.addr),
        static_cast<unsigned long long>(counterVal),
        __ATOMIC_RELEASE);
    (void)companionQp;
    (void)counterSinkAddr;
    (void)sigSinkAddr;
    (void)sigRemoteAddr;
    return;
  }
#endif

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

#if defined(__HIP_PLATFORM_AMD__)
  // AMD: signal-with-real-target path. Spin-poll on mainQp's CQ con_indx,
  // advance cqe_ci locally, then GPU-atomic-increment the local counter.
  // Same rationale as the sigVal==0 fast path above — companion-QP/WAIT
  // path is broken on AMD across BNXT and mlx5 cross-host. (BNXT-specific
  // note: don't ring the CQ doorbell — for ncqe=1 it triggers a NIC flush
  // into LOC_QP_OP.)
  spin_poll_or_trap(nic, &mainQp->cq_sq, sigIdx);
  mainQp->cq_sq.cqe_ci = sigIdx + 1;
  __atomic_fetch_add(
      reinterpret_cast<unsigned long long*>(counterRemoteAddr.addr),
      static_cast<unsigned long long>(counterVal),
      __ATOMIC_RELEASE);
  (void)companionQp;
  (void)counterSinkAddr;
#else
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
#endif
}

// =============================================================================
// DOCA-aligned overloads
// =============================================================================
//
// Each function below has the SAME signature as its `doca_*` counterpart in
// `<device/doca_gpunetio_dev_verbs_*.cuh>`, with `doca_` swapped for
// `pipes_gda_`. They overload the NicBackend-explicit forms above:
//   - Different parameter list (no leading `nic`, plus DOCA-only params).
//   - Different template parameters (`<int MODE/SCOPE/HANDLER>` vs
//     `<typename NicBackend>`) — disambiguates explicit-template calls.
// Each forwards to the NicBackend-explicit form with a stack-local
// `ActiveNicBackend nic{}`. The backend is a stateless empty struct, so
// the temporary is zero-cost (compiler elides it).

template <int MODE = 0>
__device__ __forceinline__ uint64_t pipes_gda_gpu_dev_verbs_reserve_wq_slots(
    pipes_gda_gpu_dev_verbs_qp* qp,
    uint32_t numSlots) {
  ActiveNicBackend nic{};
  return pipes_gda_gpu_dev_verbs_reserve_wq_slots(nic, qp, numSlots);
}

__device__ __forceinline__ pipes_gda_gpu_dev_verbs_wqe*
pipes_gda_gpu_dev_verbs_get_wqe_ptr(
    pipes_gda_gpu_dev_verbs_qp* qp,
    uint64_t wqeIdx) {
  ActiveNicBackend nic{};
  return pipes_gda_gpu_dev_verbs_get_wqe_ptr(nic, qp, wqeIdx);
}

__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_wqe_prepare_nop(
    pipes_gda_gpu_dev_verbs_qp* qp,
    pipes_gda_gpu_dev_verbs_wqe* wqe,
    uint64_t wqeIdx,
    uint8_t ctrlFlags) {
  (void)ctrlFlags;
  ActiveNicBackend nic{};
  pipes_gda_gpu_dev_verbs_wqe_prepare_nop(nic, qp, wqe, wqeIdx);
}

__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_wqe_prepare_write(
    pipes_gda_gpu_dev_verbs_qp* qp,
    pipes_gda_gpu_dev_verbs_wqe* wqe,
    uint64_t wqeIdx,
    uint8_t opcode,
    uint8_t ctrlFlags,
    uint32_t lkey_id,
    uint64_t remoteAddr,
    uint32_t remoteKey,
    uint64_t localAddr,
    uint32_t localKey,
    uint32_t size) {
  (void)opcode;
  (void)lkey_id;
  ActiveNicBackend nic{};
  pipes_gda_gpu_dev_verbs_wqe_prepare_write(
      nic,
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

__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_wqe_prepare_atomic(
    pipes_gda_gpu_dev_verbs_qp* qp,
    pipes_gda_gpu_dev_verbs_wqe* wqe,
    uint64_t wqeIdx,
    uint8_t opcode,
    uint8_t ctrlFlags,
    uint64_t remoteAddr,
    uint32_t remoteKey,
    uint64_t localAddr,
    uint32_t localKey,
    uint32_t size,
    uint64_t addVal,
    uint64_t compareVal) {
  (void)opcode;
  ActiveNicBackend nic{};
  pipes_gda_gpu_dev_verbs_wqe_prepare_atomic(
      nic,
      qp,
      wqe,
      wqeIdx,
      ctrlFlags,
      remoteAddr,
      remoteKey,
      localAddr,
      localKey,
      size,
      addVal,
      compareVal);
}

template <int MODE = 0>
__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_mark_wqes_ready(
    pipes_gda_gpu_dev_verbs_qp* qp,
    uint64_t firstIdx,
    uint64_t lastIdx) {
  ActiveNicBackend nic{};
  pipes_gda_gpu_dev_verbs_mark_wqes_ready(nic, qp, firstIdx, lastIdx);
}

template <int MODE = 0, int SCOPE = 0, int HANDLER = 0>
__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_submit(
    pipes_gda_gpu_dev_verbs_qp* qp,
    uint64_t nextWqeIdx) {
  ActiveNicBackend nic{};
  pipes_gda_gpu_dev_verbs_submit(nic, qp, nextWqeIdx);
}

template <int MODE = 0, int HANDLER = 0>
__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_wait(
    pipes_gda_gpu_dev_verbs_qp* qp,
    uint64_t ticket) {
  ActiveNicBackend nic{};
  pipes_gda_gpu_dev_verbs_wait(nic, qp, ticket);
}

template <int MODE = 0, int HANDLER = 0, int EXEC = 0>
__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_put(
    pipes_gda_gpu_dev_verbs_qp* qp,
    pipes_gda_gpu_dev_verbs_addr raddr,
    pipes_gda_gpu_dev_verbs_addr laddr,
    std::size_t size,
    uint64_t* out_ticket) {
  ActiveNicBackend nic{};
  pipes_gda_gpu_dev_verbs_put(nic, qp, raddr, laddr, size, out_ticket);
}

template <int OP = 0, int MODE = 0, int HANDLER = 0>
__device__ __forceinline__ void pipes_gda_gpu_dev_verbs_signal_counter(
    pipes_gda_gpu_dev_verbs_qp* mainQp,
    pipes_gda_gpu_dev_verbs_addr sigRemoteAddr,
    pipes_gda_gpu_dev_verbs_addr sigSinkAddr,
    uint64_t sigVal,
    pipes_gda_gpu_dev_verbs_qp* companionQp,
    pipes_gda_gpu_dev_verbs_addr counterRemoteAddr,
    pipes_gda_gpu_dev_verbs_addr counterSinkAddr,
    uint64_t counterVal) {
  ActiveNicBackend nic{};
  pipes_gda_gpu_dev_verbs_signal_counter(
      nic,
      mainQp,
      sigRemoteAddr,
      sigSinkAddr,
      sigVal,
      companionQp,
      counterRemoteAddr,
      counterSinkAddr,
      counterVal);
}

template <int MODE = 0>
__device__ __forceinline__ int pipes_gda_gpu_dev_verbs_poll_one_cq_at(
    pipes_gda_gpu_dev_verbs_cq* cq,
    uint64_t consIndex) {
  ActiveNicBackend nic{};
  return pipes_gda_gpu_dev_verbs_poll_one_cq_at(nic, cq, consIndex);
}

template <int MODE = 0, int HANDLER = 0>
__device__ __forceinline__ void pipes_gda_fence(
    pipes_gda_gpu_dev_verbs_qp* qp) {
  ActiveNicBackend nic{};
  pipes_gda_fence(nic, qp);
}

} // namespace pipes_gda
