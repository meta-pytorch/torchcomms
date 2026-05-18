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
// MLX5 (Mellanox/NVIDIA ConnectX) NIC Backend for pipes-gda
// =============================================================================
//
// Device-side WQE construction, doorbell, and CQ polling for mlx5 NICs.
// Uses mlx5-specific WQE format (4 x 16-byte segments = 64 bytes),
// DBREC + BlueFlame doorbell mechanism, and owner-bit CQE polling.
//
// This backend is used by P2pIbgdaTransportDevice<Mlx5NicBackend>.
// =============================================================================

#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>

#include "nic/Mlx5Hsi.h" // @manual
#include "verbs/AmdVerbsCompat.h" // @manual
#include "verbs/VerbsDev.h" // @manual

namespace pipes_gda {

struct Mlx5NicBackend {
  static constexpr const char* vendorPrefix() {
    return "mlx5";
  }
  static constexpr uint16_t vendorId() {
    return 0x02c9;
  }
  static inline uint32_t swapMkey(uint32_t key) {
    return __builtin_bswap32(key);
  }
  static inline uint32_t networkByteOrderKey(uint32_t hostKey) {
    return htobe32(hostKey);
  }

  __device__ pipes_gda_gpu_dev_verbs_wqe* getWqePtr(
      pipes_gda_gpu_dev_verbs_qp* qp,
      uint64_t wqeIdx) const {
    uint16_t maskedIdx = static_cast<uint16_t>(wqeIdx) & qp->sq_wqe_mask;
    return reinterpret_cast<pipes_gda_gpu_dev_verbs_wqe*>(qp->sq_wqe_daddr) +
        maskedIdx;
  }

  __device__ uint64_t
  reserveWqSlots(pipes_gda_gpu_dev_verbs_qp* qp, uint32_t numSlots) {
    return amd_atomic_add_device(
        &qp->sq_rsvd_index, static_cast<uint64_t>(numSlots));
  }

  __device__ void markWqesReady(
      pipes_gda_gpu_dev_verbs_qp* qp,
      uint64_t firstIdx,
      uint64_t lastIdx) {
    while (amd_load_relaxed_device(&qp->sq_ready_index) < firstIdx) {
    }
    // System-scope release fence: WQE data is in host memory (SQ buffer via
    // hipHostRegister). Must be visible to the NIC (a PCIe device) before the
    // doorbell ring. Agent-scope is insufficient — it only orders within the
    // GPU's L2 cache domain.
    amd_fence_release_system();
    amd_atomic_max_device(&qp->sq_ready_index, lastIdx + 1);
  }

  __device__ void ringDoorbell(
      pipes_gda_gpu_dev_verbs_qp* qp,
      uint64_t nextWqeIdx) {
    uint32_t pi =
        static_cast<uint32_t>(nextWqeIdx) & PIPES_GDA_VERBS_WQE_PI_MASK;

    *reinterpret_cast<volatile uint32_t*>(
        qp->sq_dbrec + PIPES_GDA_IB_MLX5_SND_DBR) = amd_bswap32(pi);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);

    uint64_t lastWqeIdx = nextWqeIdx - 1;
    pipes_gda_gpu_dev_verbs_wqe* lastWqe = getWqePtr(qp, lastWqeIdx);
    uint64_t dbVal = *reinterpret_cast<volatile uint64_t*>(lastWqe);

    amd_store_doorbell_sys_u64(qp->sq_db, dbVal);

    uint64_t dbAddr = __hip_atomic_load(
        reinterpret_cast<uint64_t*>(&qp->sq_db),
        __ATOMIC_RELAXED,
        __HIP_MEMORY_SCOPE_AGENT);
    dbAddr ^= 0x100;
    __hip_atomic_store(
        reinterpret_cast<uint64_t*>(&qp->sq_db),
        dbAddr,
        __ATOMIC_RELAXED,
        __HIP_MEMORY_SCOPE_AGENT);
  }

  __device__ void prepareRdmaWriteWqe(
      pipes_gda_gpu_dev_verbs_qp* qp,
      pipes_gda_gpu_dev_verbs_wqe* wqe,
      uint64_t wqeIdx,
      uint8_t ctrlFlags,
      uint64_t remoteAddr,
      uint32_t remoteKey,
      uint64_t localAddr,
      uint32_t localKey,
      std::size_t size) {
    pipes_gda_gpu_dev_verbs_wqe_ctrl_seg cseg = {};
    cseg.opmod_idx_opcode = amd_bswap32(
        (static_cast<uint32_t>(wqeIdx) << PIPES_GDA_VERBS_WQE_IDX_SHIFT) |
        PIPES_GDA_IB_MLX5_OPCODE_RDMA_WRITE);
    cseg.qpn_ds = amd_bswap32(qp->sq_num_shift8 | 3);
    cseg.fm_ce_se = ctrlFlags;

    pipes_gda_ib_mlx5_wqe_raddr_seg rseg = {};
    rseg.raddr = amd_bswap64(remoteAddr);
    rseg.rkey = remoteKey;

    pipes_gda_ib_mlx5_wqe_data_seg dseg = {};
    dseg.byte_count = amd_bswap32(static_cast<uint32_t>(size));
    dseg.lkey = localKey;
    dseg.addr = amd_bswap64(localAddr);

    amd_store_wqe_seg(
        reinterpret_cast<uint64_t*>(&wqe->dseg0),
        reinterpret_cast<uint64_t*>(&cseg));
    amd_store_wqe_seg(
        reinterpret_cast<uint64_t*>(&wqe->dseg1),
        reinterpret_cast<uint64_t*>(&rseg));
    amd_store_wqe_seg(
        reinterpret_cast<uint64_t*>(&wqe->dseg2),
        reinterpret_cast<uint64_t*>(&dseg));
  }

  __device__ void prepareAtomicFaWqe(
      pipes_gda_gpu_dev_verbs_qp* qp,
      pipes_gda_gpu_dev_verbs_wqe* wqe,
      uint64_t wqeIdx,
      uint8_t ctrlFlags,
      uint64_t remoteAddr,
      uint32_t remoteKey,
      uint64_t localAddr,
      uint32_t localKey,
      uint64_t addVal) {
    pipes_gda_gpu_dev_verbs_wqe_ctrl_seg cseg = {};
    cseg.opmod_idx_opcode = amd_bswap32(
        (static_cast<uint32_t>(wqeIdx) << PIPES_GDA_VERBS_WQE_IDX_SHIFT) |
        PIPES_GDA_IB_MLX5_OPCODE_ATOMIC_FA);
    cseg.qpn_ds = amd_bswap32(
        qp->sq_num_shift8 | PIPES_GDA_VERBS_WQE_SEG_CNT_ATOMIC_FA_CAS);
    cseg.fm_ce_se = ctrlFlags;

    pipes_gda_ib_mlx5_wqe_raddr_seg rseg = {};
    rseg.raddr = amd_bswap64(remoteAddr);
    rseg.rkey = remoteKey;

    pipes_gda_ib_mlx5_wqe_atomic_seg aseg = {};
    aseg.swap_add = amd_bswap64(addVal);
    aseg.compare = 0;

    pipes_gda_ib_mlx5_wqe_data_seg dseg = {};
    dseg.byte_count = amd_bswap32(sizeof(uint64_t));
    dseg.lkey = localKey;
    dseg.addr = amd_bswap64(localAddr);

    amd_store_wqe_seg(
        reinterpret_cast<uint64_t*>(&wqe->dseg0),
        reinterpret_cast<uint64_t*>(&cseg));
    amd_store_wqe_seg(
        reinterpret_cast<uint64_t*>(&wqe->dseg1),
        reinterpret_cast<uint64_t*>(&rseg));
    amd_store_wqe_seg(
        reinterpret_cast<uint64_t*>(&wqe->dseg2),
        reinterpret_cast<uint64_t*>(&aseg));
    amd_store_wqe_seg(
        reinterpret_cast<uint64_t*>(&wqe->dseg3),
        reinterpret_cast<uint64_t*>(&dseg));
  }

  __device__ void prepareNopWqe(
      pipes_gda_gpu_dev_verbs_qp* qp,
      pipes_gda_gpu_dev_verbs_wqe* wqe,
      uint64_t wqeIdx) {
    pipes_gda_gpu_dev_verbs_wqe_ctrl_seg cseg = {};
    cseg.opmod_idx_opcode = amd_bswap32(
        (static_cast<uint32_t>(wqeIdx) << PIPES_GDA_VERBS_WQE_IDX_SHIFT) |
        PIPES_GDA_IB_MLX5_OPCODE_NOP);
    cseg.qpn_ds = amd_bswap32(qp->sq_num_shift8 | 1);
    cseg.fm_ce_se = PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE;

    amd_store_wqe_seg(
        reinterpret_cast<uint64_t*>(&wqe->dseg0),
        reinterpret_cast<uint64_t*>(&cseg));
  }

  // ===========================================================================
  // prepareInlineWriteWqe - Construct inline RDMA Write WQE
  // ===========================================================================
  //
  // Builds a 3-segment WQE: ctrl + raddr + inline data.
  // Used by reset_signal() to write a zero to the remote signal buffer
  // without needing a local memory region (data is embedded in the WQE).
  template <typename T>
  __device__ void prepareInlineWriteWqe(
      pipes_gda_gpu_dev_verbs_qp* qp,
      pipes_gda_gpu_dev_verbs_wqe* wqe,
      uint64_t wqeIdx,
      uint8_t ctrlFlags,
      uint64_t remoteAddr,
      uint32_t remoteKey,
      T value) {
    static_assert(
        sizeof(T) <= PIPES_GDA_VERBS_MAX_INLINE_SIZE, "inline too large");

    uint32_t dsCount = (sizeof(T) <= 16)
        ? PIPES_GDA_VERBS_WQE_SEG_CNT_RDMA_WRITE_INL_MIN
        : PIPES_GDA_VERBS_WQE_SEG_CNT_RDMA_WRITE_INL_MAX;

    pipes_gda_gpu_dev_verbs_wqe_ctrl_seg cseg = {};
    cseg.opmod_idx_opcode = amd_bswap32(
        (static_cast<uint32_t>(wqeIdx) << PIPES_GDA_VERBS_WQE_IDX_SHIFT) |
        PIPES_GDA_IB_MLX5_OPCODE_RDMA_WRITE);
    cseg.qpn_ds = amd_bswap32(qp->sq_num_shift8 | dsCount);
    cseg.fm_ce_se = ctrlFlags;

    pipes_gda_ib_mlx5_wqe_raddr_seg rseg = {};
    rseg.raddr = amd_bswap64(remoteAddr);
    rseg.rkey = remoteKey;

    // Inline data segment: header (4 bytes) + payload
    struct {
      pipes_gda_ib_mlx5_wqe_inl_data_seg hdr;
      T payload;
    } __attribute__((__packed__)) inlSeg = {};

    inlSeg.hdr.byte_count = amd_bswap32(
        static_cast<uint32_t>(sizeof(T)) | PIPES_GDA_IB_MLX5_INLINE_SEG);
    inlSeg.payload = value;

    amd_store_wqe_seg(
        reinterpret_cast<uint64_t*>(&wqe->dseg0),
        reinterpret_cast<uint64_t*>(&cseg));
    amd_store_wqe_seg(
        reinterpret_cast<uint64_t*>(&wqe->dseg1),
        reinterpret_cast<uint64_t*>(&rseg));
    amd_store_wqe_seg(
        reinterpret_cast<uint64_t*>(&wqe->dseg2),
        reinterpret_cast<uint64_t*>(&inlSeg));
  }

  // ===========================================================================
  // prepareWaitWqe - Construct WAIT WQE for cross-QP synchronization
  // ===========================================================================
  //
  // Builds a 2-segment WQE: ctrl + wait. The WAIT WQE tells the NIC to stall
  // processing on this QP until the referenced CQ has completed the target WQE.
  // Used by companion QP to wait on main QP completion before posting counter.
  __device__ void prepareWaitWqe(
      pipes_gda_gpu_dev_verbs_qp* qp,
      pipes_gda_gpu_dev_verbs_wqe* wqe,
      uint64_t wqeIdx,
      uint8_t ctrlFlags,
      uint32_t targetCqNum,
      uint64_t targetWqeIdx) {
    pipes_gda_gpu_dev_verbs_wqe_ctrl_seg cseg = {};
    cseg.opmod_idx_opcode = amd_bswap32(
        (static_cast<uint32_t>(wqeIdx) << PIPES_GDA_VERBS_WQE_IDX_SHIFT) |
        PIPES_GDA_IB_MLX5_OPCODE_WAIT);
    cseg.qpn_ds =
        amd_bswap32(qp->sq_num_shift8 | PIPES_GDA_VERBS_WQE_SEG_CNT_WAIT);
    cseg.fm_ce_se = ctrlFlags;

    pipes_gda_gpu_dev_verbs_wqe_wait_seg wseg = {};
    wseg.max_index = amd_bswap32(
        static_cast<uint32_t>(targetWqeIdx) & PIPES_GDA_VERBS_WQE_PI_MASK);
    wseg.qpn_cqn = amd_bswap32(targetCqNum);

    amd_store_wqe_seg(
        reinterpret_cast<uint64_t*>(&wqe->dseg0),
        reinterpret_cast<uint64_t*>(&cseg));
    amd_store_wqe_seg(
        reinterpret_cast<uint64_t*>(&wqe->dseg1),
        reinterpret_cast<uint64_t*>(&wseg));
  }

  // ===========================================================================
  // pollOneCqAt - Non-blocking CQ poll at a specific index
  // ===========================================================================
  //
  // Returns 0 on success, EBUSY if not yet complete.
  // Unlike pollCqAt() which spins until completion, this returns immediately
  // so the caller can implement timeout logic.
  __device__ int pollOneCqAt(
      pipes_gda_gpu_dev_verbs_cq* cq,
      uint64_t consIndex) {
    pipes_gda_ib_mlx5_cqe64* cqeBase =
        reinterpret_cast<pipes_gda_ib_mlx5_cqe64*>(cq->cqe_daddr);
    const uint32_t cqeNum = cq->cqe_num;
    uint32_t idx = static_cast<uint32_t>(consIndex) & (cqeNum - 1);
    pipes_gda_ib_mlx5_cqe64* cqe64 = &cqeBase[idx];

    uint64_t cqeCi = amd_load_relaxed_device(&cq->cqe_ci);
    if (consIndex < cqeCi)
      return 0;

    if (consIndex >= cqeCi + cqeNum)
      return EBUSY;

    uint32_t cqeChunk =
        amd_load_relaxed_sys(reinterpret_cast<uint32_t*>(&cqe64->wqe_counter));
    cqeChunk = amd_bswap32(cqeChunk);
    uint16_t wqeCounter = cqeChunk >> 16;
    uint8_t opown = cqeChunk & 0xff;

    if ((opown & PIPES_GDA_IB_MLX5_CQE_OWNER_MASK) ^ !!(consIndex & cqeNum))
      return EBUSY;
    if (wqeCounter != (static_cast<uint32_t>(consIndex) & 0xffff))
      return EBUSY;

    uint8_t opcode = opown >> PIPES_GDA_VERBS_MLX5_CQE_OPCODE_SHIFT;

    amd_fence_acquire_system();
    amd_atomic_max_device(&cq->cqe_ci, consIndex + 1);

    uint32_t ci =
        static_cast<uint32_t>(consIndex + 1) & PIPES_GDA_VERBS_CQE_CI_MASK;
    amd_store_release_sys_u32(
        reinterpret_cast<uint32_t*>(cq->dbrec), amd_bswap32(ci));

    return (opcode == PIPES_GDA_IB_MLX5_CQE_REQ_ERR) ? -5 : 0;
  }

  __device__ int pollCqAt(
      pipes_gda_gpu_dev_verbs_qp* qp,
      pipes_gda_gpu_dev_verbs_cq* cq,
      uint64_t consIndex) {
    pipes_gda_ib_mlx5_cqe64* cqeBase =
        reinterpret_cast<pipes_gda_ib_mlx5_cqe64*>(cq->cqe_daddr);
    const uint32_t cqeNum = cq->cqe_num;
    uint32_t idx = static_cast<uint32_t>(consIndex) & (cqeNum - 1);
    pipes_gda_ib_mlx5_cqe64* cqe64 = &cqeBase[idx];

    uint8_t opown;
    uint32_t cqeChunk;
    uint16_t wqeCounter;

    do {
      uint64_t cqeCi = amd_load_relaxed_device(&cq->cqe_ci);
      if (consIndex < cqeCi)
        return 0;

      cqeChunk = amd_load_relaxed_sys(
          reinterpret_cast<uint32_t*>(&cqe64->wqe_counter));
      cqeChunk = amd_bswap32(cqeChunk);
      wqeCounter = cqeChunk >> 16;
      opown = cqeChunk & 0xff;
    } while (
        (consIndex >= amd_load_relaxed_device(&cq->cqe_ci) + cqeNum) ||
        ((opown & PIPES_GDA_IB_MLX5_CQE_OWNER_MASK) ^ !!(consIndex & cqeNum)) ||
        (wqeCounter != (static_cast<uint32_t>(consIndex) & 0xffff)));

    uint8_t opcode = opown >> PIPES_GDA_VERBS_MLX5_CQE_OPCODE_SHIFT;

    amd_fence_acquire_system();
    amd_atomic_max_device(&cq->cqe_ci, consIndex + 1);

    uint32_t ci =
        static_cast<uint32_t>(consIndex + 1) & PIPES_GDA_VERBS_CQE_CI_MASK;
    amd_store_release_sys_u32(
        reinterpret_cast<uint32_t*>(cq->dbrec), amd_bswap32(ci));

    return (opcode == PIPES_GDA_IB_MLX5_CQE_REQ_ERR) ? -5 : 0;
  }
};

} // namespace pipes_gda
