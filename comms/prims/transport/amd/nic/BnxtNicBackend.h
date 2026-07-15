// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// BNXT (Broadcom) NIC Backend for pipes-gda
// =============================================================================
//
// Device-side WQE construction, doorbell, and CQ polling for Broadcom BNXT
// NICs. Uses BNXT-specific WQE format (3 x 16-byte slots = 48 bytes), single
// 64-bit atomic doorbell with epoch bit, and phase-bit CQE polling.
//
// Key differences from mlx5:
//   - WQE: 3 x 16-byte slots (48 bytes) vs mlx5's 4 x 16-byte segments (64
//   bytes)
//   - Doorbell: single 64-bit atomic write with epoch bit (no DBREC +
//   BlueFlame)
//   - CQ: CQE compression (depth=1), phase bit toggles on each completion
//   - Keys: native byte order (no big-endian swap needed)
//   - MSN table: at end of SQ buffer, tracks PSN per WQE
//
// This backend is used by P2pIbgdaTransportDevice<BnxtNicBackend>.
// =============================================================================

#pragma once

#include <cerrno> // EBUSY for non-blocking pollOneCqAt
#include <cstddef>
#include <cstdint>

#include "HipDeviceCompat.h" // @manual
#include "nic/BnxtHsi.h" // @manual
#include "pipes_gda/PipesGdaDev.h" // @manual

namespace pipes_gda {

struct BnxtNicBackend {
  static constexpr const char* vendorPrefix() {
    return "bnxt_re";
  }
  static constexpr uint16_t vendorId() {
    return 0x14E4;
  }
  static inline uint32_t swapMkey(uint32_t key) {
    return key; // bnxt uses native byte order
  }
  static inline uint32_t networkByteOrderKey(uint32_t hostKey) {
    return hostKey; // bnxt uses native byte order
  }

  // Spinlock around bnxt_re's per-QP shared state (msn, sq_tail, sq_flags
  // epoch bit). Concurrent warps that update those fields without holding
  // the lock corrupt the MSN-table tail and produce doorbell values with
  // the wrong epoch, which surfaces on the device as `Memory access fault
  // by GPU on address (nil)`.
  __device__ void lockQp(pipes_gda_gpu_dev_verbs_qp* qp) {
    int expected;
    do {
      expected = 0;
    } while (0 ==
             __hip_atomic_compare_exchange_strong(
                 &qp->nic.bnxt.sq_lock,
                 &expected,
                 1,
                 __ATOMIC_ACQUIRE,
                 __ATOMIC_ACQUIRE,
                 __HIP_MEMORY_SCOPE_SYSTEM));
  }

  __device__ void unlockQp(pipes_gda_gpu_dev_verbs_qp* qp) {
    __hip_atomic_store(
        &qp->nic.bnxt.sq_lock, 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }

  // Back-pressure: spin until the SQ has at least `requestedSlots` free
  // entries. Without it, warps can submit WQEs faster than the NIC completes
  // them; the producer index then wraps past the consumer, new WQEs overwrite
  // slots still owned by the NIC, and the resulting NIC-side fault surfaces
  // as `Memory access fault by GPU on address (nil)`.
  // Caller must hold the QP spinlock.
  __device__ void waitForSqSlots(
      pipes_gda_gpu_dev_verbs_qp* qp,
      uint32_t requestedSlots) {
    uint32_t sqDepth = qp->nic.bnxt.sq_depth;
    if (sqDepth == 0) {
      return;
    }
    volatile char* cqeBase =
        reinterpret_cast<volatile char*>(qp->cq_sq.cqe_daddr);
    volatile pipes_gda_bnxt_req_cqe* cqe =
        reinterpret_cast<volatile pipes_gda_bnxt_req_cqe*>(cqeBase);
    // Bounded spin: an SQ that never drains means the NIC has stopped
    // completing WQEs (link down or QP in error), so an unbounded spin would
    // hang the kernel forever. Trap once exhausted so the stall surfaces.
    constexpr uint64_t kMaxSpins = 10000000ULL;
    for (uint64_t spins = 0; spins < kMaxSpins; ++spins) {
      uint32_t conIdx = cqe->con_indx & 0xFFFF;
      uint32_t sqHead = (conIdx * PIPES_GDA_BNXT_GDA_WQE_SLOT_COUNT) % sqDepth;
      uint32_t sqTail = qp->nic.bnxt.sq_tail;
      uint32_t consumed = (sqTail - sqHead + sqDepth) % sqDepth;
      uint32_t available = sqDepth - consumed;
      if (available >= requestedSlots) {
        return;
      }
    }
    printf(
        "BNXT waitForSqSlots TIMEOUT: requested=%u sq_depth=%u sq_tail=%u "
        "con_indx=%u sq_id=%u\n",
        requestedSlots,
        sqDepth,
        qp->nic.bnxt.sq_tail,
        cqe->con_indx & 0xFFFF,
        qp->nic.bnxt.sq_id);
    // __builtin_trap() is the portable device trap (s_trap on AMDGPU); the
    // CUDA-only __trap() intrinsic is not declared in this HIP header context.
    __builtin_trap();
  }

  __device__ void* getBnxtWqeSlot(
      pipes_gda_gpu_dev_verbs_qp* qp,
      uint32_t slotIdx) const {
    // Wrap per slot so WQEs that straddle the SQ buffer end don't write
    // past it into the MSN-table region. Without this wrap a 3-slot WQE
    // posted with slotIdx == sq_depth-1 writes slots [sq_depth-1, sq_depth,
    // sq_depth+1], which trashes the MSN-table tail and the next WQE's
    // CQE comes back with LOC_QP_OP.
    if (slotIdx >= qp->nic.bnxt.sq_depth) {
      slotIdx -= qp->nic.bnxt.sq_depth;
    }
    return reinterpret_cast<uint8_t*>(qp->sq_wqe_daddr) +
        static_cast<size_t>(slotIdx) * PIPES_GDA_BNXT_SLOT_SIZE_BB;
  }

  __device__ uint32_t
  bnxtWqeIdxToSlot(pipes_gda_gpu_dev_verbs_qp* qp, uint64_t wqeIdx) const {
    return static_cast<uint32_t>(
        (wqeIdx * PIPES_GDA_BNXT_GDA_WQE_SLOT_COUNT) % qp->nic.bnxt.sq_depth);
  }

  __device__ pipes_gda_gpu_dev_verbs_wqe* getWqePtr(
      pipes_gda_gpu_dev_verbs_qp* qp,
      uint64_t wqeIdx) const {
    uint32_t slotIdx = bnxtWqeIdxToSlot(qp, wqeIdx);
    return reinterpret_cast<pipes_gda_gpu_dev_verbs_wqe*>(
        reinterpret_cast<uint8_t*>(qp->sq_wqe_daddr) +
        static_cast<size_t>(slotIdx) * PIPES_GDA_BNXT_SLOT_SIZE_BB);
  }

  __device__ uint64_t
  reserveWqSlots(pipes_gda_gpu_dev_verbs_qp* qp, uint32_t numWqes) {
    return amd_atomic_add_device(
        &qp->sq_rsvd_index, static_cast<uint64_t>(numWqes));
  }

  __device__ void markWqesReady(
      pipes_gda_gpu_dev_verbs_qp* qp,
      uint64_t firstIdx,
      uint64_t lastIdx) {
    while (amd_load_relaxed_device(&qp->sq_ready_index) < firstIdx) {
    }
    amd_fence_release_device();
    amd_atomic_max_device(&qp->sq_ready_index, lastIdx + 1);
  }

  __device__ void bnxtIncrTail(
      pipes_gda_gpu_dev_verbs_qp* qp,
      uint32_t slotCount) {
    uint32_t newTail = qp->nic.bnxt.sq_tail + slotCount;
    if (newTail >= qp->nic.bnxt.sq_depth) {
      newTail %= qp->nic.bnxt.sq_depth;
      qp->nic.bnxt.sq_flags ^= 1UL << PIPES_GDA_BNXT_FLAG_EPOCH_TAIL_SHIFT;
    }
    qp->nic.bnxt.sq_tail = newTail;
  }

  // MSN-entry packing helpers. The bnxt_re MSN table stores one 64-bit
  // entry per WQE describing the PSN range that WQE consumes. The layout
  // is fixed by the NIC HSI; the helpers below are pure bit-encoders so
  // the call site reads as data flow (compute PSNs -> encode -> store).
  __device__ static __forceinline__ uint32_t
  packetsForMessage(uint32_t msgLen, uint32_t mtu) {
    // Zero-length messages (e.g. RDMA-WRITE-IMM with len=0) still consume
    // one PSN slot; positive lengths take ceil(msgLen / mtu).
    if (msgLen == 0) {
      return 1;
    }
    return mtu > 0 ? (msgLen + mtu - 1) / mtu : 0;
  }

  __device__ static __forceinline__ uint64_t
  encodeMsnEntry(uint32_t slotIdx, uint32_t startPsn, uint32_t nextPsn) {
    constexpr uint64_t kStartIdxBits = 0xFFFFULL;
    constexpr uint64_t kPsnBits = 0xFFFFFFULL;
    const uint64_t startIdxField =
        (static_cast<uint64_t>(slotIdx) & kStartIdxBits)
        << PIPES_GDA_BNXT_MSN_START_IDX_SHIFT;
    const uint64_t nextPsnField = (static_cast<uint64_t>(nextPsn) & kPsnBits)
        << PIPES_GDA_BNXT_MSN_NEXT_PSN_SHIFT;
    const uint64_t startPsnField = (static_cast<uint64_t>(startPsn) & kPsnBits)
        << PIPES_GDA_BNXT_MSN_START_PSN_SHIFT;
    return startIdxField | nextPsnField | startPsnField;
  }

  // Advance qp->nic.bnxt.{psn,msn} and write one packed MSN entry at the
  // current msn cursor. Caller has already prepared the WQE at sq_tail.
  __device__ void bnxtFillMsn(pipes_gda_gpu_dev_verbs_qp* qp, uint32_t msgLen) {
    auto& bnxt = qp->nic.bnxt;

    const uint32_t startPsn = bnxt.psn;
    const uint32_t newPsn = startPsn + packetsForMessage(msgLen, bnxt.mtu);

    // Compute the slot the about-to-post WQE occupies, then write the packed
    // entry directly at the current msn index (no temporary struct).
    auto* msnSlot = reinterpret_cast<uint64_t*>(
        reinterpret_cast<uint8_t*>(bnxt.msntbl) +
        (static_cast<size_t>(bnxt.msn) << bnxt.psn_sz_log2));
    *msnSlot = encodeMsnEntry(bnxt.sq_tail, startPsn, newPsn);

    // Single-writer cursor advance — bnxt.msn is a per-QP table index that
    // wraps at msn_tbl_sz. PSN is the per-QP packet sequence number.
    bnxt.psn = newPsn;
    bnxt.msn = (bnxt.msn + 1) % bnxt.msn_tbl_sz;
  }

  // Encode the SQ doorbell word for the current (sq_tail, epoch, sq_id)
  // state. The doorbell is a single 64-bit value written atomically; we
  // compose the low/high halves separately because the HSI bit fields are
  // grouped that way, then fold them via OR-shift into the final word.
  __device__ static __forceinline__ uint64_t
  composeSqDoorbell(uint32_t sqTail, uint32_t sqFlags, uint32_t sqId) {
    // Low half holds sq_tail in [23:0] and the epoch bit at [24]. The epoch
    // bit toggles every time the tail wraps and the NIC uses it to
    // distinguish "wrap-completed" from "wrap-pending" doorbells.
    const uint32_t epochBit = (sqFlags & PIPES_GDA_BNXT_FLAG_EPOCH_TAIL_MASK)
        << PIPES_GDA_BNXT_DB_EPOCH_TAIL_SHIFT;
    const uint32_t low = sqTail | epochBit;

    // High half encodes the QP id, queue type (SQ here), and a valid bit.
    constexpr uint32_t kTypeSqShifted =
        (static_cast<uint32_t>(PIPES_GDA_BNXT_QUE_TYPE_SQ) &
         PIPES_GDA_BNXT_DB_TYP_MASK)
        << PIPES_GDA_BNXT_DB_TYP_SHIFT;
    constexpr uint32_t kValidBit = 1u << PIPES_GDA_BNXT_DB_VALID_SHIFT;
    const uint32_t high =
        (sqId & PIPES_GDA_BNXT_DB_QID_MASK) | kTypeSqShifted | kValidBit;

    return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32U);
  }

  __device__ void ringDoorbell(
      pipes_gda_gpu_dev_verbs_qp* qp,
      uint64_t /* nextWqeIdx */) {
    const auto& bnxt = qp->nic.bnxt;
    const uint64_t doorbellWord =
        composeSqDoorbell(bnxt.sq_tail, bnxt.sq_flags, bnxt.sq_id);

    // System-scope release: all WQE bytes must be globally visible to the
    // NIC before the doorbell write makes its way through PCIe.
    amd_fence_system();

    // The doorbell BAR is a write-only mapped page; the seq-cst + system
    // scope is what the NIC PCIe spec requires for the write to be
    // observable in-order at the device.
    uint64_t* dbrPtr = const_cast<uint64_t*>(bnxt.dbr);
    __hip_atomic_store(
        dbrPtr, doorbellWord, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  }

  __device__ void prepareRdmaWriteWqe(
      pipes_gda_gpu_dev_verbs_qp* qp,
      pipes_gda_gpu_dev_verbs_wqe* /* unused */,
      uint64_t wqeIdx,
      uint8_t ctrlFlags,
      uint64_t remoteAddr,
      uint32_t remoteKey,
      uint64_t localAddr,
      uint32_t localKey,
      std::size_t size) {
    // Back-pressure to keep producer behind NIC consumer (caller holds lock).
    waitForSqSlots(qp, PIPES_GDA_BNXT_GDA_WQE_SLOT_COUNT);
    uint32_t slotIdx = bnxtWqeIdxToSlot(qp, wqeIdx);

    pipes_gda_bnxt_bsqe* hdr =
        reinterpret_cast<pipes_gda_bnxt_bsqe*>(getBnxtWqeSlot(qp, slotIdx));
    pipes_gda_bnxt_rdma* rdma =
        reinterpret_cast<pipes_gda_bnxt_rdma*>(getBnxtWqeSlot(qp, slotIdx + 1));
    void* sgeSlot = getBnxtWqeSlot(qp, slotIdx + 2);

    // Use INLINE encoding for payloads <= 16B (one WQE slot). NIC reads
    // data directly from the WQE; no SGE-based PCIe fetch from local
    // memory. The inline length goes in the IL field of `rsv_ws_fl_wt`,
    // NOT in `qkey_len`.
    bool useInline = (size <= PIPES_GDA_BNXT_SLOT_SIZE_BB);

    uint32_t wqeType =
        PIPES_GDA_BNXT_HDR_WT_MASK & PIPES_GDA_BNXT_WR_OPCD_RDMA_WRITE;
    uint32_t wqeSize =
        PIPES_GDA_BNXT_HDR_WS_MASK & PIPES_GDA_BNXT_GDA_WQE_SLOT_COUNT;
    uint8_t hdrFlags = PIPES_GDA_BNXT_WR_FLAGS_SIGNALED;
    if (useInline) {
      hdrFlags |= PIPES_GDA_BNXT_WR_FLAGS_INLINE;
    }
    if (ctrlFlags & PIPES_GDA_IB_MLX5_WQE_CTRL_FENCE) {
      hdrFlags |= PIPES_GDA_BNXT_WR_FLAGS_RD_FENCE;
    }

    pipes_gda_bnxt_bsqe hdrVal = {};
    hdrVal.rsv_ws_fl_wt = (wqeSize << PIPES_GDA_BNXT_HDR_WS_SHIFT) |
        (hdrFlags << PIPES_GDA_BNXT_HDR_FLAGS_SHIFT) | wqeType;
    if (useInline) {
      hdrVal.rsv_ws_fl_wt |=
          ((static_cast<uint32_t>(size) & PIPES_GDA_BNXT_HDR_IL_MASK)
           << PIPES_GDA_BNXT_HDR_IL_SHIFT);
    }
    hdrVal.key_immd = 0;
    hdrVal.lhdr.qkey_len = static_cast<uint64_t>(size);

    pipes_gda_bnxt_rdma rdmaVal = {};
    rdmaVal.rva = remoteAddr;
    rdmaVal.rkey = remoteKey;
    rdmaVal.bytes = 0;

    *hdr = hdrVal;
    *rdma = rdmaVal;

    if (useInline) {
      // Copy payload directly from local GPU memory into WQE slot 2.
      const uint8_t* src = reinterpret_cast<const uint8_t*>(localAddr);
      uint8_t* dst = static_cast<uint8_t*>(sgeSlot);
      for (std::size_t i = 0; i < size; ++i) {
        dst[i] = src[i];
      }
      (void)localKey;
    } else {
      pipes_gda_bnxt_sge sgeVal = {};
      sgeVal.pa = localAddr;
      sgeVal.lkey = localKey;
      sgeVal.length = static_cast<uint32_t>(size);
      *static_cast<pipes_gda_bnxt_sge*>(sgeSlot) = sgeVal;
    }

    bnxtFillMsn(qp, static_cast<uint32_t>(size));
    bnxtIncrTail(qp, PIPES_GDA_BNXT_GDA_WQE_SLOT_COUNT);
  }

  __device__ void prepareAtomicFaWqe(
      pipes_gda_gpu_dev_verbs_qp* qp,
      pipes_gda_gpu_dev_verbs_wqe* /* unused */,
      uint64_t wqeIdx,
      uint8_t ctrlFlags,
      uint64_t remoteAddr,
      uint32_t remoteKey,
      uint64_t localAddr,
      uint32_t localKey,
      uint64_t addVal) {
    // Back-pressure to keep producer behind NIC consumer (caller holds lock).
    waitForSqSlots(qp, PIPES_GDA_BNXT_GDA_WQE_SLOT_COUNT);
    uint32_t slotIdx = bnxtWqeIdxToSlot(qp, wqeIdx);

    pipes_gda_bnxt_bsqe* hdr =
        reinterpret_cast<pipes_gda_bnxt_bsqe*>(getBnxtWqeSlot(qp, slotIdx));
    pipes_gda_bnxt_atomic* atomic = reinterpret_cast<pipes_gda_bnxt_atomic*>(
        getBnxtWqeSlot(qp, slotIdx + 1));
    pipes_gda_bnxt_sge* sge =
        reinterpret_cast<pipes_gda_bnxt_sge*>(getBnxtWqeSlot(qp, slotIdx + 2));

    uint32_t wqeType =
        PIPES_GDA_BNXT_HDR_WT_MASK & PIPES_GDA_BNXT_WR_OPCD_ATOMIC_FA;
    uint32_t wqeSize =
        PIPES_GDA_BNXT_HDR_WS_MASK & PIPES_GDA_BNXT_GDA_WQE_SLOT_COUNT;
    // ALWAYS set RD_FENCE on atomic FA. The companion-QP "put-with-counter"
    // pattern requires the atomic to fire only after the preceding RDMA
    // WRITE has been retired by the local NIC. For 8B/64B writes the WRITE
    // finishes inside hardware before the atomic dispatches, so missing
    // fence "appears to work"; once payloads exceed ~1KB the WRITE is still
    // draining and the atomic completes against stale ordering, which on
    // BNXT manifests as the atomic CQE never being retired and the
    // wait_counter loop spinning forever.
    uint8_t hdrFlags =
        PIPES_GDA_BNXT_WR_FLAGS_SIGNALED | PIPES_GDA_BNXT_WR_FLAGS_RD_FENCE;
    (void)ctrlFlags;

    pipes_gda_bnxt_bsqe hdrVal = {};
    hdrVal.rsv_ws_fl_wt = (wqeSize << PIPES_GDA_BNXT_HDR_WS_SHIFT) |
        (hdrFlags << PIPES_GDA_BNXT_HDR_FLAGS_SHIFT) | wqeType;
    hdrVal.key_immd = remoteKey;
    hdrVal.lhdr.rva = remoteAddr;

    pipes_gda_bnxt_atomic atomicVal = {};
    atomicVal.swp_dt = addVal;
    atomicVal.cmp_dt = 0;

    pipes_gda_bnxt_sge sgeVal = {};
    sgeVal.pa = localAddr;
    sgeVal.lkey = localKey;
    sgeVal.length = sizeof(uint64_t);

    *hdr = hdrVal;
    *atomic = atomicVal;
    *sge = sgeVal;

    bnxtFillMsn(qp, sizeof(uint64_t));
    bnxtIncrTail(qp, PIPES_GDA_BNXT_GDA_WQE_SLOT_COUNT);
  }

  __device__ void prepareNopWqe(
      pipes_gda_gpu_dev_verbs_qp* qp,
      pipes_gda_gpu_dev_verbs_wqe* /* unused */,
      uint64_t wqeIdx) {
    uint32_t slotIdx = bnxtWqeIdxToSlot(qp, wqeIdx);

    pipes_gda_bnxt_bsqe* hdr =
        reinterpret_cast<pipes_gda_bnxt_bsqe*>(getBnxtWqeSlot(qp, slotIdx));
    pipes_gda_bnxt_rdma* rdma =
        reinterpret_cast<pipes_gda_bnxt_rdma*>(getBnxtWqeSlot(qp, slotIdx + 1));
    pipes_gda_bnxt_sge* sge =
        reinterpret_cast<pipes_gda_bnxt_sge*>(getBnxtWqeSlot(qp, slotIdx + 2));

    uint32_t wqeType =
        PIPES_GDA_BNXT_HDR_WT_MASK & PIPES_GDA_BNXT_WR_OPCD_RDMA_WRITE;
    uint32_t wqeSize =
        PIPES_GDA_BNXT_HDR_WS_MASK & PIPES_GDA_BNXT_GDA_WQE_SLOT_COUNT;

    pipes_gda_bnxt_bsqe hdrVal = {};
    hdrVal.rsv_ws_fl_wt = (wqeSize << PIPES_GDA_BNXT_HDR_WS_SHIFT) |
        (PIPES_GDA_BNXT_WR_FLAGS_SIGNALED << PIPES_GDA_BNXT_HDR_FLAGS_SHIFT) |
        wqeType;
    hdrVal.key_immd = 0;
    hdrVal.lhdr.qkey_len = 0;

    pipes_gda_bnxt_rdma rdmaVal = {};
    pipes_gda_bnxt_sge sgeVal = {};

    *hdr = hdrVal;
    *rdma = rdmaVal;
    *sge = sgeVal;

    bnxtFillMsn(qp, 0);
    bnxtIncrTail(qp, PIPES_GDA_BNXT_GDA_WQE_SLOT_COUNT);
  }

  __device__ void bnxtUpdateCqDbrec(pipes_gda_gpu_dev_verbs_qp* qp) {
    uint32_t cqConsIdx = static_cast<uint32_t>(qp->cq_sq.cqe_ci);
    uint32_t cqDepth = qp->nic.bnxt.cq_depth;
    if (cqDepth == 0)
      cqDepth = 1;

    uint32_t cycle = (cqConsIdx / cqDepth);
    uint32_t epoch = (cycle & 0x1) << PIPES_GDA_BNXT_DB_EPOCH_TAIL_SHIFT;
    uint32_t cqIdx = cqConsIdx % cqDepth;

    uint64_t keyLo = static_cast<uint64_t>(cqIdx | epoch);
    uint64_t keyHi =
        (static_cast<uint64_t>(qp->cq_sq.cq_num) & PIPES_GDA_BNXT_DB_QID_MASK) |
        ((static_cast<uint64_t>(PIPES_GDA_BNXT_QUE_TYPE_CQ) &
          PIPES_GDA_BNXT_DB_TYP_MASK)
         << PIPES_GDA_BNXT_DB_TYP_SHIFT) |
        (0x1ULL << PIPES_GDA_BNXT_DB_VALID_SHIFT);

    uint64_t dbVal = keyLo | (keyHi << 32);

    __hip_atomic_store(
        const_cast<uint64_t*>(qp->nic.bnxt.dbr),
        dbVal,
        __ATOMIC_SEQ_CST,
        __HIP_MEMORY_SCOPE_SYSTEM);
  }

  // Non-blocking single-CQE poll. Returns 0 on success, EBUSY if not yet
  // ready, -5 on NIC error. Polls `cqe->con_indx` (monotonic counter the
  // NIC writes when a WQE completes) instead of phase bit. After con_indx
  // advances, we read the bcqe status field — silently dropping errors
  // leads to misleading "fast" benchmark numbers because the GPU thinks
  // the WQE completed locally even though the NIC actually failed (no
  // remote ACK). Force-inlined for caller spin loops.
  __device__ __forceinline__ int pollOneCqAt(
      pipes_gda_gpu_dev_verbs_cq* cq,
      uint64_t consIndex) {
    volatile char* cqeBase = reinterpret_cast<volatile char*>(cq->cqe_daddr);
    volatile pipes_gda_bnxt_req_cqe* cqe =
        reinterpret_cast<volatile pipes_gda_bnxt_req_cqe*>(cqeBase);
    uint32_t conIdx = cqe->con_indx & 0xFFFF;
    uint32_t target = static_cast<uint32_t>((consIndex + 1) & 0xFFFF);
    // Signed 16-bit diff handles wraparound up to ~32K WQEs between polls.
    int16_t diff = static_cast<int16_t>(conIdx - target);
    if (diff < 0) {
      return EBUSY;
    }
    // CQE arrived — verify status. bcqe lives right after the req_cqe.
    volatile pipes_gda_bnxt_bcqe* bcqe =
        reinterpret_cast<volatile pipes_gda_bnxt_bcqe*>(
            cqeBase + sizeof(pipes_gda_bnxt_req_cqe));
    uint32_t flg = bcqe->flg_st_typ_ph;
    uint8_t status = (flg >> PIPES_GDA_BNXT_BCQE_STATUS_SHIFT) &
        PIPES_GDA_BNXT_BCQE_STATUS_MASK;
    if (status != PIPES_GDA_BNXT_REQ_ST_OK) {
      printf(
          "BNXT CQE error: status=%u con_indx=%u target=%u\n",
          status,
          conIdx,
          target);
      return -5;
    }
    return 0;
  }

  // Inline RDMA write — payload data lives in the SGE slot rather than
  // being fetched via lkey/addr. BNXT supports this via the INLINE flag
  // and a memcpy of payload into slot 2. Templated to match Mlx5 backend
  // signature; value is copied into slot 2 of the WQE.
  template <typename T>
  __device__ void prepareInlineWriteWqe(
      pipes_gda_gpu_dev_verbs_qp* qp,
      pipes_gda_gpu_dev_verbs_wqe* /* unused */,
      uint64_t wqeIdx,
      uint8_t /* ctrlFlags */,
      uint64_t remoteAddr,
      uint32_t remoteKey,
      T value) {
    const void* localData = static_cast<const void*>(&value);
    std::size_t size = sizeof(T);
    uint32_t slotIdx = bnxtWqeIdxToSlot(qp, wqeIdx);

    pipes_gda_bnxt_bsqe* hdr =
        reinterpret_cast<pipes_gda_bnxt_bsqe*>(getBnxtWqeSlot(qp, slotIdx));
    pipes_gda_bnxt_rdma* rdma =
        reinterpret_cast<pipes_gda_bnxt_rdma*>(getBnxtWqeSlot(qp, slotIdx + 1));
    void* sgeSlot = getBnxtWqeSlot(qp, slotIdx + 2);

    uint32_t wqeType =
        PIPES_GDA_BNXT_HDR_WT_MASK & PIPES_GDA_BNXT_WR_OPCD_RDMA_WRITE;
    uint32_t wqeSize =
        PIPES_GDA_BNXT_HDR_WS_MASK & PIPES_GDA_BNXT_GDA_WQE_SLOT_COUNT;
    uint8_t hdrFlags =
        PIPES_GDA_BNXT_WR_FLAGS_SIGNALED | PIPES_GDA_BNXT_WR_FLAGS_INLINE;

    pipes_gda_bnxt_bsqe hdrVal = {};
    hdrVal.rsv_ws_fl_wt = (wqeSize << PIPES_GDA_BNXT_HDR_WS_SHIFT) |
        (hdrFlags << PIPES_GDA_BNXT_HDR_FLAGS_SHIFT) | wqeType |
        ((static_cast<uint32_t>(size) & PIPES_GDA_BNXT_HDR_IL_MASK)
         << PIPES_GDA_BNXT_HDR_IL_SHIFT);
    hdrVal.key_immd = 0;
    hdrVal.lhdr.qkey_len = static_cast<uint64_t>(size);

    pipes_gda_bnxt_rdma rdmaVal = {};
    rdmaVal.rva = remoteAddr;
    rdmaVal.rkey = remoteKey;

    *hdr = hdrVal;
    *rdma = rdmaVal;
    // Inline payload in slot 2.
    const uint8_t* src = static_cast<const uint8_t*>(localData);
    uint8_t* dst = static_cast<uint8_t*>(sgeSlot);
    for (std::size_t i = 0; i < size && i < PIPES_GDA_BNXT_SLOT_SIZE_BB; ++i) {
      dst[i] = src[i];
    }

    bnxtFillMsn(qp, static_cast<uint32_t>(size));
    bnxtIncrTail(qp, PIPES_GDA_BNXT_GDA_WQE_SLOT_COUNT);
  }

  // Wait-on-counter WQE. BNXT has no native wait-WQE primitive (mlx5
  // synthesizes one via a special opcode). For now post a NOP so the SQ
  // index advances; real wait semantics would need a different mechanism
  // (host-side polling or a custom RDMA-read pattern).
  __device__ void prepareWaitWqe(
      pipes_gda_gpu_dev_verbs_qp* qp,
      pipes_gda_gpu_dev_verbs_wqe* wqe,
      uint64_t wqeIdx,
      uint8_t /* ctrlFlags */,
      uint32_t /* targetCqNum */,
      uint64_t /* targetWqeIdx */) {
    prepareNopWqe(qp, wqe, wqeIdx);
  }

  // Blocking poll for a specific WQE's completion. Spin loop reads
  // `cqe->con_indx`; no fence in the hot path because we don't read other
  // CQE fields on the success path. After completion we advance
  // cq_sq.cqe_ci and ring the CQ doorbell so the NIC can overwrite the
  // (single, ncqe=1) CQE slot for the next completion.
  __device__ __forceinline__ int pollCqAt(
      pipes_gda_gpu_dev_verbs_qp* qp,
      pipes_gda_gpu_dev_verbs_cq* /* unused */,
      uint64_t consIndex) {
    volatile pipes_gda_bnxt_req_cqe* cqe =
        reinterpret_cast<volatile pipes_gda_bnxt_req_cqe*>(qp->nic.bnxt.cq_buf);
    uint32_t target = static_cast<uint32_t>((consIndex + 1) & 0xFFFF);

    constexpr uint64_t kMaxSpins = 10000000ULL;
    for (uint64_t spins = 0; spins < kMaxSpins; ++spins) {
      uint32_t conIdx = cqe->con_indx & 0xFFFF;
      int16_t diff = static_cast<int16_t>(conIdx - target);
      if (diff >= 0) {
        qp->cq_sq.cqe_ci = consIndex + 1;
        bnxtUpdateCqDbrec(qp);
        return 0;
      }
    }
    printf(
        "BNXT pollCqAt TIMEOUT: target=%u con_indx=%u "
        "sq_id=%u sq_tail=%u psn=%u msn=%u\n",
        target,
        cqe->con_indx & 0xFFFF,
        qp->nic.bnxt.sq_id,
        qp->nic.bnxt.sq_tail,
        qp->nic.bnxt.psn,
        qp->nic.bnxt.msn);
    return -5;
  }
};

} // namespace pipes_gda
