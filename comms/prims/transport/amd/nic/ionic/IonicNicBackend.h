// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// IONIC (AMD Pensando AI NIC) NIC Backend for pipes-gda
// =============================================================================
//
// Device-side WQE construction, doorbell, and CQ polling for the AMD Pensando
// "ionic" AI NIC. Implements the same method contract as BnxtNicBackend /
// Mlx5NicBackend so PipesGdaOps.h can drive it through `ActiveNicBackend`.
//
// Implements the AMD Pensando ionic GDA send/doorbell protocol, driven by the
// hardware WQE layout (nic/ionic/IonicHsi.h) and the ionic doorbell/CQ register
// semantics mandated by the NIC.
//
// Key differences from mlx5 / bnxt:
//   - WQE: ionic_v1_wqe (64-byte stride). The COLOR flag (BIT 4) toggles per
//     SQ wrap and is the WQE "valid" marker; it MUST be written LAST with a
//     release store so the NIC never observes a half-built WQE.
//   - Keys / addresses are big-endian on the wire but carried NATIVE through
//     pipes (matching IbgdaBuffer.h's bnxt/ionic grouping); this backend
//     byteswaps them at WQE-build time. `base.wqe_idx` is little-endian.
//   - Doorbell: single 64-bit store of `db_val | (sq_mask & prod)`. The
//     producer index must never move backwards, so a lock-free fetch-max
//     high-water mark guards it: only the thread that advances the mark writes
//     the doorbell (see ringDoorbell).
//   - CQ: 24-bit MSN completion. The host configures a compressed (CCQE)
//     completion queue — a single CQE slot whose `msg_msn` reports the latest
//     completed message, analogous to bnxt's ncqe=1 CQ. One completion can
//     retire many WQEs.
//
// This backend is used by P2pIbgdaTransportDevice<IonicNicBackend>.
// =============================================================================

#pragma once

#include <cerrno> // EBUSY for non-blocking pollOneCqAt
#include <cstddef>
#include <cstdint>

#include "HipDeviceCompat.h" // @manual
#include "nic/ionic/IonicHsi.h" // @manual
#include "pipes_gda/PipesGdaDev.h" // @manual

namespace pipes_gda {

// The GDA SQ uses a 64-byte WQE stride (== sizeof(ionic_v1_wqe)); the host
// setup asserts the provider's reported stride_log2 matches this.
static_assert(
    sizeof(ionic_v1_wqe) == 64,
    "ionic_v1_wqe must be 64 bytes for the assumed SQ stride");

struct IonicNicBackend {
  static constexpr const char* vendorPrefix() {
    return "ionic";
  }
  static constexpr uint16_t vendorId() {
    return 0x1DD8; // AMD Pensando
  }
  // Keys are carried in native byte order through pipes (see IbgdaBuffer.h);
  // this backend byteswaps them when writing the (big-endian) WQE fields.
  static inline uint32_t swapMkey(uint32_t key) {
    return key;
  }
  static inline uint32_t networkByteOrderKey(uint32_t hostKey) {
    return hostKey;
  }

  // ---------------------------------------------------------------------------
  // SQ ring helpers
  // ---------------------------------------------------------------------------

  __device__ __forceinline__ ionic_v1_wqe* ionicWqe(
      pipes_gda_gpu_dev_verbs_qp* qp,
      uint64_t wqeIdx) const {
    auto* base = reinterpret_cast<ionic_v1_wqe*>(qp->nic.ionic.sq_buf);
    uint32_t slot = static_cast<uint32_t>(wqeIdx & qp->nic.ionic.sq_mask);
    return &base[slot];
  }

  __device__ pipes_gda_gpu_dev_verbs_wqe* getWqePtr(
      pipes_gda_gpu_dev_verbs_qp* qp,
      uint64_t wqeIdx) const {
    return reinterpret_cast<pipes_gda_gpu_dev_verbs_wqe*>(ionicWqe(qp, wqeIdx));
  }

  __device__ uint64_t
  reserveWqSlots(pipes_gda_gpu_dev_verbs_qp* qp, uint32_t numWqes) {
    return amd_atomic_add_device(
        &qp->sq_rsvd_index, static_cast<uint64_t>(numWqes));
  }

  // Ordered hand-off: wait until all earlier WQEs are ready, publish a release
  // fence, then advance the ready watermark. Combined with the monotonic
  // doorbell below this keeps producer doorbells in non-decreasing order
  // across concurrent putters.
  __device__ void markWqesReady(
      pipes_gda_gpu_dev_verbs_qp* qp,
      uint64_t firstIdx,
      uint64_t lastIdx) {
    while (amd_load_relaxed_device(&qp->sq_ready_index) < firstIdx) {
    }
    amd_fence_release_device();
    amd_atomic_max_device(&qp->sq_ready_index, lastIdx + 1);
  }

  // ---------------------------------------------------------------------------
  // WQE flag helper (host-order; byteswapped once at the publish store)
  // ---------------------------------------------------------------------------

  // COLOR is set on even wrap cycles of the SQ (the NIC's slot-valid marker).
  __device__ __forceinline__ uint16_t
  baseColorFlag(pipes_gda_gpu_dev_verbs_qp* qp, uint64_t wqeIdx) const {
    uint64_t wrapBit = static_cast<uint64_t>(qp->nic.ionic.sq_mask) + 1;
    return (wqeIdx & wrapBit) ? 0 : static_cast<uint16_t>(IONIC_V1_FLAG_COLOR);
  }

  // Publish the WQE: write the (big-endian) flags last with a release store so
  // the NIC never sees a WQE whose COLOR flipped before its body was written.
  __device__ __forceinline__ void publishWqe(ionic_v1_wqe* w, uint16_t flags) {
    __hip_atomic_store(
        &w->base.flags,
        __builtin_bswap16(flags),
        __ATOMIC_RELEASE,
        __HIP_MEMORY_SCOPE_AGENT);
  }

  // ---------------------------------------------------------------------------
  // WQE builders
  // ---------------------------------------------------------------------------

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
    ionic_v1_wqe* w = ionicWqe(qp, wqeIdx);

    uint16_t flags = baseColorFlag(qp, wqeIdx);
    if (ctrlFlags & PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE) {
      flags |= IONIC_V1_FLAG_SIG;
    }
    if (ctrlFlags & PIPES_GDA_IB_MLX5_WQE_CTRL_FENCE) {
      flags |= IONIC_V1_FLAG_FENCE;
    }

    // Inline payloads up to 32B (one ionic_v1_pld) avoid an SGE fetch.
    bool useInline = (size > 0 && size <= sizeof(w->common.pld.data));

    w->base.wqe_idx = wqeIdx; // little-endian, written raw
    w->base.op = IONIC_V2_OP_RDMA_WRITE;
    w->base.num_sge_key = (size && !useInline) ? 1 : 0;
    w->base.imm_data_key = 0;

    w->common.rdma.remote_va_high =
        __builtin_bswap32(static_cast<uint32_t>(remoteAddr >> 32));
    w->common.rdma.remote_va_low =
        __builtin_bswap32(static_cast<uint32_t>(remoteAddr));
    w->common.rdma.remote_rkey = __builtin_bswap32(remoteKey);
    w->common.length = __builtin_bswap32(static_cast<uint32_t>(size));

    if (useInline) {
      flags |= IONIC_V1_FLAG_INL;
      const uint8_t* src = reinterpret_cast<const uint8_t*>(localAddr);
      for (std::size_t i = 0; i < size; ++i) {
        w->common.pld.data[i] = src[i];
      }
      (void)localKey;
    } else if (size) {
      w->common.pld.sgl[0].va = __builtin_bswap64(localAddr);
      w->common.pld.sgl[0].len = __builtin_bswap32(static_cast<uint32_t>(size));
      w->common.pld.sgl[0].lkey = __builtin_bswap32(localKey);
    }

    publishWqe(w, flags);
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
    ionic_v1_wqe* w = ionicWqe(qp, wqeIdx);

    uint16_t flags = baseColorFlag(qp, wqeIdx);
    if (ctrlFlags & PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE) {
      flags |= IONIC_V1_FLAG_SIG;
    }
    // Honor an explicit FENCE: the put-with-counter pattern uses it to order
    // the atomic after a preceding large WRITE.
    if (ctrlFlags & PIPES_GDA_IB_MLX5_WQE_CTRL_FENCE) {
      flags |= IONIC_V1_FLAG_FENCE;
    }

    w->base.wqe_idx = wqeIdx;
    w->base.op = IONIC_V2_OP_ATOMIC_FA;
    w->base.num_sge_key = 1;
    w->base.imm_data_key = 0;

    w->atomic_v2.remote_va_high =
        __builtin_bswap32(static_cast<uint32_t>(remoteAddr >> 32));
    w->atomic_v2.remote_va_low =
        __builtin_bswap32(static_cast<uint32_t>(remoteAddr));
    w->atomic_v2.remote_rkey = __builtin_bswap32(remoteKey);
    w->atomic_v2.swap_add_high =
        __builtin_bswap32(static_cast<uint32_t>(addVal >> 32));
    w->atomic_v2.swap_add_low =
        __builtin_bswap32(static_cast<uint32_t>(addVal));
    w->atomic_v2.compare_high = 0;
    w->atomic_v2.compare_low = 0;
    w->atomic_v2.lkey = __builtin_bswap32(localKey);
    w->atomic_v2.local_va = __builtin_bswap64(localAddr);

    publishWqe(w, flags);
  }

  // ionic has no NOP opcode; pipes_gda_fence posts a signaled zero-length
  // RDMA-WRITE as a drain marker (completes in order after all prior WQEs).
  __device__ void prepareNopWqe(
      pipes_gda_gpu_dev_verbs_qp* qp,
      pipes_gda_gpu_dev_verbs_wqe* /* unused */,
      uint64_t wqeIdx) {
    ionic_v1_wqe* w = ionicWqe(qp, wqeIdx);

    uint16_t flags = baseColorFlag(qp, wqeIdx) | IONIC_V1_FLAG_SIG;

    w->base.wqe_idx = wqeIdx;
    w->base.op = IONIC_V2_OP_RDMA_WRITE;
    w->base.num_sge_key = 0;
    w->base.imm_data_key = 0;
    w->common.rdma.remote_va_high = 0;
    w->common.rdma.remote_va_low = 0;
    w->common.rdma.remote_rkey = 0;
    w->common.length = 0;

    publishWqe(w, flags);
  }

  // Inline RDMA write of a small scalar (data embedded in the WQE; no MR).
  template <typename T>
  __device__ void prepareInlineWriteWqe(
      pipes_gda_gpu_dev_verbs_qp* qp,
      pipes_gda_gpu_dev_verbs_wqe* /* unused */,
      uint64_t wqeIdx,
      uint8_t ctrlFlags,
      uint64_t remoteAddr,
      uint32_t remoteKey,
      T value) {
    ionic_v1_wqe* w = ionicWqe(qp, wqeIdx);
    constexpr std::size_t kSize = sizeof(T);
    static_assert(
        kSize <= 32, "prepareInlineWriteWqe: payload exceeds inline capacity");

    uint16_t flags = baseColorFlag(qp, wqeIdx) | IONIC_V1_FLAG_INL;
    if (ctrlFlags & PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE) {
      flags |= IONIC_V1_FLAG_SIG;
    }

    w->base.wqe_idx = wqeIdx;
    w->base.op = IONIC_V2_OP_RDMA_WRITE;
    w->base.num_sge_key = 0;
    w->base.imm_data_key = 0;
    w->common.rdma.remote_va_high =
        __builtin_bswap32(static_cast<uint32_t>(remoteAddr >> 32));
    w->common.rdma.remote_va_low =
        __builtin_bswap32(static_cast<uint32_t>(remoteAddr));
    w->common.rdma.remote_rkey = __builtin_bswap32(remoteKey);
    w->common.length = __builtin_bswap32(static_cast<uint32_t>(kSize));

    const uint8_t* src = reinterpret_cast<const uint8_t*>(&value);
    for (std::size_t i = 0; i < kSize; ++i) {
      w->common.pld.data[i] = src[i];
    }

    publishWqe(w, flags);
  }

  // ionic has no WAIT primitive; the AMD path uses CQ-poll + GPU atomic instead
  // of companion-QP WAIT, so this is never invoked under __HIP_PLATFORM_AMD__.
  // Defined (forwarding to a NOP) to satisfy the backend contract.
  __device__ void prepareWaitWqe(
      pipes_gda_gpu_dev_verbs_qp* qp,
      pipes_gda_gpu_dev_verbs_wqe* wqe,
      uint64_t wqeIdx,
      uint8_t /* ctrlFlags */,
      uint32_t /* targetCqNum */,
      uint64_t /* targetWqeIdx */) {
    prepareNopWqe(qp, wqe, wqeIdx);
  }

  // ---------------------------------------------------------------------------
  // Doorbell
  // ---------------------------------------------------------------------------

  __device__ void ringDoorbell(
      pipes_gda_gpu_dev_verbs_qp* qp,
      uint64_t nextWqeIdx) {
    auto& io = qp->nic.ionic;

    // All WQE bytes (and the release-stored COLOR flags) must be globally
    // visible to the NIC before the doorbell write crosses PCIe.
    amd_fence_system();

    // Lock-free doorbell. sq_dbprod is a monotonic producer high-water mark
    // (the same fetch-max pattern markWqesReady uses for sq_ready_index); a
    // thread that does not advance it leaves the ring to the thread that did.
    // Two advancing putters' MMIO doorbell stores can still reorder, and a
    // smaller producer index landing last would strand the higher WQEs until
    // the next put — so the ringer re-reads sq_dbprod after each store and
    // rings again if a concurrent putter advanced it. The last value the NIC
    // observes therefore equals the current high-water mark. This avoids a
    // busy-wait lock (which can livelock across GPU waves); the loop is bounded
    // by real producer progress, not by lock contention.
    if (amd_atomic_max_device(&io.sq_dbprod, nextWqeIdx) >= nextWqeIdx) {
      return; // a concurrent putter is already at/ahead of us and will ring
    }
    uint64_t rung = nextWqeIdx;
    for (;;) {
      __hip_atomic_store(
          const_cast<uint64_t*>(io.sq_dbreg),
          io.sq_dbval | (static_cast<uint64_t>(io.sq_mask) & rung),
          __ATOMIC_SEQ_CST,
          __HIP_MEMORY_SCOPE_SYSTEM);
      const uint64_t cur = amd_load_relaxed_device(&io.sq_dbprod);
      if (cur == rung) {
        break; // our doorbell reflects the current high-water mark
      }
      rung = cur; // a concurrent putter advanced it; ring the newer value
    }
  }

  // ---------------------------------------------------------------------------
  // Completion (24-bit MSN; CCQE single-slot CQ)
  // ---------------------------------------------------------------------------

  // Non-blocking poll. Returns 0 if the WQE at `consIndex` has completed,
  // EBUSY if not yet, -5 on NIC error. Idempotent: no state mutated, so repeat
  // calls during a spin are safe. The compressed CQ exposes the latest
  // completed 24-bit MSN in a single CQE slot (cq->cqe_daddr).
  __device__ __forceinline__ int pollOneCqAt(
      pipes_gda_gpu_dev_verbs_cq* cq,
      uint64_t consIndex) {
    const ionic_v1_cqe* cqe =
        reinterpret_cast<const ionic_v1_cqe*>(cq->cqe_daddr);

    if (pipes_gda_ionic_cqe_error(cqe)) {
      uint32_t st = __builtin_bswap32(amd_load_relaxed_sys(
          const_cast<uint32_t*>(
              reinterpret_cast<const uint32_t*>(&cqe->status_length))));
      printf(
          "IONIC CQE error: status=%u consIndex=%llu\n",
          st,
          static_cast<unsigned long long>(consIndex));
      return -5;
    }

    uint32_t msn = pipes_gda_ionic_cqe_msn(cqe);
    uint32_t target = static_cast<uint32_t>(consIndex + 1) & IONIC_MSN_MASK;
    // 24-bit wrapping compare: sign bit set => msn is behind target.
    if ((msn - target) & IONIC_MSN_SIGN_BIT) {
      return EBUSY;
    }
    return 0;
  }

  // Blocking poll for the WQE at `consIndex`. Advances the local consumer and,
  // for a non-compressed CQ, rings the CQ doorbell periodically to free space.
  __device__ __forceinline__ int pollCqAt(
      pipes_gda_gpu_dev_verbs_qp* qp,
      pipes_gda_gpu_dev_verbs_cq* cq,
      uint64_t consIndex) {
    constexpr uint64_t kMaxSpins = 10000000ULL;
    for (uint64_t spins = 0; spins < kMaxSpins; ++spins) {
      int rc = pollOneCqAt(cq, consIndex);
      if (rc == EBUSY) {
        continue;
      }
      if (rc != 0) {
        return rc;
      }
      qp->cq_sq.cqe_ci = consIndex + 1;

      // Only the compressed (CCQE, cq_mask == 0) CQ is wired up today: it is a
      // single persistent slot that needs no consumer doorbell, so nothing
      // below runs. The non-compressed branch updates cq_pos/cq_dbpos and rings
      // the CQ doorbell WITHOUT serialization, so it is NOT safe for concurrent
      // pollers of the same CQ — before enabling a non-compressed CQ it must be
      // serialized (e.g. the lock-free fetch-max the SQ doorbell uses).
      auto& io = qp->nic.ionic;
      if (io.cq_mask) {
        // Non-compressed CQ only: release CQ space every 100 entries.
        io.cq_pos = static_cast<uint32_t>(consIndex + 1);
        if ((io.cq_pos - io.cq_dbpos) >= 100) {
          io.cq_dbpos = io.cq_pos;
          __hip_atomic_store(
              const_cast<uint64_t*>(io.cq_dbreg),
              io.cq_dbval | (static_cast<uint64_t>(io.cq_mask) & io.cq_dbpos),
              __ATOMIC_SEQ_CST,
              __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
      return 0;
    }
    printf(
        "IONIC pollCqAt TIMEOUT: target=%u\n",
        static_cast<uint32_t>(consIndex + 1));
    return -5;
  }
};

} // namespace pipes_gda
