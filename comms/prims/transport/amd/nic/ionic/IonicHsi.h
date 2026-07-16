/* SPDX-License-Identifier: GPL-2.0 OR Linux-OpenIB */
/*
 * Copyright (c) 2018-2022 Pensando Systems, Inc.  All rights reserved.
 *               2022-2026 Advanced Micro Devices, Inc.  All rights reserved.
 */

// Modifications: (c) Meta Platforms, Inc. and affiliates.

// =============================================================================
// IONIC (AMD Pensando AI NIC) Hardware-Software Interface for pipes-gda
// =============================================================================
//
// IONIC NIC WQE, CQE, doorbell, and direct-verbs (ionic_dv) structures for
// GPU-initiated RDMA. This is the IONIC equivalent of BnxtHsi.h / Mlx5Hsi.h.
//
// The wire structs (`ionic_sge`, `ionic_v1_*`) are copied verbatim from the
// `ionic_fw.h` firmware ABI header in AMD's public rdma-core ionic provider.
// These layouts, opcodes, and bit positions are fixed by the NIC firmware and
// are reproduced unchanged so the on-wire ABI matches. The direct-verbs structs
// (`ionic_dv_*`) are copied from `ionic_dv.h`; they describe the GDA handles
// the host extracts via `ionic_dv_get_{ctx,qp,cq}()` and are consumed by the
// host setup path (`IonicReDv.h` / `PipesGdaHost.cc`).
//
// Key properties (vs mlx5 / bnxt):
//   - Keys / addresses are big-endian on the wire (like mlx5), but pipes
//     carries them NATIVE through the stack (like bnxt) and byteswaps them at
//     WQE-build time. `base.wqe_idx` is the only little-endian WQE field.
//   - Doorbell: single 64-bit store of `db_val | (sq_mask & prod)`.
//   - CQ: color-bit ownership + 24-bit MSN completion. In CCQE mode
//     (`cq_mask == 0`) a single CQE slot carries the latest completed MSN,
//     analogous to bnxt's compressed (ncqe=1) CQ.
//
// The upstream `ionic_v1_cqe_*` accessors that use `htobe32`/`be32toh` are
// gated `#if !defined(__HIP_PLATFORM_AMD__)` and do NOT compile on device;
// HIP-safe device equivalents are re-authored at the bottom of this file.
// =============================================================================

#pragma once

#include <linux/types.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __HIP_PLATFORM_AMD__
#include "HipDeviceCompat.h" // @manual
#endif

#ifndef BIT
#define BIT(n) (1u << (n))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Common wire structures (all firmware versions)
// =============================================================================

// WQE scatter-gather element (16 bytes, all big-endian).
struct ionic_sge {
  __be64 va;
  __be32 len;
  __be32 lkey;
};

// CQE status codes (reported in `status_length` when the error bit is set).
enum ionic_status {
  IONIC_STS_OK,
  IONIC_STS_LOCAL_LEN_ERR,
  IONIC_STS_LOCAL_QP_OPER_ERR,
  IONIC_STS_LOCAL_PROT_ERR,
  IONIC_STS_WQE_FLUSHED_ERR,
  IONIC_STS_MEM_MGMT_OPER_ERR,
  IONIC_STS_BAD_RESP_ERR,
  IONIC_STS_LOCAL_ACC_ERR,
  IONIC_STS_REMOTE_INV_REQ_ERR,
  IONIC_STS_REMOTE_ACC_ERR,
  IONIC_STS_REMOTE_OPER_ERR,
  IONIC_STS_RETRY_EXCEEDED,
  IONIC_STS_RNR_RETRY_EXCEEDED,
  IONIC_STS_XRC_VIO_ERR,
};

// =============================================================================
// Firmware ABI v1 WQE / CQE
// =============================================================================

// Data payload portion of a v1 WQE (32 bytes).
union ionic_v1_pld {
  struct ionic_sge sgl[2];
  __be32 spec32[8];
  __be16 spec16[16];
  __u8 data[32];
};

// v1 base WQE header (16 bytes). NOTE the mixed endianness: `wqe_idx` is
// little-endian (written raw); `flags` / `imm_data_key` are big-endian.
struct ionic_v1_base_hdr {
  __le64 wqe_idx;
  __u8 op;
  __u8 num_sge_key;
  __be16 flags;
  __be32 imm_data_key;
};

// v1 receive WQE body.
struct ionic_v1_recv_bdy {
  __u8 rsvd[16];
  union ionic_v1_pld pld;
};

// v1 send / RDMA WQE body (common; carries the SGL / inline payload).
struct ionic_v1_common_bdy {
  union {
    struct {
      __be32 ah_id;
      __be32 dest_qpn;
      __be32 dest_qkey;
    } send;
    struct {
      __be32 remote_va_high;
      __be32 remote_va_low;
      __be32 remote_rkey;
    } rdma;
  };
  __be32 length;
  union ionic_v1_pld pld;
};

// v1 atomic WQE body.
struct ionic_v1_atomic_bdy {
  __be32 remote_va_high;
  __be32 remote_va_low;
  __be32 remote_rkey;
  __be32 swap_add_high;
  __be32 swap_add_low;
  __be32 compare_high;
  __be32 compare_low;
  __u8 rsvd[4];
  struct ionic_sge sge;
};

// v2 atomic WQE body (the deployed AI NIC HW is fw abi v2).
struct ionic_v2_atomic_bdy {
  __be32 remote_va_high;
  __be32 remote_va_low;
  __be32 remote_rkey;
  __be32 swap_add_high;
  __be32 swap_add_low;
  __be32 compare_high;
  __be32 compare_low;
  __be32 lkey;
  __be64 local_va;
  __u8 rsvd_expdb[8];
};

// v1 bind-mw WQE body.
struct ionic_v1_bind_mw_bdy {
  __be64 va;
  __be64 length;
  __be32 lkey;
  __be16 flags;
  __u8 rsvd[26];
};

// v1 send/recv WQE. The send-queue stride is provider-supplied; for the GDA
// configuration pipes uses, the stride equals sizeof(struct ionic_v1_wqe)
// (64 bytes); the host setup asserts the provider's stride_log2 matches.
struct ionic_v1_wqe {
  struct ionic_v1_base_hdr base;
  union {
    struct ionic_v1_recv_bdy recv;
    struct ionic_v1_common_bdy common;
    struct ionic_v1_atomic_bdy atomic;
    struct ionic_v2_atomic_bdy atomic_v2;
    struct ionic_v1_bind_mw_bdy bind_mw;
  };
};

// v1 send completion (the GDA fast path reads `msg_msn`).
struct ionic_v1_cqe_send {
  __u8 rsvd[4];
  __be32 msg_msn;
  __u8 rsvd2[8];
  __le64 npg_wqe_idx_timestamp;
};

struct ionic_v1_cqe_recv {
  __le64 wqe_idx_timestamp;
  __be32 src_qpn_op;
  __u8 src_mac[6];
  __be16 vlan_tag;
  __be32 imm_data_rkey;
};

struct ionic_v1_cqe_rcqe {
  __be64 wqe_idx_timestamp;
  __u8 rsvd[8];
  __be32 seq_op_flags;
  __be32 imm_data_rkey;
};

// v1 completion queue entry (32 bytes).
struct ionic_v1_cqe {
  union {
    struct ionic_v1_cqe_send send;
    struct ionic_v1_cqe_recv recv;
    struct ionic_v1_cqe_rcqe rcqe;
  };
  __be32 status_length;
  __be32 qid_type_flags;
};

// =============================================================================
// Opcodes / flags / CQE bits
// =============================================================================

// v1 send opcodes and WQE flags (the flag bits live in `base.flags`).
enum ionic_v1_op {
  IONIC_V1_OP_SEND,
  IONIC_V1_OP_SEND_INV,
  IONIC_V1_OP_SEND_IMM,
  IONIC_V1_OP_RDMA_READ,
  IONIC_V1_OP_RDMA_WRITE,
  IONIC_V1_OP_RDMA_WRITE_IMM,
  IONIC_V1_OP_ATOMIC_CS,
  IONIC_V1_OP_ATOMIC_FA,
  IONIC_V1_OP_REG_MR,
  IONIC_V1_OP_LOCAL_INV,
  IONIC_V1_OP_BIND_MW,

  /* flags (set in base.flags, big-endian on the wire) */
  IONIC_V1_FLAG_FENCE = BIT(0),
  IONIC_V1_FLAG_SOL = BIT(1),
  IONIC_V1_FLAG_INL = BIT(2),
  IONIC_V1_FLAG_SIG = BIT(3),
  IONIC_V1_FLAG_COLOR = BIT(4),

  /* spec-sgl format (last four bits) */
  IONIC_V1_FLAG_SPEC32 = (1u << 12),
  IONIC_V1_FLAG_SPEC16 = (2u << 12),
  IONIC_V1_SPEC_FIRST_SGE = 2,
};

// v2 send opcodes (bit-encoded). The GDA path emits these.
enum ionic_v2_op {
  IONIC_V2_OPSL_OUT = 0x20,
  IONIC_V2_OPSL_IMM = 0x40,
  IONIC_V2_OPSL_INV = 0x80,

  IONIC_V2_OP_SEND = 0x0 | IONIC_V2_OPSL_OUT,
  IONIC_V2_OP_SEND_IMM = IONIC_V2_OP_SEND | IONIC_V2_OPSL_IMM,
  IONIC_V2_OP_SEND_INV = IONIC_V2_OP_SEND | IONIC_V2_OPSL_INV,

  IONIC_V2_OP_RDMA_WRITE = 0x1 | IONIC_V2_OPSL_OUT,
  IONIC_V2_OP_RDMA_WRITE_IMM = IONIC_V2_OP_RDMA_WRITE | IONIC_V2_OPSL_IMM,

  IONIC_V2_OP_RDMA_READ = 0x2,

  IONIC_V2_OP_ATOMIC_CS = 0x4,
  IONIC_V2_OP_ATOMIC_FA = 0x5,
  IONIC_V2_OP_REG_MR = 0x6,
  IONIC_V2_OP_LOCAL_INV = 0x7,
  IONIC_V2_OP_BIND_MW = 0x8,
};

// CQE `qid_type_flags` bit layout (the value is big-endian on the wire).
enum ionic_v1_cqe_qtf_bits {
  IONIC_V1_CQE_COLOR = BIT(0),
  IONIC_V1_CQE_ERROR = BIT(1),
  IONIC_V1_CQE_TYPE_SHIFT = 5,
  IONIC_V1_CQE_TYPE_MASK = 0x7,
  IONIC_V1_CQE_QID_SHIFT = 8,

  IONIC_V1_CQE_TYPE_RECV = 1,
  IONIC_V1_CQE_TYPE_SEND_MSN = 2,
  IONIC_V1_CQE_TYPE_SEND_NPG = 3,
  IONIC_V1_CQE_TYPE_RECV_RCQE = 4,
};

enum ionic_v1_cqe_wqe_idx_timestamp_bits {
  IONIC_V1_CQE_WQE_IDX_MASK = 0xffff,
  IONIC_V1_CQE_TIMESTAMP_SHIFT = 16,
};

// MSN is a 24-bit wrapping sequence number; 0x800000 is its sign bit.
#define IONIC_MSN_MASK 0xFFFFFFu
#define IONIC_MSN_SIGN_BIT 0x800000u

// Doorbell value layout (from ionic_queue.h). The per-queue `db_val` returned
// by ionic_dv already encodes (qid << IONIC_DBELL_QID_SHIFT); the device ORs in
// the (masked) producer/consumer index.
#define IONIC_DBELL_QID_MASK ((1u << 24) - 1)
#define IONIC_DBELL_QID_SHIFT 24

// Host-order CQE field accessors (take a value already converted to host byte
// order). Safe on host and device.
static inline uint8_t ionic_v1_cqe_qtf_type(uint32_t qtf) {
  return (qtf >> IONIC_V1_CQE_TYPE_SHIFT) & IONIC_V1_CQE_TYPE_MASK;
}

static inline uint32_t ionic_v1_cqe_qtf_qid(uint32_t qtf) {
  return qtf >> IONIC_V1_CQE_QID_SHIFT;
}

static inline int ionic_to_ibv_status(int sts) {
  switch (sts) {
    case IONIC_STS_OK:
      return 0; // IBV_WC_SUCCESS
    case IONIC_STS_LOCAL_LEN_ERR:
      return 1; // IBV_WC_LOC_LEN_ERR
    case IONIC_STS_LOCAL_QP_OPER_ERR:
      return 2; // IBV_WC_LOC_QP_OP_ERR
    case IONIC_STS_LOCAL_PROT_ERR:
      return 4; // IBV_WC_LOC_PROT_ERR
    case IONIC_STS_WQE_FLUSHED_ERR:
      return 5; // IBV_WC_WR_FLUSH_ERR
    default:
      return 13; // IBV_WC_GENERAL_ERR (and other remote/transport errors)
  }
}

// =============================================================================
// ionic_dv direct-verbs handles (copied from ionic_dv.h)
// =============================================================================
//
// These are populated on the host by `ionic_dv_get_{ctx,qp,cq}()` and feed the
// device QP descriptor. Kept here so both the host wrapper (IonicReDv.h) and
// the device backend share one definition.

struct ionic_dv_ctx {
  void* db_page;
  uint64_t* db_ptr;
  uint8_t sq_qtype;
  uint8_t rq_qtype;
  uint8_t cq_qtype;
};

struct ionic_dv_queue {
  void* ptr;
  size_t size;
  uint64_t db_val;
  uint16_t mask;
  uint8_t depth_log2;
  uint8_t stride_log2;
};

struct ionic_dv_cq {
  struct ionic_dv_queue q;
};

struct ionic_dv_qp {
  struct ionic_dv_queue rq;
  struct ionic_dv_queue sq;
};

enum ionic_cq_init_attr_mask {
  IONIC_CQ_INIT_ATTR_MASK_FLAGS = 1 << 0,
};

enum ionic_cq_init_attr_flags {
  IONIC_CQ_INIT_ATTR_CCQE = 1 << 0,
};

struct ionic_cq_init_attr_ex {
  uint32_t comp_mask;
  uint32_t flags;
};

#ifdef __cplusplus
}
#endif

// =============================================================================
// HIP device-safe CQE accessors
// =============================================================================
//
// The upstream `ionic_v1_cqe_color/error` helpers use `htobe32`, which is not
// available in device code; these re-author them with `__builtin_bswap32`. The
// CQE `qid_type_flags` is big-endian on the wire, so masks are byteswapped
// before comparison (a constant fold at compile time).

#if defined(__HIP_PLATFORM_AMD__) && defined(__cplusplus)

__device__ __forceinline__ uint32_t
pipes_gda_ionic_cqe_qtf_raw(const struct ionic_v1_cqe* cqe) {
  // Coherent system-scope read so the GPU observes the NIC's latest write.
  return amd_load_relaxed_sys(
      const_cast<uint32_t*>(
          reinterpret_cast<const uint32_t*>(&cqe->qid_type_flags)));
}

__device__ __forceinline__ bool pipes_gda_ionic_cqe_color(
    const struct ionic_v1_cqe* cqe) {
  return (pipes_gda_ionic_cqe_qtf_raw(cqe) &
          __builtin_bswap32(IONIC_V1_CQE_COLOR)) != 0;
}

__device__ __forceinline__ bool pipes_gda_ionic_cqe_error(
    const struct ionic_v1_cqe* cqe) {
  return (pipes_gda_ionic_cqe_qtf_raw(cqe) &
          __builtin_bswap32(IONIC_V1_CQE_ERROR)) != 0;
}

// Latest completed 24-bit MSN reported by a send CQE (converted to host order).
__device__ __forceinline__ uint32_t
pipes_gda_ionic_cqe_msn(const struct ionic_v1_cqe* cqe) {
  uint32_t be = amd_load_relaxed_sys(
      const_cast<uint32_t*>(
          reinterpret_cast<const uint32_t*>(&cqe->send.msg_msn)));
  return __builtin_bswap32(be) & IONIC_MSN_MASK;
}

#endif // __HIP_PLATFORM_AMD__ && __cplusplus
