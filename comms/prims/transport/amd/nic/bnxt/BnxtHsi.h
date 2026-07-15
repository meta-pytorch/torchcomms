/*
 * Copyright (c) 2025, Broadcom. All rights reserved.  The term
 * Broadcom refers to Broadcom Limited and/or its subsidiaries.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Description: Fast path definitions for bnxt_re
 */

// Modifications: (c) Meta Platforms, Inc. and affiliates.

// =============================================================================
// BNXT (Broadcom) Hardware-Software Interface for pipes-gda
// =============================================================================
//
// BNXT NIC WQE, CQE, and doorbell structures for GPU-initiated RDMA.
//
// Key differences from mlx5:
//   - WQE: 3 x 16-byte slots (48 bytes) vs mlx5's 4 x 16-byte segments (64
//   bytes)
//   - Doorbell: single 64-bit atomic write with epoch bit (no DBREC +
//   BlueFlame)
//   - CQ: CQE compression (depth=1), consumer index tracks all completions
//   - Keys: native byte order (no big-endian swap needed)
//   - MSN table: at end of SQ buffer, tracks PSN per WQE
// =============================================================================

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// BNXT WQE Slot / Size Constants
// =============================================================================

#define PIPES_GDA_BNXT_SLOT_SIZE_BB 16
#define PIPES_GDA_BNXT_GDA_WQE_SLOT_COUNT 3
#define PIPES_GDA_BNXT_GDA_WQE_SIZE \
  (PIPES_GDA_BNXT_GDA_WQE_SLOT_COUNT * PIPES_GDA_BNXT_SLOT_SIZE_BB) // 48 bytes

#define PIPES_GDA_BNXT_STATIC_WQE_SIZE_SLOTS 4
#define PIPES_GDA_BNXT_STATIC_WQE_BB      \
  (PIPES_GDA_BNXT_STATIC_WQE_SIZE_SLOTS * \
   PIPES_GDA_BNXT_SLOT_SIZE_BB) // 64 bytes
#define PIPES_GDA_BNXT_STATIC_WQE_SHIFT 6

#define PIPES_GDA_BNXT_STATIC_CQE_SIZE_SLOTS 4
#define PIPES_GDA_BNXT_STATIC_CQE_BB      \
  (PIPES_GDA_BNXT_STATIC_CQE_SIZE_SLOTS * \
   PIPES_GDA_BNXT_SLOT_SIZE_BB) // 64 bytes

// =============================================================================
// BNXT WQE Opcodes
// =============================================================================

enum pipes_gda_bnxt_wr_opcode {
  PIPES_GDA_BNXT_WR_OPCD_SEND = 0x00,
  PIPES_GDA_BNXT_WR_OPCD_SEND_IMM = 0x01,
  PIPES_GDA_BNXT_WR_OPCD_SEND_INVAL = 0x02,
  PIPES_GDA_BNXT_WR_OPCD_RDMA_WRITE = 0x04,
  PIPES_GDA_BNXT_WR_OPCD_RDMA_WRITE_IMM = 0x05,
  PIPES_GDA_BNXT_WR_OPCD_RDMA_READ = 0x06,
  PIPES_GDA_BNXT_WR_OPCD_ATOMIC_CS = 0x08,
  PIPES_GDA_BNXT_WR_OPCD_ATOMIC_FA = 0x0B,
  PIPES_GDA_BNXT_WR_OPCD_LOC_INVAL = 0x0C,
  PIPES_GDA_BNXT_WR_OPCD_BIND = 0x0E,
  PIPES_GDA_BNXT_WR_OPCD_FR_PPMR = 0x0F,
  PIPES_GDA_BNXT_WR_OPCD_RECV = 0x80,
};

// =============================================================================
// BNXT WQE Flags
// =============================================================================

enum pipes_gda_bnxt_wr_flags {
  PIPES_GDA_BNXT_WR_FLAGS_DBG_TRACE = 0x40,
  PIPES_GDA_BNXT_WR_FLAGS_TS_EN = 0x20,
  PIPES_GDA_BNXT_WR_FLAGS_INLINE = 0x10,
  PIPES_GDA_BNXT_WR_FLAGS_SE = 0x08,
  PIPES_GDA_BNXT_WR_FLAGS_UC_FENCE = 0x04,
  PIPES_GDA_BNXT_WR_FLAGS_RD_FENCE = 0x02,
  PIPES_GDA_BNXT_WR_FLAGS_SIGNALED = 0x01,
};

// =============================================================================
// BNXT WQE Header Bit Encoding
// =============================================================================

enum pipes_gda_bnxt_hdr_offset {
  PIPES_GDA_BNXT_HDR_WT_MASK = 0xFF,
  PIPES_GDA_BNXT_HDR_FLAGS_MASK = 0xFF,
  PIPES_GDA_BNXT_HDR_FLAGS_SHIFT = 0x08,
  PIPES_GDA_BNXT_HDR_WS_MASK = 0xFF,
  PIPES_GDA_BNXT_HDR_WS_SHIFT = 0x10,
  PIPES_GDA_BNXT_HDR_ZB_SHIFT = 0x16,
  PIPES_GDA_BNXT_HDR_MW_SHIFT = 0x17,
  PIPES_GDA_BNXT_HDR_ACC_SHIFT = 0x18,
  PIPES_GDA_BNXT_HDR_IL_MASK = 0x0F,
  PIPES_GDA_BNXT_HDR_IL_SHIFT = 0x18,
};

// =============================================================================
// BNXT WQE Structures (3 x 16-byte slots)
// =============================================================================

// Lower sub-header union in WQE header
union pipes_gda_bnxt_lower_shdr {
  uint64_t qkey_len; // For send: qkey + length
  uint64_t lkey_plkey; // For bind
  uint64_t rva; // For atomic: remote virtual address
};

// Slot 0: Base SQ WQE header (16 bytes)
struct pipes_gda_bnxt_bsqe {
  uint32_t rsv_ws_fl_wt; // WQE type [7:0], flags [15:8], WQE size [23:16]
  uint32_t key_immd; // R-Key or immediate data
  union pipes_gda_bnxt_lower_shdr lhdr;
} __attribute__((packed));

// Slot 1 (for RDMA): RDMA segment (16 bytes)
struct pipes_gda_bnxt_rdma {
  uint64_t rva; // Remote Virtual Address
  uint32_t rkey; // Remote Key (native byte order)
  uint32_t bytes; // Reserved / timestamp for V3
} __attribute__((packed));

// Slot 1 (for atomics): Atomic segment (16 bytes)
struct pipes_gda_bnxt_atomic {
  uint64_t swp_dt; // Swap/add data
  uint64_t cmp_dt; // Compare data (for CAS)
} __attribute__((packed));

// Slot 2: SG entry (16 bytes)
struct pipes_gda_bnxt_sge {
  uint64_t pa; // Physical/Virtual Address
  uint32_t lkey; // Local Key (native byte order)
  uint32_t length; // Data Length
} __attribute__((packed));

// Complete WQE for GDA (48 bytes = 3 slots)
struct pipes_gda_bnxt_gda_wqe {
  struct pipes_gda_bnxt_bsqe hdr; // Slot 0: header
  union {
    struct pipes_gda_bnxt_rdma rdma; // Slot 1: RDMA segment
    struct pipes_gda_bnxt_atomic atomic; // Slot 1: atomic segment
  };
  struct pipes_gda_bnxt_sge sge; // Slot 2: SG entry
} __attribute__((packed));

// =============================================================================
// BNXT CQE Structures
// =============================================================================

// Request completion CQE (24 bytes payload)
struct pipes_gda_bnxt_req_cqe {
  uint64_t qp_handle;
  uint32_t con_indx; // Consumer index (WQE index, 16 bits valid)
  uint32_t rsvd1;
  uint64_t rsvd2;
} __attribute__((packed));

// Base CQE header (8 bytes, follows the payload)
struct pipes_gda_bnxt_bcqe {
  uint32_t flg_st_typ_ph; // Flags, Status, Type, Phase
  uint32_t qphi_rwrid;
} __attribute__((packed));

// CQE bit field encoding
enum pipes_gda_bnxt_bcqe_mask {
  PIPES_GDA_BNXT_BCQE_PH_MASK = 0x01,
  PIPES_GDA_BNXT_BCQE_TYPE_MASK = 0x0F,
  PIPES_GDA_BNXT_BCQE_TYPE_SHIFT = 0x01,
  PIPES_GDA_BNXT_BCQE_RESIZE_TOG_MASK = 0x03,
  PIPES_GDA_BNXT_BCQE_RESIZE_TOG_SHIFT = 0x05,
  PIPES_GDA_BNXT_BCQE_STATUS_MASK = 0xFF,
  PIPES_GDA_BNXT_BCQE_STATUS_SHIFT = 0x08,
  PIPES_GDA_BNXT_BCQE_FLAGS_MASK = 0xFFFF,
  PIPES_GDA_BNXT_BCQE_FLAGS_SHIFT = 0x10,
};

// CQE error status codes
enum pipes_gda_bnxt_req_wc_status {
  PIPES_GDA_BNXT_REQ_ST_OK = 0x00,
  PIPES_GDA_BNXT_REQ_ST_BAD_RESP = 0x01,
  PIPES_GDA_BNXT_REQ_ST_LOC_LEN = 0x02,
  PIPES_GDA_BNXT_REQ_ST_LOC_QP_OP = 0x03,
  PIPES_GDA_BNXT_REQ_ST_PROT = 0x04,
  PIPES_GDA_BNXT_REQ_ST_MEM_OP = 0x05,
  PIPES_GDA_BNXT_REQ_ST_REM_INVAL = 0x06,
  PIPES_GDA_BNXT_REQ_ST_REM_ACC = 0x07,
  PIPES_GDA_BNXT_REQ_ST_REM_OP = 0x08,
  PIPES_GDA_BNXT_REQ_ST_RNR_NAK_XCED = 0x09,
  PIPES_GDA_BNXT_REQ_ST_TRNSP_XCED = 0x0A,
  PIPES_GDA_BNXT_REQ_ST_WR_FLUSH = 0x0B,
};

// =============================================================================
// BNXT Doorbell Structures
// =============================================================================

// Doorbell header (64-bit value)
struct pipes_gda_bnxt_db_hdr {
  uint64_t typ_qid_indx;
};

// Doorbell bit field masks
enum pipes_gda_bnxt_db_mask {
  PIPES_GDA_BNXT_DB_INDX_MASK = 0xFFFFFFUL,
  PIPES_GDA_BNXT_DB_PILO_MASK = 0x0FFUL,
  PIPES_GDA_BNXT_DB_PILO_SHIFT = 0x18,
  PIPES_GDA_BNXT_DB_QID_MASK = 0xFFFFFUL,
  PIPES_GDA_BNXT_DB_PIHI_MASK = 0xF00UL,
  PIPES_GDA_BNXT_DB_PIHI_SHIFT = 0x0C,
  PIPES_GDA_BNXT_DB_TYP_MASK = 0x0FUL,
  PIPES_GDA_BNXT_DB_TYP_SHIFT = 0x1C,
  PIPES_GDA_BNXT_DB_VALID_SHIFT = 0x1A,
  PIPES_GDA_BNXT_DB_EPOCH_SHIFT = 0x18,
  PIPES_GDA_BNXT_DB_TOGGLE_SHIFT = 0x19,
};

// Doorbell queue types
enum pipes_gda_bnxt_db_que_type {
  PIPES_GDA_BNXT_QUE_TYPE_SQ = 0x00,
  PIPES_GDA_BNXT_QUE_TYPE_RQ = 0x01,
  PIPES_GDA_BNXT_QUE_TYPE_SRQ = 0x02,
  PIPES_GDA_BNXT_QUE_TYPE_SRQ_ARM = 0x03,
  PIPES_GDA_BNXT_QUE_TYPE_CQ = 0x04,
  PIPES_GDA_BNXT_QUE_TYPE_CQ_ARMSE = 0x05,
  PIPES_GDA_BNXT_QUE_TYPE_CQ_ARMALL = 0x06,
  PIPES_GDA_BNXT_QUE_TYPE_CQ_ARMENA = 0x07,
  PIPES_GDA_BNXT_QUE_TYPE_CQ_CUT_ACK = 0x09,
  PIPES_GDA_BNXT_QUE_TYPE_NULL = 0x0F,
};

// Epoch flag tracking
enum pipes_gda_bnxt_que_flags_mask {
  PIPES_GDA_BNXT_FLAG_EPOCH_TAIL_SHIFT = 0x0UL,
  PIPES_GDA_BNXT_FLAG_EPOCH_HEAD_SHIFT = 0x1UL,
  PIPES_GDA_BNXT_FLAG_EPOCH_TAIL_MASK = 0x1UL,
  PIPES_GDA_BNXT_FLAG_EPOCH_HEAD_MASK = 0x2UL,
};

enum pipes_gda_bnxt_db_epoch_flag_shift {
  PIPES_GDA_BNXT_DB_EPOCH_TAIL_SHIFT = PIPES_GDA_BNXT_DB_EPOCH_SHIFT,
  PIPES_GDA_BNXT_DB_EPOCH_HEAD_SHIFT = (PIPES_GDA_BNXT_DB_EPOCH_SHIFT - 1),
};

// =============================================================================
// BNXT MSN Table (Packet Sequence Number tracking at end of SQ buffer)
// =============================================================================

struct pipes_gda_bnxt_msns {
  uint64_t start_idx_next_psn_start_psn;
  // bits [23:0]  = start_psn
  // bits [47:24] = next_psn
  // bits [63:48] = start_idx
} __attribute__((packed));

enum pipes_gda_bnxt_msns_mask {
  PIPES_GDA_BNXT_MSN_START_PSN_MASK = 0xFFFFFFUL,
  PIPES_GDA_BNXT_MSN_START_PSN_SHIFT = 0,
  PIPES_GDA_BNXT_MSN_NEXT_PSN_SHIFT = 0x18,
  PIPES_GDA_BNXT_MSN_START_IDX_SHIFT = 0x30,
};

#ifdef __cplusplus
}
#endif
