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
// MLX5 (Mellanox/NVIDIA ConnectX) Hardware-Software Interface for pipes-gda
// =============================================================================
//
// MLX5 NIC WQE, CQE, and doorbell structures for GPU-initiated RDMA.
// MLX5 hardware interface definitions for GPU-initiated RDMA.
//
// This file is the MLX5 equivalent of BnxtHsi.h:
//   - WQE segment structs (data, control, raddr, atomic, inline)
//   - CQE structs (cqe64, error CQE)
//   - MLX5 opcodes, control flags, CQE status codes
//   - MLX5-specific constants (SQ shift, doorbell record indices)
// =============================================================================

#pragma once

#include <linux/types.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// MLX5 WQE Constants
// =============================================================================

#define PIPES_GDA_IB_MLX5_WQE_SQ_SHIFT 6

// =============================================================================
// MLX5 WQE Opcodes
// =============================================================================

enum {
  PIPES_GDA_IB_MLX5_OPCODE_NOP = 0x00,
  PIPES_GDA_IB_MLX5_OPCODE_SEND_INVAL = 0x01,
  PIPES_GDA_IB_MLX5_OPCODE_RDMA_WRITE = 0x08,
  PIPES_GDA_IB_MLX5_OPCODE_RDMA_WRITE_IMM = 0x09,
  PIPES_GDA_IB_MLX5_OPCODE_SEND = 0x0a,
  PIPES_GDA_IB_MLX5_OPCODE_SEND_IMM = 0x0b,
  PIPES_GDA_IB_MLX5_OPCODE_TSO = 0x0e,
  PIPES_GDA_IB_MLX5_OPCODE_RDMA_READ = 0x10,
  PIPES_GDA_IB_MLX5_OPCODE_ATOMIC_CS = 0x11,
  PIPES_GDA_IB_MLX5_OPCODE_ATOMIC_FA = 0x12,
  PIPES_GDA_IB_MLX5_OPCODE_ATOMIC_MASKED_CS = 0x14,
  PIPES_GDA_IB_MLX5_OPCODE_ATOMIC_MASKED_FA = 0x15,
  PIPES_GDA_IB_MLX5_OPCODE_FMR = 0x19,
  PIPES_GDA_IB_MLX5_OPCODE_LOCAL_INVAL = 0x1b,
  PIPES_GDA_IB_MLX5_OPCODE_WAIT = 0x0f,
  PIPES_GDA_IB_MLX5_OPCODE_CONFIG_CMD = 0x1f,
  PIPES_GDA_IB_MLX5_OPCODE_SET_PSV = 0x20,
  PIPES_GDA_IB_MLX5_OPCODE_DUMP = 0x23,
  PIPES_GDA_IB_MLX5_OPCODE_UMR = 0x25,
  PIPES_GDA_IB_MLX5_OPCODE_TAG_MATCHING = 0x28,
  PIPES_GDA_IB_MLX5_OPCODE_FLOW_TBL_ACCESS = 0x2c,
  PIPES_GDA_IB_MLX5_OPCODE_MMO = 0x2F,
};

// =============================================================================
// MLX5 WQE Control Flags
// =============================================================================

enum {
  PIPES_GDA_IB_MLX5_WQE_CTRL_CE_CQE_ON_CQE_ERROR = 0x0,
  PIPES_GDA_IB_MLX5_WQE_CTRL_CE_CQE_ON_FIRST_CQE_ERROR = 0x1,
  PIPES_GDA_IB_MLX5_WQE_CTRL_CE_CQE_ALWAYS = 0x2,
  PIPES_GDA_IB_MLX5_WQE_CTRL_CE_CQE_AND_EQE = 0x3,
};

enum {
  PIPES_GDA_IB_MLX5_WQE_CTRL_FM_NO_FENCE = 0x0,
  PIPES_GDA_IB_MLX5_WQE_CTRL_FM_INITIATOR_SMALL_FENCE = 0x1,
  PIPES_GDA_IB_MLX5_WQE_CTRL_FM_FENCE = 0x2,
  PIPES_GDA_IB_MLX5_WQE_CTRL_FM_STRONG_ORDERING = 0x3,
  PIPES_GDA_IB_MLX5_WQE_CTRL_FM_FENCE_AND_INITIATOR_SMALL_FENCE = 0x4,
  PIPES_GDA_IB_MLX5_WQE_CTRL_FM_CUSTOM = 0x100,
};

enum pipes_gda_gpu_dev_verbs_wqe_ctrl_flags {
  PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE =
      PIPES_GDA_IB_MLX5_WQE_CTRL_CE_CQE_ALWAYS << 2,
  PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_ERROR_UPDATE =
      PIPES_GDA_IB_MLX5_WQE_CTRL_CE_CQE_ON_CQE_ERROR << 2,
  PIPES_GDA_IB_MLX5_WQE_CTRL_CQ_FIRST_CQE_ERROR =
      PIPES_GDA_IB_MLX5_WQE_CTRL_CE_CQE_ON_FIRST_CQE_ERROR << 2,
  PIPES_GDA_IB_MLX5_WQE_CTRL_SOLICITED = 1 << 1,
  PIPES_GDA_IB_MLX5_WQE_CTRL_FENCE =
      PIPES_GDA_IB_MLX5_WQE_CTRL_FM_FENCE_AND_INITIATOR_SMALL_FENCE << 5,
  PIPES_GDA_IB_MLX5_WQE_CTRL_INITIATOR_SMALL_FENCE =
      PIPES_GDA_IB_MLX5_WQE_CTRL_FM_INITIATOR_SMALL_FENCE << 5,
  PIPES_GDA_IB_MLX5_WQE_CTRL_STRONG_ORDERING =
      PIPES_GDA_IB_MLX5_WQE_CTRL_FM_STRONG_ORDERING << 5
};

// =============================================================================
// MLX5 Doorbell Record Indices
// =============================================================================

enum {
  PIPES_GDA_IB_MLX5_RCV_DBR = 0,
  PIPES_GDA_IB_MLX5_SND_DBR = 1,
};

// =============================================================================
// MLX5 WQE Segment Count Constants
// =============================================================================

enum {
  PIPES_GDA_VERBS_WQE_SEG_CNT_RDMA_WRITE_INL_MIN = 3,
  PIPES_GDA_VERBS_WQE_SEG_CNT_RDMA_WRITE_INL_MAX = 4,
  PIPES_GDA_VERBS_WQE_SEG_CNT_ATOMIC_FA_CAS = 4,
  PIPES_GDA_VERBS_WQE_SEG_CNT_WAIT = 2
};

enum {
  PIPES_GDA_IB_MLX5_INLINE_SEG = 0x80000000,
};

// =============================================================================
// MLX5 CQE Status Codes
// =============================================================================

#define PIPES_GDA_VERBS_MLX5_CQE_OPCODE_SHIFT 4

enum {
  PIPES_GDA_IB_MLX5_CQE_OWNER_MASK = 1,
  PIPES_GDA_IB_MLX5_CQE_REQ = 0,
  PIPES_GDA_IB_MLX5_CQE_RESP_WR_IMM = 1,
  PIPES_GDA_IB_MLX5_CQE_RESP_SEND = 2,
  PIPES_GDA_IB_MLX5_CQE_RESP_SEND_IMM = 3,
  PIPES_GDA_IB_MLX5_CQE_RESP_SEND_INV = 4,
  PIPES_GDA_IB_MLX5_CQE_RESIZE_CQ = 5,
  PIPES_GDA_IB_MLX5_CQE_NO_PACKET = 6,
  PIPES_GDA_IB_MLX5_CQE_SIG_ERR = 12,
  PIPES_GDA_IB_MLX5_CQE_REQ_ERR = 13,
  PIPES_GDA_IB_MLX5_CQE_RESP_ERR = 14,
  PIPES_GDA_IB_MLX5_CQE_INVALID = 15,
};

// =============================================================================
// MLX5 WQE Segment Structures
// =============================================================================

struct pipes_gda_ib_mlx5_wqe_data_seg {
  __be32 byte_count;
  __be32 lkey;
  __be64 addr;
};

struct pipes_gda_ib_mlx5_wqe_ctrl_seg {
  __be32 opmod_idx_opcode;
  __be32 qpn_ds;
  uint8_t signature;
  __be16 dci_stream_channel_id;
  uint8_t fm_ce_se;
  __be32 imm;
} __attribute__((__packed__)) __attribute__((__aligned__(4)));

struct pipes_gda_ib_mlx5_wqe_raddr_seg {
  __be64 raddr;
  __be32 rkey;
  __be32 reserved;
};

struct pipes_gda_ib_mlx5_wqe_atomic_seg {
  __be64 swap_add;
  __be64 compare;
};

struct pipes_gda_ib_mlx5_wqe_inl_data_seg {
  uint32_t byte_count;
};

// =============================================================================
// MLX5 CQE Structures
// =============================================================================

struct pipes_gda_ib_mlx5_tm_cqe {
  __be32 success;
  __be16 hw_phase_cnt;
  uint8_t rsvd0[12];
};

struct pipes_gda_ib_ibv_tmh {
  uint8_t opcode;
  uint8_t reserved[3];
  __be32 app_ctx;
  __be64 tag;
};

struct pipes_gda_ib_mlx5_cqe64 {
  union {
    struct {
      uint8_t rsvd0[2];
      __be16 wqe_id;
      uint8_t rsvd4[13];
      uint8_t ml_path;
      uint8_t rsvd20[4];
      __be16 slid;
      __be32 flags_rqpn;
      uint8_t hds_ip_ext;
      uint8_t l4_hdr_type_etc;
      __be16 vlan_info;
    };
    struct pipes_gda_ib_mlx5_tm_cqe tm_cqe;
    struct pipes_gda_ib_ibv_tmh tmh;
  };
  __be32 srqn_uidx;
  __be32 imm_inval_pkey;
  uint8_t app;
  uint8_t app_op;
  __be16 app_info;
  __be32 byte_cnt;
  __be64 timestamp;
  __be32 sop_drop_qpn;
  __be16 wqe_counter;
  uint8_t signature;
  uint8_t op_own;
};

struct pipes_gda_ib_mlx5_err_cqe_ex {
  uint8_t rsvd0[32];
  __be32 srqn;
  uint8_t rsvd1[16];
  uint8_t hw_err_synd;
  uint8_t hw_synd_type;
  uint8_t vendor_err_synd;
  uint8_t syndrome;
  __be32 s_wqe_opcode_qpn;
  __be16 wqe_counter;
  uint8_t signature;
  uint8_t op_own;
};

#ifdef __cplusplus
}
#endif
