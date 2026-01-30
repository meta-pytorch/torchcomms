// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

/**
 * MLX5 (Mellanox ConnectX) hardware-specific structs and constants.
 *
 * This header contains low-level C structs that map directly to MLX5 NIC
 * hardware structures (CQEs, WQEs, doorbells, etc.). These are separated
 * from the higher-level ibverbs abstractions in Ibvcore.h.
 *
 * These structs are used for:
 * - GPU-direct RDMA (IBGDA) where kernels directly manipulate NIC structures
 * - Host-side mlx5dv (direct verbs) operations
 * - Parsing completion queue entries and work queue elements
 */

#include <linux/types.h>
#include <stdint.h>
#include <sys/types.h>

namespace ibverbx {

// Forward declarations for ibv types used in mlx5dv_obj
struct ibv_qp;
struct ibv_cq;
struct ibv_srq;
struct ibv_wq;
struct ibv_dm;
struct ibv_ah;
struct ibv_pd;

//------------------------------------------------------------------------------
// MLX5 DMA Buffer Access Flags
//------------------------------------------------------------------------------

enum mlx5dv_reg_dmabuf_access {
  MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT = (1 << 0),
};

//------------------------------------------------------------------------------
// Tag Matching Header (used in CQE)
//------------------------------------------------------------------------------

struct ibv_tmh {
  uint8_t opcode; /* from enum ibv_tmh_op */
  uint8_t reserved[3]; /* must be zero */
  __be32 app_ctx; /* opaque user data */
  __be64 tag;
};

//------------------------------------------------------------------------------
// Completion Queue Entry (CQE) Structures
//------------------------------------------------------------------------------

struct mlx5_err_cqe {
  uint8_t rsvd0[32];
  uint32_t srqn;
  uint8_t rsvd1[18];
  uint8_t vendor_err_synd;
  uint8_t syndrome;
  uint32_t s_wqe_opcode_qpn;
  uint16_t wqe_counter;
  uint8_t signature;
  uint8_t op_own;
};

struct mlx5_tm_cqe {
  __be32 success;
  __be16 hw_phase_cnt;
  uint8_t rsvd0[12];
};

struct mlx5_cqe64 {
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
    struct mlx5_tm_cqe tm_cqe;
    /* TMH is scattered to CQE upon match */
    struct ibv_tmh tmh;
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

//------------------------------------------------------------------------------
// MLX5 Direct Verbs (mlx5dv) Structures
//------------------------------------------------------------------------------

struct mlx5dv_qp {
  __be32* dbrec;
  struct {
    void* buf;
    uint32_t wqe_cnt;
    uint32_t stride;
  } sq;
  struct {
    void* buf;
    uint32_t wqe_cnt;
    uint32_t stride;
  } rq;
  struct {
    void* reg;
    uint32_t size;
  } bf;
  uint64_t comp_mask;
  off_t uar_mmap_offset;
  uint32_t tirn;
  uint32_t tisn;
  uint32_t rqn;
  uint32_t sqn;
  uint64_t tir_icm_addr;
};

struct mlx5dv_cq {
  void* buf;
  __be32* dbrec;
  uint32_t cqe_cnt;
  uint32_t cqe_size;
  void* cq_uar;
  uint32_t cqn;
  uint64_t comp_mask;
};

struct mlx5dv_srq {
  void* buf;
  __be32* dbrec;
  uint32_t stride;
  uint32_t head;
  uint32_t tail;
  uint64_t comp_mask;
  uint32_t srqn;
};

struct mlx5dv_rwq {
  void* buf;
  __be32* dbrec;
  uint32_t wqe_cnt;
  uint32_t stride;
  uint64_t comp_mask;
};

struct mlx5dv_dm {
  void* buf;
  uint64_t length;
  uint64_t comp_mask;
  uint64_t remote_va;
};

struct mlx5_wqe_av;

struct mlx5dv_ah {
  struct mlx5_wqe_av* av;
  uint64_t comp_mask;
};

struct mlx5dv_pd {
  uint32_t pdn;
  uint64_t comp_mask;
};

struct mlx5dv_obj {
  struct {
    struct ibv_qp* in;
    struct mlx5dv_qp* out;
  } qp;
  struct {
    struct ibv_cq* in;
    struct mlx5dv_cq* out;
  } cq;
  struct {
    struct ibv_srq* in;
    struct mlx5dv_srq* out;
  } srq;
  struct {
    struct ibv_wq* in;
    struct mlx5dv_rwq* out;
  } rwq;
  struct {
    struct ibv_dm* in;
    struct mlx5dv_dm* out;
  } dm;
  struct {
    struct ibv_ah* in;
    struct mlx5dv_ah* out;
  } ah;
  struct {
    struct ibv_pd* in;
    struct mlx5dv_pd* out;
  } pd;
};

enum mlx5dv_obj_type {
  MLX5DV_OBJ_QP = 1 << 0,
  MLX5DV_OBJ_CQ = 1 << 1,
  MLX5DV_OBJ_SRQ = 1 << 2,
  MLX5DV_OBJ_RWQ = 1 << 3,
  MLX5DV_OBJ_DM = 1 << 4,
  MLX5DV_OBJ_AH = 1 << 5,
  MLX5DV_OBJ_PD = 1 << 6,
};

//------------------------------------------------------------------------------
// DC (Dynamically Connected) Transport Structures
//------------------------------------------------------------------------------

enum mlx5dv_qp_init_attr_mask {
  MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS = 1 << 0,
  MLX5DV_QP_INIT_ATTR_MASK_DC = 1 << 1,
  MLX5DV_QP_INIT_ATTR_MASK_SEND_OPS_FLAGS = 1 << 2,
  MLX5DV_QP_INIT_ATTR_MASK_DCI_STREAMS = 1 << 3,
};

enum mlx5dv_dc_type {
  MLX5DV_DCTYPE_DCT = 1,
  MLX5DV_DCTYPE_DCI,
};

struct mlx5dv_dci_streams {
  uint8_t log_num_concurent;
  uint8_t log_num_errored;
};

struct mlx5dv_dc_init_attr {
  enum mlx5dv_dc_type dc_type;
  union {
    uint64_t dct_access_key;
    struct mlx5dv_dci_streams dci_streams;
  };
};

struct mlx5dv_qp_init_attr {
  uint64_t comp_mask; // Use enum mlx5dv_qp_init_attr_mask
  uint32_t create_flags; // Use enum mlx5dv_qp_create_flags
  struct mlx5dv_dc_init_attr dc_init_attr;
  uint64_t send_ops_flags; // Use enum mlx5dv_qp_create_send_ops_flags
};

//------------------------------------------------------------------------------
// CQE Opcode and Status Constants
//------------------------------------------------------------------------------

enum {
  MLX5_CQE_OWNER_MASK = 1,
  MLX5_CQE_REQ = 0,
  MLX5_CQE_RESP_WR_IMM = 1,
  MLX5_CQE_RESP_SEND = 2,
  MLX5_CQE_RESP_SEND_IMM = 3,
  MLX5_CQE_RESP_SEND_INV = 4,
  MLX5_CQE_RESIZE_CQ = 5,
  MLX5_CQE_NO_PACKET = 6,
  MLX5_CQE_SIG_ERR = 12,
  MLX5_CQE_REQ_ERR = 13,
  MLX5_CQE_RESP_ERR = 14,
  MLX5_CQE_INVALID = 15,
};

enum {
  MLX5_INVALID_LKEY = 0x100,
};

enum {
  MLX5_EXTENDED_UD_AV = 0x80000000,
};

//------------------------------------------------------------------------------
// WQE Control Flags
//------------------------------------------------------------------------------

enum {
  MLX5_WQE_CTRL_CQ_UPDATE = 2 << 2,
  MLX5_WQE_CTRL_SOLICITED = 1 << 1,
  MLX5_WQE_CTRL_FENCE = 4 << 5,
  MLX5_WQE_CTRL_INITIATOR_SMALL_FENCE = 1 << 5,
};

enum {
  MLX5_SEND_WQE_BB = 64,
  MLX5_SEND_WQE_SHIFT = 6,
};

enum {
  MLX5_INLINE_SEG = 0x80000000,
};

enum {
  MLX5_ETH_WQE_L3_CSUM = (1 << 6),
  MLX5_ETH_WQE_L4_CSUM = (1 << 7),
};

//------------------------------------------------------------------------------
// Work Queue Element (WQE) Segment Structures
//------------------------------------------------------------------------------

struct mlx5_wqe_srq_next_seg {
  uint8_t rsvd0[2];
  __be16 next_wqe_index;
  uint8_t signature;
  uint8_t rsvd1[11];
};

struct mlx5_wqe_data_seg {
  __be32 byte_count;
  __be32 lkey;
  __be64 addr;
};

struct mlx5_wqe_ctrl_seg {
  __be32 opmod_idx_opcode;
  __be32 qpn_ds;
  uint8_t signature;
  __be16 dci_stream_channel_id;
  uint8_t fm_ce_se;
  __be32 imm;
} __attribute__((__packed__)) __attribute__((__aligned__(4)));

struct mlx5_wqe_raddr_seg {
  __be64 raddr;
  __be32 rkey;
  __be32 reserved;
};

struct mlx5_wqe_atomic_seg {
  __be64 swap_add;
  __be64 compare;
};

//------------------------------------------------------------------------------
// MLX5 Opcodes
//------------------------------------------------------------------------------

enum {
  MLX5_OPCODE_NOP = 0x00,
  MLX5_OPCODE_SEND_INVAL = 0x01,
  MLX5_OPCODE_RDMA_WRITE = 0x08,
  MLX5_OPCODE_RDMA_WRITE_IMM = 0x09,
  MLX5_OPCODE_SEND = 0x0a,
  MLX5_OPCODE_SEND_IMM = 0x0b,
  MLX5_OPCODE_TSO = 0x0e,
  MLX5_OPCODE_RDMA_READ = 0x10,
  MLX5_OPCODE_ATOMIC_CS = 0x11,
  MLX5_OPCODE_ATOMIC_FA = 0x12,
  MLX5_OPCODE_ATOMIC_MASKED_CS = 0x14,
  MLX5_OPCODE_ATOMIC_MASKED_FA = 0x15,
  MLX5_OPCODE_FMR = 0x19,
  MLX5_OPCODE_LOCAL_INVAL = 0x1b,
  MLX5_OPCODE_CONFIG_CMD = 0x1f,
  MLX5_OPCODE_SET_PSV = 0x20,
  MLX5_OPCODE_UMR = 0x25,
  MLX5_OPCODE_TAG_MATCHING = 0x28,
  MLX5_OPCODE_FLOW_TBL_ACCESS = 0x2c,
  MLX5_OPCODE_MMO = 0x2F,
};

} // namespace ibverbx
