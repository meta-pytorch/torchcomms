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

/**
 * @file VerbsDev.h
 * @brief GDAKI common definitions
 *
 * @{
 */
#ifndef PIPES_GDA_VERBS_DEV_H
#define PIPES_GDA_VERBS_DEV_H

#include "nic/Mlx5Hsi.h" // @manual
#include "verbs/VerbsDef.h" // @manual

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @typedef pipes_gda_gpu_dev_verbs_ticket_t
 * @brief Ticket type used in one-sided APIs.
 */
typedef uint64_t pipes_gda_gpu_dev_verbs_ticket_t;

/**
 * Describes IBGDA dev WQE crtl segment.
 */
struct pipes_gda_gpu_dev_verbs_wqe_ctrl_seg {
  __be32 opmod_idx_opcode; /**< opcode + wqe idx */
  __be32 qpn_ds; /**< qp number */
  union {
    struct {
      uint8_t signature; /**< signature */
      uint8_t rsvd[2]; /**< reserved */
      uint8_t fm_ce_se; /**< fm_ce_se */
    };
    struct {
      __be32 signature_fm_ce_se; /**< all flags in or */
    };
  };

  __be32 imm; /**< immediate */
} __attribute__((__aligned__(8)));

/**
 * Describes IBGDA dev WQE crtl segment.
 */
struct pipes_gda_gpu_dev_verbs_wqe_wait_seg {
  uint32_t resv[2];
  __be32 max_index;
  __be32 qpn_cqn;
} __attribute__((__packed__)) __attribute__((__aligned__(8)));

/**
 * @struct pipes_gda_gpu_dev_verbs_addr
 * @brief This structure holds the address and key of a memory region.
 */
struct pipes_gda_gpu_dev_verbs_addr {
  uint64_t addr;
  __be32 key;
};

/**
 * Describes IBGDA dev general WQE.
 */
struct pipes_gda_gpu_dev_verbs_wqe {
  union {
    /* Generic inline Data */
    struct {
      uint8_t inl_data[64];
    };

    /* Generic Data */
    struct {
      struct pipes_gda_ib_mlx5_wqe_data_seg dseg0;
      struct pipes_gda_ib_mlx5_wqe_data_seg dseg1;
      struct pipes_gda_ib_mlx5_wqe_data_seg dseg2;
      struct pipes_gda_ib_mlx5_wqe_data_seg dseg3;
    };

    /* Read/Write */
    struct {
      struct pipes_gda_gpu_dev_verbs_wqe_ctrl_seg rw_cseg;
      struct pipes_gda_ib_mlx5_wqe_raddr_seg rw_rseg;
      struct pipes_gda_ib_mlx5_wqe_data_seg rw_dseg0;
      struct pipes_gda_ib_mlx5_wqe_data_seg rw_dseg1;
    };

    /* Atomic */
    struct {
      struct pipes_gda_gpu_dev_verbs_wqe_ctrl_seg at_cseg;
      struct pipes_gda_ib_mlx5_wqe_raddr_seg at_rseg;
      struct pipes_gda_ib_mlx5_wqe_atomic_seg at_seg;
      struct pipes_gda_ib_mlx5_wqe_data_seg at_dseg;
    };

    /* Send */
    struct {
      struct pipes_gda_gpu_dev_verbs_wqe_ctrl_seg snd_cseg;
      struct pipes_gda_ib_mlx5_wqe_data_seg snd_dseg0;
      struct pipes_gda_ib_mlx5_wqe_data_seg snd_dseg1;
      struct pipes_gda_ib_mlx5_wqe_data_seg snd_dseg2;
    };

    /* Wait */
    struct {
      struct pipes_gda_gpu_dev_verbs_wqe_ctrl_seg wait_cseg;
      struct pipes_gda_gpu_dev_verbs_wqe_wait_seg wait_dseg;
      struct pipes_gda_ib_mlx5_wqe_data_seg padding0;
      struct pipes_gda_ib_mlx5_wqe_data_seg padding1;
    };
  };
} __attribute__((__aligned__(8)));

/**
 * Describes IBGDA dev CQ
 */
struct pipes_gda_gpu_dev_verbs_cq {
  uint8_t* cqe_daddr; /**< CQE address */
  uint32_t cq_num; /**< CQ number */
  uint32_t cqe_num; /**< Total number of CQEs in CQ */
  __be32* dbrec; /**< CQE Doorbell Record */
  uint64_t cqe_ci; /**< CQE Consumer Index */
  uint32_t cqe_mask; /**< Mask of total number of CQEs in CQ */
  uint8_t cqe_size; /**< Single CQE size (64B default) */
  uint64_t cqe_rsvd; /**< All previous CQEs are polled */
  enum pipes_gda_gpu_dev_verbs_mem_type
      mem_type; ///< Memory type of the completion queue
};

// =============================================================================
// NIC-specific QP extension structures
// =============================================================================
// Each NIC backend can store its private state here. The QP struct uses a
// union so that only one NIC's fields are active at a time. This avoids
// #ifdef in this header and allows all NIC backends to compile cleanly.

/**
 * BNXT-specific QP extension fields.
 */
struct pipes_gda_gpu_dev_verbs_qp_bnxt {
  uint32_t sq_depth; ///< SQ depth in slots (not WQEs)
  uint32_t sq_head; ///< Consumer head pointer (in slots)
  uint32_t sq_tail; ///< Producer tail pointer (in slots)
  uint32_t sq_flags; ///< Epoch flags for wrap-around detection
  uint32_t sq_id; ///< QP ID used in doorbell value

  void* msntbl; ///< MSN table pointer (GPU memory)
  uint32_t msn; ///< Current MSN index
  uint32_t msn_tbl_sz; ///< MSN table size (number of entries)
  uint32_t psn; ///< Current Packet Sequence Number
  uint32_t psn_sz_log2; ///< log2(PSN entry size)
  uint64_t mtu; ///< MTU for packet counting

  volatile uint64_t* dbr; ///< GPU-accessible doorbell register pointer

  void* cq_buf; ///< CQ buffer pointer (GPU memory)
  uint32_t cq_depth; ///< CQ depth (typically 1 for CQE compression)

  int sq_lock; ///< GPU-side spinlock for SQ serialization
};

/**
 * MLX5-specific QP extension fields (placeholder for future use).
 * Currently mlx5 uses the common QP fields directly.
 */
struct pipes_gda_gpu_dev_verbs_qp_mlx5 {
  uint8_t reserved; ///< Placeholder (mlx5 uses common QP fields)
};

/**
 * Ionic-specific QP extension fields.
 * Ionic uses color-bit CQ polling and MSN-based completion tracking.
 * Doorbell is a 64-bit write to a memory-mapped register.
 */
struct pipes_gda_gpu_dev_verbs_qp_ionic {
  // SQ doorbell register (GPU-mapped via HSA)
  volatile uint64_t* sq_dbreg; ///< SQ doorbell register pointer
  uint64_t sq_dbval; ///< SQ base doorbell value (OR'd with masked position)
  uint16_t sq_mask; ///< SQ index mask (depth - 1)

  // SQ buffer (ionic_v1_wqe entries)
  void* sq_buf; ///< SQ WQE buffer pointer (GPU-accessible)

  // SQ producer tracking
  uint32_t sq_prod; ///< Next SQ producer index (atomic)
  uint32_t sq_dbprod; ///< Last doorbell'd producer index
  int sq_lock; ///< Spinlock for doorbell serialization

  // CQ doorbell register (GPU-mapped via HSA)
  volatile uint64_t* cq_dbreg; ///< CQ doorbell register pointer
  uint64_t cq_dbval; ///< CQ base doorbell value
  uint16_t cq_mask; ///< CQ index mask (depth - 1, 0 for CCQE mode)

  // CQ buffer (ionic_v1_cqe entries)
  void* cq_buf; ///< CQ buffer pointer (GPU-accessible)

  // CQ consumer tracking
  uint32_t cq_pos; ///< Current CQ consumer position
  uint32_t cq_dbpos; ///< Last doorbell'd CQ consumer position
  int cq_lock; ///< Spinlock for CQ polling serialization

  // MSN (Message Sequence Number) tracking
  uint32_t sq_msn; ///< Last completed MSN from CQ
};

/**
 * Describes IBGDA dev QP
 */
struct pipes_gda_gpu_dev_verbs_qp {
  uint64_t sq_rsvd_index; ///< All WQE slots prior to this index are reserved
  uint64_t sq_ready_index; ///< All WQE slots prior to this index are ready
  uint64_t sq_wqe_pi; /**< SQ WQE producer index */
  uint32_t sq_num; /**< SQ num */
  uint32_t sq_num_shift8; /**< SQ num << 8 */
  uint32_t sq_num_shift8_be; /**< SQ num << 8 big endian */
  uint32_t sq_num_shift8_be_1ds; /**< SQ num << 8 big endian */
  uint32_t sq_num_shift8_be_2ds; /**< SQ num << 8 big endian */
  uint32_t sq_num_shift8_be_3ds; /**< SQ num << 8 big endian */
  uint32_t sq_num_shift8_be_4ds; /**< SQ num << 8 big endian */
  int sq_lock; /**< SQ lock */
  uint16_t sq_wqe_num; /**< Number of SQ WQE slots */
  uint16_t sq_wqe_mask; /**< SQ WQE index mask (sq_wqe_num - 1) */
  uint8_t* sq_wqe_daddr; /**< SQ WQE buffer device address */
  __be32* sq_dbrec; /**< SQ doorbell record address */
  uint64_t* sq_db; /**< SQ doorbell (BlueFlame UAR) address */

  /* Unused fields (reserved for compatibility) */
  uint32_t rq_num; /**< RQ number (unused) */
  uint64_t rq_wqe_pi; /**< RQ WQE producer index (unused) */
  uint32_t rq_wqe_num; /**< Number of RQ WQE slots (unused) */
  uint32_t rq_wqe_mask; /**< RQ WQE index mask (unused) */
  uint8_t* rq_wqe_daddr; /**< RQ WQE buffer device address (unused) */
  __be32* rq_dbrec; /**< RQ doorbell record address (unused) */
  uint32_t rcv_wqe_size; /**< Receive WQE size (unused) */
  uint64_t rq_rsvd_index; /**< All previous WQEs are reserved */
  uint64_t rq_ready_index; /**< All previous WQEs are ready */
  int rq_lock; /**< RQ lock */

  struct pipes_gda_gpu_dev_verbs_cq cq_sq; /**< SQ CQ connected to QP */
  struct pipes_gda_gpu_dev_verbs_cq cq_rq; /**< RQ CQ connected to QP */

  enum pipes_gda_gpu_dev_verbs_nic_handler nic_handler; ///< NIC handler
  enum pipes_gda_gpu_dev_verbs_mem_type
      mem_type; ///< Memory type of the completion

  /**
   * NIC-specific extension fields (union: only one active at a time).
   * Access via qp->nic.bnxt, qp->nic.mlx5, etc.
   */
  union {
    struct pipes_gda_gpu_dev_verbs_qp_bnxt bnxt;
    struct pipes_gda_gpu_dev_verbs_qp_mlx5 mlx5;
    struct pipes_gda_gpu_dev_verbs_qp_ionic ionic;
  } nic;
} __attribute__((__aligned__(8)));

#ifdef __cplusplus
}
#endif

#endif /* PIPES_GDA_VERBS_DEV_H */

/** @} */
