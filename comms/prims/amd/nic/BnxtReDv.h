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
 * Description: Direct verb support user interface header
 */

// Modifications: (c) Meta Platforms, Inc. and affiliates.

// =============================================================================
// BNXT Direct Verbs API Definitions for pipes-gda
// =============================================================================
//
// Function pointer types and struct definitions for the bnxt_re direct verbs
// library (libbnxt_re.so). These are loaded at runtime via dlopen/dlsym.
// =============================================================================

#pragma once

#include <infiniband/verbs.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// bnxt_re_dv struct definitions
// =============================================================================

struct bnxt_re_dv_qp {
  uint64_t wqe_cnt;
  uint64_t comp_mask;
};

struct bnxt_re_dv_cq {
  uint32_t cqn;
  uint32_t cqe_size;
  uint64_t comp_mask;
};

struct bnxt_re_dv_obj {
  struct {
    struct ibv_qp* in;
    struct bnxt_re_dv_qp* out;
  } qp;
  struct {
    struct ibv_cq* in;
    struct bnxt_re_dv_cq* out;
  } cq;
  struct {
    void* in;
    void* out;
  } srq;
  struct {
    void* in;
    void* out;
  } ah;
  struct {
    void* in;
    void* out;
  } pd;
};

enum bnxt_re_dv_obj_type {
  BNXT_RE_DV_OBJ_QP = 1 << 0,
  BNXT_RE_DV_OBJ_CQ = 1 << 1,
  BNXT_RE_DV_OBJ_SRQ = 1 << 2,
  BNXT_RE_DV_OBJ_AH = 1 << 3,
  BNXT_RE_DV_OBJ_PD = 1 << 4,
};

struct bnxt_re_dv_db_region_attr {
  uint32_t handle;
  uint32_t dpi;
  uint64_t umdbr;
  uint64_t* dbr; // doorbell register pointer
};

struct bnxt_re_dv_umem_reg_attr {
  void* addr;
  size_t size;
  uint32_t access_flags;
  uint64_t pgsz_bitmap;
  uint64_t comp_mask;
  int dmabuf_fd;
};

struct bnxt_re_dv_cq_attr {
  uint32_t ncqe;
  uint32_t cqe_size;
};

struct bnxt_re_dv_cq_init_attr {
  uint64_t cq_handle;
  void* umem_handle;
  uint64_t cq_umem_offset;
  uint32_t ncqe;
};

struct bnxt_re_dv_qp_init_attr {
  enum ibv_qp_type qp_type;
  uint32_t max_send_wr;
  uint32_t max_recv_wr;
  uint32_t max_send_sge;
  uint32_t max_recv_sge;
  uint32_t max_inline_data;
  struct ibv_cq* send_cq;
  struct ibv_cq* recv_cq;
  struct ibv_srq* srq;

  uint64_t qp_handle;
  void* dbr_handle;
  void* sq_umem_handle;
  uint64_t sq_umem_offset;
  uint32_t sq_len;
  uint32_t sq_slots;
  void* rq_umem_handle;
  uint64_t rq_umem_offset;
  uint32_t sq_wqe_sz;
  uint32_t sq_psn_sz;
  uint32_t sq_npsn;
  uint32_t rq_len;
  uint32_t rq_slots;
  uint32_t rq_wqe_sz;
  uint64_t comp_mask;
};

struct bnxt_re_dv_qp_mem_info {
  uint64_t qp_handle;
  uint64_t sq_va;
  uint32_t sq_len;
  uint32_t sq_slots;
  uint32_t sq_wqe_sz;
  uint32_t sq_psn_sz;
  uint32_t sq_npsn;
  uint64_t rq_va;
  uint32_t rq_len;
  uint32_t rq_slots;
  uint32_t rq_wqe_sz;
  uint64_t comp_mask;
};

// =============================================================================
// Function pointer types for dlsym resolution
// =============================================================================

typedef int (*bnxt_re_dv_init_obj_fn)(struct bnxt_re_dv_obj*, uint64_t);
typedef struct bnxt_re_dv_db_region_attr* (*bnxt_re_dv_alloc_db_region_fn)(
    struct ibv_context*);
typedef int (*bnxt_re_dv_free_db_region_fn)(
    struct ibv_context*,
    struct bnxt_re_dv_db_region_attr*);
typedef void* (*bnxt_re_dv_umem_reg_fn)(
    struct ibv_context*,
    struct bnxt_re_dv_umem_reg_attr*);
typedef int (*bnxt_re_dv_umem_dereg_fn)(void*);
typedef void* (*bnxt_re_dv_cq_mem_alloc_fn)(
    struct ibv_context*,
    int,
    struct bnxt_re_dv_cq_attr*);
typedef struct ibv_cq* (*bnxt_re_dv_create_cq_fn)(
    struct ibv_context*,
    struct bnxt_re_dv_cq_init_attr*);
typedef int (*bnxt_re_dv_destroy_cq_fn)(struct ibv_cq*);
typedef int (*bnxt_re_dv_qp_mem_alloc_fn)(
    struct ibv_pd*,
    struct ibv_qp_init_attr*,
    struct bnxt_re_dv_qp_mem_info*);
typedef struct ibv_qp* (
    *bnxt_re_dv_create_qp_fn)(struct ibv_pd*, struct bnxt_re_dv_qp_init_attr*);
typedef int (*bnxt_re_dv_destroy_qp_fn)(struct ibv_qp*);
typedef int (*bnxt_re_dv_modify_qp_fn)(
    struct ibv_qp*,
    struct ibv_qp_attr*,
    int,
    uint32_t,
    uint32_t);

// =============================================================================
// Runtime-loaded function table
// =============================================================================

struct bnxt_re_dv_funcs {
  void* dl_handle;
  bnxt_re_dv_init_obj_fn init_obj;
  bnxt_re_dv_alloc_db_region_fn alloc_db_region;
  bnxt_re_dv_free_db_region_fn free_db_region;
  bnxt_re_dv_umem_reg_fn umem_reg;
  bnxt_re_dv_umem_dereg_fn umem_dereg;
  bnxt_re_dv_cq_mem_alloc_fn cq_mem_alloc;
  bnxt_re_dv_create_cq_fn create_cq;
  bnxt_re_dv_destroy_cq_fn destroy_cq;
  bnxt_re_dv_qp_mem_alloc_fn qp_mem_alloc;
  bnxt_re_dv_create_qp_fn create_qp;
  bnxt_re_dv_destroy_qp_fn destroy_qp;
  bnxt_re_dv_modify_qp_fn modify_qp;
};

#ifdef __cplusplus
}
#endif
