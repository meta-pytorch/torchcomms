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
// Common Verbs Definitions (NIC-agnostic)
// =============================================================================
//
// NIC-agnostic constants, enums, and macros shared by all NIC backends.
// NIC-specific hardware structs are in separate files:
//   - nic/Mlx5Hsi.h: MLX5 WQE/CQE structs, opcodes, control flags
//   - nic/BnxtHsi.h: BNXT WQE/CQE structs, opcodes, doorbell
//
// Shared constants and enums for GPU-initiated RDMA verbs.
// =============================================================================

#pragma once

#include <limits.h>
#include <linux/types.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Common Macros
// =============================================================================

#define PIPES_GDA_VOLATILE(x) (*(volatile typeof(x)*)&(x))

#define PIPES_GDA_VERBS_WARP_SIZE 32
#define PIPES_GDA_VERBS_WARP_FULL_MASK 0xffffffff
#define PIPES_GDA_VERBS_PAGE_SIZE 65536

#define PIPES_GDA_VERBS_CQE_CI_MASK 0xFFFFFF
#define PIPES_GDA_VERBS_WQE_PI_MASK 0xFFFF

#define PIPES_GDA_VERBS_MKEY_SWAPPED 1

#ifndef PIPES_GDA_VERBS_ENABLE_DEBUG
#define PIPES_GDA_VERBS_ENABLE_DEBUG 0
#endif

#if PIPES_GDA_VERBS_ENABLE_DEBUG == 1
#include <assert.h>
#define PIPES_GDA_VERBS_ASSERT(x) assert(x)
#else
#define PIPES_GDA_VERBS_ASSERT(x) \
  do {                            \
  } while (0)
#endif

#define PIPES_GDA_VERBS_MAX_INLINE_SIZE 28
#define PIPES_GDA_VERBS_CQE_SIZE 64
#define PIPES_GDA_VERBS_WQE_IDX_SHIFT 8

#define PIPES_GDA_VERBS_MAX_TRANSFER_SIZE_SHIFT 30
#define PIPES_GDA_VERBS_MAX_TRANSFER_SIZE \
  (1ULL << PIPES_GDA_VERBS_MAX_TRANSFER_SIZE_SHIFT) // 1GiB

#ifndef ACCESS_ONCE
#define ACCESS_ONCE(x) (*(volatile typeof(x)*)&(x))
#endif

#ifndef READ_ONCE
#define READ_ONCE(x) ACCESS_ONCE(x)
#endif

#ifndef WRITE_ONCE
#define WRITE_ONCE(x, v) (ACCESS_ONCE(x) = (v))
#endif

// =============================================================================
// Common Enums (NIC-agnostic)
// =============================================================================

enum pipes_gda_gpu_dev_verbs_mem_type {
  PIPES_GDA_VERBS_MEM_TYPE_AUTO = 0,
  PIPES_GDA_VERBS_MEM_TYPE_HOST = 1,
  PIPES_GDA_VERBS_MEM_TYPE_GPU = 2,
  PIPES_GDA_VERBS_MEM_TYPE_MAX = INT_MAX
};

enum pipes_gda_gpu_dev_verbs_qp_type {
  PIPES_GDA_VERBS_QP_SQ = 0,
};

enum pipes_gda_gpu_dev_verbs_exec_scope {
  PIPES_GDA_VERBS_EXEC_SCOPE_THREAD = 0,
  PIPES_GDA_VERBS_EXEC_SCOPE_WARP
};

enum pipes_gda_gpu_dev_verbs_sync_scope {
  PIPES_GDA_VERBS_SYNC_SCOPE_SYS = 0,
  PIPES_GDA_VERBS_SYNC_SCOPE_GPU = 1,
  PIPES_GDA_VERBS_SYNC_SCOPE_CTA = 2,
  PIPES_GDA_VERBS_SYNC_SCOPE_MAX = INT_MAX
};

enum pipes_gda_gpu_dev_verbs_resource_sharing_mode {
  PIPES_GDA_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE = 0,
  PIPES_GDA_VERBS_RESOURCE_SHARING_MODE_CTA = 1,
  PIPES_GDA_VERBS_RESOURCE_SHARING_MODE_GPU = 2,
  PIPES_GDA_VERBS_RESOURCE_SHARING_MODE_MAX = INT_MAX
};

enum pipes_gda_gpu_dev_verbs_nic_handler {
  PIPES_GDA_VERBS_NIC_HANDLER_AUTO = 0,
  PIPES_GDA_VERBS_NIC_HANDLER_CPU_PROXY = 1,
  PIPES_GDA_VERBS_NIC_HANDLER_GPU_SM_DB = 2,
  PIPES_GDA_VERBS_NIC_HANDLER_GPU_SM_BF = 3,
  PIPES_GDA_VERBS_NIC_HANDLER_TYPE_MAX,
};

enum pipes_gda_gpu_dev_verbs_gpu_code_opt {
  PIPES_GDA_VERBS_GPU_CODE_OPT_DEFAULT = 0,
  PIPES_GDA_VERBS_GPU_CODE_OPT_ASYNC_STORE_RELEASE = (1 << 0),
  PIPES_GDA_VERBS_GPU_CODE_OPT_MAX = INT_MAX
};

enum pipes_gda_gpu_dev_verbs_signal_op {
  PIPES_GDA_VERBS_SIGNAL_OP_ADD = 0,
};

#ifdef __cplusplus
}
#endif
