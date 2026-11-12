/*************************************************************************
 * Copyright (c), Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Stub definitions of rocshmem::constmem and rocshmem::logd_constants.
//
// The rocshmem QP device bitcode (queue_pair.bc) references these as
// extern __constant__ symbols.  In a full rocshmem build they are defined
// in rocshmem_gpu.cpp, but the GIN build only links minimal QP bitcode
// into the symmetric fat-object kernels.
//
// constmem.gda_provider IS read at runtime: every QueuePair operation GIN GDA
// uses (put_nbi, atomic_add, quiet) switches on it (rocshmem queue_pair.cpp).
// This copy is linked into each GIN code object with -mlink-builtin-bitcode
// and has no host shadow, so rocshmem_gin_init_constmem() cannot reach it:
// the provider is baked in at build time (RCCL_GIN_GDA_PROVIDER, set when
// rocSHMEM is built for a single provider). 0 (UNSET) makes every QueuePair
// operation hit assert(false /* invalid nic provider */).
#ifndef RCCL_GIN_GDA_PROVIDER
#define RCCL_GIN_GDA_PROVIDER 0
#endif

#include <hip/hip_runtime.h>
#include "gda/gda_enums.hpp"
#include "rocshmem/rocshmem_common.hpp"

namespace rocshmem {

struct constmem_t {
  BackendType backend_type;
  GDAProvider gda_provider;
  uint64_t alltoall_wg_algo;
  int my_pe;
  int num_pes;
  int ipc_first_pe;
  int ipc_stride;
  int ipc_shm_size;
} __attribute__((aligned(16)));

struct logd_constants {
  int pe_number;
  int flags;
} __attribute__((aligned(16)));

__constant__ constmem_t constmem = {BackendType::GDA_BACKEND, static_cast<GDAProvider>(RCCL_GIN_GDA_PROVIDER)};
// SHOW_ERROR, so device-side QP error reports (BUILD_DEBUG_DEVICE builds) print.
__constant__ struct logd_constants logd_constants = {-1, 1};

}  // namespace rocshmem
