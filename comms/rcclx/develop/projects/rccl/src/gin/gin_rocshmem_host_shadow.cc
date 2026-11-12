/*************************************************************************
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host definitions of the rocSHMEM device globals that the GIN host objects reference
// (RCCL_GIN_LINK_ROCSHMEM). rocSHMEM defines them in rocshmem_gpu.cpp, an RDC object whose fatbin
// exists only in a -fgpu-rdc --hip-link, so linking that object into librccl.so would leave its
// fatbin undefined. The GIN code objects carry their own device copies (gin_rocshmem_constmem.hip),
// so rocshmem_gin_init_constmem() finds no registered symbol here and leaves them alone.

namespace rocshmem {
struct constmem_t {
  unsigned char bytes[64];
} __attribute__((aligned(16)));
struct logd_constants {
  int pe_number;
  int flags;
} __attribute__((aligned(16)));

constmem_t constmem;
struct logd_constants logd_constants;
int log_pe_number;
}  // namespace rocshmem
