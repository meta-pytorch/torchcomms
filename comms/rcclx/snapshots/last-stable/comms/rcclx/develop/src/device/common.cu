/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "common.h"

#ifndef RCCL_DEVICE_LINKER
__shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ < 700
  __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif
#endif

#if defined(USE_INDIRECT_FUNCTION_CALL) || defined(RCCL_DEVICE_LINKER)
__device__ void ncclDevFunc_Nop();
#else
__device__ __attribute__((noinline)) void ncclDevFunc_Nop();
#endif
