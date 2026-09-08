// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/GpeKernel.h"
#include "comms/ctran/commstate/CommStateXDev.h"

#if defined(__HIP_PLATFORM_AMD__)
/* For AMD/HIP:
 - Buck's HIP compiler treats "extern __shared__" for pointer types as host
 variables
 - CMake's linker fails with duplicate symbols if we use __shared__ without
 extern
 - Hencing using static for the variables.
   (this is fine since __shared__ variables are per-kernel-block anyway)
*/
extern __shared__ char dynamicSharedMem[];
static __shared__ ctran::CommStateXDev* statex;
static __shared__ int* kernelFlag;
static __shared__ bool kernelDoAbort;
// TODO: remove once all kernels migrated to populate kernelFlag
static __constant__ int placeHolderKernelFlag;

// Accessor function to get the device state from dynamic shared memory
__device__ __forceinline__ CtranAlgoDeviceState& getShmDevState() {
  return *reinterpret_cast<CtranAlgoDeviceState*>(dynamicSharedMem);
}
#define shmDevState getShmDevState()

#else
// Use dynamic shared memory because on GB200 shemDevState may exceeed the
// static shared memory limit 48KB - declare as char array and cast to struct
extern __shared__ char dynamicSharedMem[];
/* statex/kernelFlag/kernelDoAbort must keep internal linkage.
 *
 * nvlink gives every externally-linked __shared__ variable in a device-link
 * unit one address in a single flat module-wide block. CTRAN is device-linked
 * together with ncclx's src/device objects, whose common.cu defines
 * __shared__ ncclShmemData ncclShmem (~36.8KB). As extern, these three land
 * above ncclShmem in that block, so every kernel touching one of them reserves
 * ~36.8KB of static shared memory it never reads - enough to keep a CTRAN
 * kernel from being co-resident with a large GEMM.
 *
 * Consequence: every function that reads or writes them must be inlined into
 * the kernel. A non-inlined __device__ function in another TU would silently
 * bind to a different copy.
 */
static __shared__ ctran::CommStateXDev* statex;
static __shared__ int* kernelFlag;
static __shared__ bool kernelDoAbort;
// TODO: remove once all kernels migrated to populate kernelFlag
extern __constant__ int placeHolderKernelFlag;

// Accessor function to get the device state from dynamic shared memory
__device__ __forceinline__ CtranAlgoDeviceState& getShmDevState() {
  return *reinterpret_cast<CtranAlgoDeviceState*>(dynamicSharedMem);
}
#define shmDevState getShmDevState()
#endif
