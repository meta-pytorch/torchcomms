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
// statex needs to be extern and will be defined in DevCommon.cu
extern __shared__ ctran::CommStateXDev* statex;
extern __shared__ int* kernelFlag;
extern __shared__ bool kernelDoAbort;
// TODO: remove once all kernels migrated to populate kernelFlag
extern __constant__ int placeHolderKernelFlag;

// Accessor function to get the device state from dynamic shared memory
__device__ __forceinline__ CtranAlgoDeviceState& getShmDevState() {
  return *reinterpret_cast<CtranAlgoDeviceState*>(dynamicSharedMem);
}
#define shmDevState getShmDevState()
#endif
