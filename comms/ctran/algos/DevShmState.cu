// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/DevShmState.cuh"
#include "comms/ctran/algos/common/GpeKernel.h"

#if defined(__HIP_PLATFORM_AMD__)
#else
// For CUDA, define the shared memory variables that are declared as extern in
// DevCommon.cuh
__shared__ ctran::CommStateXDev* statex;
__shared__ int* kernelFlag;
__shared__ bool kernelDoAbort;
// TODO: remove once all kernels migrated to populate kernelFlag
__constant__ int placeHolderKernelFlag = KERNEL_STARTED;
#endif
