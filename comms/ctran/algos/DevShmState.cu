// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/DevShmState.cuh"
#include "comms/ctran/algos/common/GpeKernel.h"

#if defined(__HIP_PLATFORM_AMD__)
#else
// statex/kernelFlag/kernelDoAbort are static __shared__ in DevShmState.cuh, so
// they need no definition here.
// TODO: remove once all kernels migrated to populate kernelFlag
__constant__ int placeHolderKernelFlag = KERNEL_STARTED;
#endif
