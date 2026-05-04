// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// HIP device shim: provide __trap() (mapped to abort()) and pull in
// hip_runtime.h so device-side blockIdx/threadIdx are visible to any header
// that uses them (e.g. printf-before-trap diagnostics in IbgdaBuffer.h).
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
#include <hip/hip_runtime.h>
#define __trap() abort()
#endif
