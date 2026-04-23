// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// On HIP, __trap() is not available; use abort() instead.
// Include this header in any device code that calls __trap().
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
#define __trap() abort()
#endif
