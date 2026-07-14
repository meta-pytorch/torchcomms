// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// =============================================================================
// PIPES_IS_DEVICE_COMPILE
// =============================================================================
//
// True (1) when the current compiler pass is the device pass (CUDA or HIP).
// HIP parses `__device__` function bodies during the host pass too, so
// device-only intrinsics like `__trap()`, atomic ops, and certain warp/wave
// primitives need to be gated to avoid host-pass parse errors.
//
// Usage:
//   #if PIPES_IS_DEVICE_COMPILE
//     // device-only code
//   #endif
//
// Prefer this over the raw `#if defined(__CUDA_ARCH__) ||
// defined(__HIP_DEVICE_COMPILE__)` pattern, which appears in many places across
// `comms/prims` device headers.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define PIPES_IS_DEVICE_COMPILE 1
#else
#define PIPES_IS_DEVICE_COMPILE 0
#endif

// =============================================================================
// PIPES_DEVICE_TRAP
// =============================================================================
//
// Cross-platform fatal-trap macro. `__trap()` is a device-only intrinsic;
// HIP parses `__device__` function bodies during the host pass too and
// rejects `__trap()` there. Expand to `__trap()` only in the device pass
// (CUDA or HIP) so the host pass parses cleanly. Expands to a no-op on
// the host pass.
//
// Usage:
//   if (bad_state) {
//     printf("FATAL: ...\n");
//     PIPES_DEVICE_TRAP();
//   }
#if PIPES_IS_DEVICE_COMPILE
#define PIPES_DEVICE_TRAP() __trap()
#else
#define PIPES_DEVICE_TRAP() ((void)0)
#endif
