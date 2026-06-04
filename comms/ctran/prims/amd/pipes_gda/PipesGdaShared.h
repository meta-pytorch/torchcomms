// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// PipesGdaShared — Re-exports shared ctran::prims types into pipes_gda
// =============================================================================
//
// The shared ctran::prims headers (ThreadGroup.cuh, Timeout.cuh, IbgdaBuffer.h)
// support both CUDA and HIP. This header re-exports all their types into the
// pipes_gda namespace so AMD code can use a consistent namespace.

#pragma once

#include "comms/ctran/prims/IbgdaBuffer.h"
#include "comms/ctran/prims/ThreadGroup.cuh"
#include "comms/ctran/prims/Timeout.cuh"

namespace pipes_gda {

// ---------------------------------------------------------------------------
// IbgdaBuffer types
// ---------------------------------------------------------------------------
using ctran::prims::HostLKey;
using ctran::prims::HostRKey;
using ctran::prims::IbgdaBufferExchInfo;
using ctran::prims::IbgdaCmpOp;
using ctran::prims::IbgdaLocalBuffer;
using ctran::prims::IbgdaRemoteBuffer;
using ctran::prims::IbgdaSignalOp;
using ctran::prims::NetworkLKey;
using ctran::prims::NetworkLKeys;
using ctran::prims::NetworkRKey;
using ctran::prims::NetworkRKeys;

// ---------------------------------------------------------------------------
// ThreadGroup types and factory functions
// ---------------------------------------------------------------------------
using ctran::prims::PartitionResult;
using ctran::prims::SyncScope;
using ctran::prims::ThreadGroup;

using ctran::prims::make_block_group;
using ctran::prims::make_multiwarp_group;
using ctran::prims::make_thread_group;
using ctran::prims::make_thread_solo;
using ctran::prims::make_warp_group;

// AMD alias: make_wavefront_group() = make_warp_group()
// (kWarpSize is already 64 on AMD via DeviceConstants.cuh)
__device__ inline ThreadGroup make_wavefront_group() {
  return ctran::prims::make_warp_group();
}

using comms::device::kWarpSize;
constexpr uint32_t kWavefrontSize = comms::device::kWarpSize;
constexpr uint32_t kMultiwarpWavefrontCount = 4;
using ctran::prims::kMaxMultiwarpsPerBlock;
using ctran::prims::kMultiwarpSize;

// ---------------------------------------------------------------------------
// Timeout types and helpers
// ---------------------------------------------------------------------------
using ctran::prims::gpu_clock64;
using ctran::prims::Timeout;

// AMD wall_clock64() clock rate: 100 MHz = 100 ticks per microsecond
constexpr uint64_t kAmdWallClockTicksPerUs = 100;

// Convenience: create a Timeout from microseconds (AMD wall_clock64 @ 100 MHz)
inline Timeout make_timeout_us(uint64_t timeoutUs) {
  return Timeout(timeoutUs * kAmdWallClockTicksPerUs);
}

} // namespace pipes_gda

// TIMEOUT_TRAP_IF_EXPIRED_SINGLE is already defined in
// comms/ctran/prims/Timeout.cuh
