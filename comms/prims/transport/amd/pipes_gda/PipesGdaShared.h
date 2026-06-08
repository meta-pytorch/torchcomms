// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// PipesGdaShared — Re-exports shared comms::prims types into pipes_gda
// =============================================================================
//
// The shared comms::prims headers (ThreadGroup.cuh, Timeout.cuh, IbgdaBuffer.h)
// support both CUDA and HIP. This header re-exports all their types into the
// pipes_gda namespace so AMD code can use a consistent namespace.

#pragma once

#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"

namespace pipes_gda {

// ---------------------------------------------------------------------------
// IbgdaBuffer types
// ---------------------------------------------------------------------------
using comms::prims::HostLKey;
using comms::prims::HostRKey;
using comms::prims::IbgdaBufferExchInfo;
using comms::prims::IbgdaCmpOp;
using comms::prims::IbgdaLocalBuffer;
using comms::prims::IbgdaRemoteBuffer;
using comms::prims::IbgdaSignalOp;
using comms::prims::NetworkLKey;
using comms::prims::NetworkLKeys;
using comms::prims::NetworkRKey;
using comms::prims::NetworkRKeys;

// ---------------------------------------------------------------------------
// ThreadGroup types and factory functions
// ---------------------------------------------------------------------------
using comms::prims::PartitionResult;
using comms::prims::SyncScope;
using comms::prims::ThreadGroup;

using comms::prims::make_block_group;
using comms::prims::make_multiwarp_group;
using comms::prims::make_thread_group;
using comms::prims::make_thread_solo;
using comms::prims::make_warp_group;

// AMD alias: make_wavefront_group() = make_warp_group()
// (kWarpSize is already 64 on AMD via DeviceConstants.cuh)
__device__ inline ThreadGroup make_wavefront_group() {
  return comms::prims::make_warp_group();
}

using comms::device::kWarpSize;
constexpr uint32_t kWavefrontSize = comms::device::kWarpSize;
constexpr uint32_t kMultiwarpWavefrontCount = 4;
using comms::prims::kMaxMultiwarpsPerBlock;
using comms::prims::kMultiwarpSize;

// ---------------------------------------------------------------------------
// Timeout types and helpers
// ---------------------------------------------------------------------------
using comms::prims::gpu_clock64;
using comms::prims::Timeout;

// AMD wall_clock64() clock rate: 100 MHz = 100 ticks per microsecond
constexpr uint64_t kAmdWallClockTicksPerUs = 100;

// Convenience: create a Timeout from microseconds (AMD wall_clock64 @ 100 MHz)
inline Timeout make_timeout_us(uint64_t timeoutUs) {
  return Timeout(timeoutUs * kAmdWallClockTicksPerUs);
}

} // namespace pipes_gda

// TIMEOUT_TRAP_IF_EXPIRED_SINGLE is already defined in
// comms/prims/core/Timeout.cuh
