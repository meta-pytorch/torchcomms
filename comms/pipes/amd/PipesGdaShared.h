// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// PipesGdaShared — Re-exports shared comms::pipes types into pipes_gda
// =============================================================================
//
// The shared comms::pipes headers (ThreadGroup.cuh, Timeout.cuh, IbgdaBuffer.h)
// support both CUDA and HIP. This header re-exports all their types into the
// pipes_gda namespace so AMD code can use a consistent namespace.

#pragma once

#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"

namespace pipes_gda {

// ---------------------------------------------------------------------------
// IbgdaBuffer types
// ---------------------------------------------------------------------------
using comms::pipes::HostLKey;
using comms::pipes::HostRKey;
using comms::pipes::IbgdaBufferExchInfo;
using comms::pipes::IbgdaCmpOp;
using comms::pipes::IbgdaLocalBuffer;
using comms::pipes::IbgdaRemoteBuffer;
using comms::pipes::IbgdaSignalOp;
using comms::pipes::NetworkLKey;
using comms::pipes::NetworkLKeys;
using comms::pipes::NetworkRKey;
using comms::pipes::NetworkRKeys;

// ---------------------------------------------------------------------------
// ThreadGroup types and factory functions
// ---------------------------------------------------------------------------
using comms::pipes::PartitionResult;
using comms::pipes::SyncScope;
using comms::pipes::ThreadGroup;

using comms::pipes::make_block_group;
using comms::pipes::make_multiwarp_group;
using comms::pipes::make_thread_group;
using comms::pipes::make_thread_solo;
using comms::pipes::make_warp_group;

// AMD alias: make_wavefront_group() = make_warp_group()
// (kWarpSize is already 64 on AMD via DeviceConstants.cuh)
__device__ inline ThreadGroup make_wavefront_group() {
  return comms::pipes::make_warp_group();
}

using comms::device::kWarpSize;
constexpr uint32_t kWavefrontSize = comms::device::kWarpSize;
constexpr uint32_t kMultiwarpWavefrontCount = 4;
using comms::pipes::kMaxMultiwarpsPerBlock;
using comms::pipes::kMultiwarpSize;

// ---------------------------------------------------------------------------
// Timeout types and helpers
// ---------------------------------------------------------------------------
using comms::pipes::gpu_clock64;
using comms::pipes::Timeout;

// AMD wall_clock64() clock rate: 100 MHz = 100 ticks per microsecond
constexpr uint64_t kAmdWallClockTicksPerUs = 100;

// Convenience: create a Timeout from microseconds (AMD wall_clock64 @ 100 MHz)
inline Timeout make_timeout_us(uint64_t timeoutUs) {
  return Timeout(timeoutUs * kAmdWallClockTicksPerUs);
}

} // namespace pipes_gda

// TIMEOUT_TRAP_IF_EXPIRED_SINGLE is already defined in comms/pipes/Timeout.cuh
