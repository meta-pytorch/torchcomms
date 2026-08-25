// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// PrimsAmdGdaShared — Re-exports shared comms::prims types into prims_amd_gda
// =============================================================================
//
// The shared comms::prims headers (ThreadGroup.cuh, Timeout.cuh, IbgdaBuffer.h)
// support both CUDA and HIP. This header re-exports all their types into the
// prims_amd_gda namespace so AMD code can use a consistent namespace.

#pragma once

#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"

namespace prims_amd_gda {

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

} // namespace prims_amd_gda

// The FT_ABORT_* checks come from
// comms/common/fault_tolerance/AbortMacros.cuh, re-exported via
// comms/prims/core/Timeout.cuh
