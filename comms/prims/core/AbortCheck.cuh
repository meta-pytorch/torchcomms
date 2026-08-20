// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <cstdio>

#include "comms/common/fault_tolerance/AbortDevice.cuh"
#include "comms/common/fault_tolerance/AbortMacros.cuh"
#include "comms/prims/transport/amd/HipHostCompat.h"

namespace comms::prims {

using AbortDevice = comms::fault_tolerance::AbortDevice;

__device__ __forceinline__ uint64_t gpu_clock64() {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
  return wall_clock64();
#elif defined(__CUDA_ARCH__)
  return clock64();
#else
  return 0;
#endif
}

/**
 * Group-uniform "has this operation aborted?", for use between a wait and the
 * peer-visible side effect that follows it.
 *
 * `FT_ABORT_BREAK` terminates the *spin* it is written in; it says nothing to
 * the pipeline loop around that spin. A loop that keeps going after its wait
 * gave up will still issue the put, the fused DATA_READY, and the SLOT_FREE
 * credit for every remaining chunk -- which does not merely publish garbage, it
 * releases peers that were correctly blocked and stops them ever reaching their
 * own deadline. The fault then looks, to those peers, like a successful
 * collective. So the loop needs its own exit, and this is the predicate for it.
 *
 * Group-uniform by construction: the leader polls and broadcasts. A per-thread
 * verdict is unsafe here because `checkExpired()` gates its shared read on a
 * per-copy poll interval, so two threads can legitimately disagree for a few
 * microseconds -- and a subset breaking would leave the rest of the group
 * stranded at the next `group.sync()`, which is undefined behavior.
 *
 * Templated on the group type to avoid pulling ThreadGroup.cuh into this
 * header, matching `AbortDevice::checkExpired(group)`.
 */
template <typename Group>
__device__ __forceinline__ bool groupAborted(
    Group& group,
    const AbortDevice& abort) {
  uint32_t aborted = 0;
  if (group.is_leader()) {
    aborted = abort.checkExpired() ? 1U : 0U;
  }
  return group.template broadcast<uint32_t>(aborted) != 0U;
}

} // namespace comms::prims

// The FT_ABORT_* checks live in comms/common/fault_tolerance/AbortMacros.cuh
// and are re-exported here so Prims device code keeps a single include for the
// abort handle, the device clock, and the checks.
