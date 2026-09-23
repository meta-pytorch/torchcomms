// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#ifndef __HIP_PLATFORM_AMD__
#include <cuda/atomic>
#endif

#include <cstdint>

#include "comms/prims/core/AbortCheck.cuh"
#include "comms/prims/core/DeviceCheck.cuh"
#include "comms/prims/core/ThreadGroup.cuh"

namespace comms::prims {

/**
 * Cooperative, group-local view of one AbortDevice.
 *
 * Only the stable group leader polls the device handle. Once it observes an
 * abort, it publishes that observation into caller-owned shared memory so
 * workers can leave otherwise-blocking waits without first rendezvousing.
 * The observation is monotonic for the lifetime of this object and may be
 * reused across calls involving different transport peers.
 *
 * `observed()` is intentionally not group-uniform: workers discover the shared
 * bit independently. A caller must converge the complete group before using
 * the result to control any peer-visible side effect.
 *
 * Shared storage is CTA-local. The group must not use CLUSTER scope, and each
 * concurrently active group must receive distinct Shared storage. Every lane
 * must pass an AbortDevice lvalue for the same operation and agree on whether
 * it is enabled.
 */
class GroupAbort {
 public:
  struct alignas(uint32_t) Shared {
   private:
    mutable uint32_t abortObserved;

    // CTA-local sticky bit. SignalState is system-scoped, 128-byte aligned,
    // and its group signal rendezvous cannot run while workers are spinning.
    __device__ __forceinline__ bool loadAcquire() const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#ifdef __HIP_PLATFORM_AMD__
      return __hip_atomic_load(
                 &abortObserved,
                 __ATOMIC_ACQUIRE,
                 __HIP_MEMORY_SCOPE_WORKGROUP) != 0U;
#else
      return cuda::atomic_ref<uint32_t, cuda::thread_scope_block>{abortObserved}
                 .load(cuda::memory_order_acquire) != 0U;
#endif
#else
      return false;
#endif
    }

    __device__ __forceinline__ void storeRelease(uint32_t value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#ifdef __HIP_PLATFORM_AMD__
      __hip_atomic_store(
          &abortObserved,
          value,
          __ATOMIC_RELEASE,
          __HIP_MEMORY_SCOPE_WORKGROUP);
#else
      cuda::atomic_ref<uint32_t, cuda::thread_scope_block>{abortObserved}.store(
          value, cuda::memory_order_release);
#endif
#else
      (void)value;
#endif
    }

    friend class GroupAbort;
  };

  __device__ __forceinline__ static GroupAbort
  initialize(ThreadGroup& group, const AbortDevice& device, Shared& shared) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // GroupAbort uses per-CTA shared memory and block-scope atomics. A cluster
    // group requires distributed shared memory and cluster-scope ordering.
    PIPES_DEVICE_CHECK(group.scope != SyncScope::CLUSTER);
    if (device.isEnabled()) {
      if (group.is_leader()) {
        shared.storeRelease(0U);
      }
      group.sync();
    }
#else
    (void)group;
#endif
    return GroupAbort(device, shared);
  }

  __device__ static GroupAbort
  initialize(ThreadGroup&, AbortDevice&&, Shared&) = delete;
  __device__ static GroupAbort
  initialize(ThreadGroup&, const AbortDevice&&, Shared&) = delete;

  __host__ __device__ __forceinline__ bool isEnabled() const {
    return device_.isEnabled();
  }

  /** Return this lane's current observation without synchronizing the group. */
  __device__ __forceinline__ bool observed() const {
    return isEnabled() && shared_.loadAcquire();
  }

  /**
   * Let the stable group leader poll the underlying AbortDevice.
   *
   * Returns true only to the leader after an abort is already sticky or this
   * call observes one. Workers return false and learn the result through
   * observed().
   */
  __device__ __forceinline__ bool checkLeader(ThreadGroup& group) const {
    if (!isEnabled() || !group.is_leader()) {
      return false;
    }
    if (shared_.loadAcquire()) {
      return true;
    }
    if (FT_ABORT_CHECK(device_, "GroupAbort cooperative wait")) {
      shared_.storeRelease(1U);
      return true;
    }
    return false;
  }

 private:
  __host__ __device__ GroupAbort(const AbortDevice& device, Shared& shared)
      : device_(device), shared_(shared) {}

  const AbortDevice& device_;
  Shared& shared_;
};

} // namespace comms::prims
