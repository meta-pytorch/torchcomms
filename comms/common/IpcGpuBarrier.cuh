// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <memory>
#include "comms/common/IpcMemHandler.h"
#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/utils/CudaRAII.h"

namespace meta::comms {

namespace {

template <std::memory_order Sem>
__device__ __forceinline__ uint32_t
cas(uint32_t* addr, uint32_t compare, uint32_t val) {
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
  ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system> ref(*addr);
  ref.compare_exchange_strong(compare, val, ::cuda::std::memory_order(Sem));
  return compare;
#elif defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
  __atomic_compare_exchange_n(
      addr, &compare, val, false, static_cast<int>(Sem), __ATOMIC_RELAXED);
  return compare;
#endif
}

template <std::memory_order Sem>
__device__ __forceinline__ void putFlag(uint32_t* addr) {
  while (cas<Sem>(addr, 0, 1) != 0)
    ;
}

template <std::memory_order Sem>
__device__ __forceinline__ void waitFlag(uint32_t* addr) {
  while (cas<Sem>(addr, 1, 0) != 1)
    ;
}

constexpr int NRANKS = 8;

} // namespace

// A mailbox in device memory to facilitate the GPU barrier operations. The
// DeviceMailbox's underlying memory is managed by the DeviceBuffer returned by
// the mallocAndInit function.
class DeviceMailbox {
 public:
  using FlagType = uint32_t;
  // Allocate device memory and initialize DeviceMailbox. The returned
  // DeviceBuffer is the underlying memory of DeviceMailbox and needs to be kept
  // alive throughout the lifetime of DeviceMailbox.
  static __host__ std::pair<std::unique_ptr<DeviceBuffer>, DeviceMailbox>
  mallocAndInit(int nRanks, int nBlocks);

  DeviceMailbox() = default;

  // Init from an user-provided device mem pointer.
  __host__ DeviceMailbox(int nRanks, int nBlocks, void* flagsBuf);

  __device__ inline void setFlagNoMemFence(int senderRank, int senderBlock) {
    putFlag<std::memory_order_relaxed>(
        flags_ + getFlagIdx(senderRank, senderBlock));
  }

  __device__ inline void waitFlagNoMemFence(int senderRank, int senderBlock) {
    waitFlag<std::memory_order_relaxed>(
        flags_ + getFlagIdx(senderRank, senderBlock));
  }

  __device__ inline void setFlagWithMemFence(int senderRank, int senderBlock) {
    putFlag<std::memory_order_release>(
        flags_ + getFlagIdx(senderRank, senderBlock));
  }

  __device__ inline void waitFlagWithMemFence(int senderRank, int senderBlock) {
    waitFlag<std::memory_order_acquire>(
        flags_ + getFlagIdx(senderRank, senderBlock));
  }

 private:
  int nBlocks_;
  // flags_ is an array of size [nBlocks][nRanks]
  FlagType* flags_;

  __device__ inline int getFlagIdx(int rank, int block) {
    return block * NRANKS + rank;
  }
};

class IpcGpuBarrier;

// A RAII class for the device memory owned by IpcGpuBarrier. This class
// should be kept alive throughout the lifetime of IpcGpuBarrier
struct IpcGpuBarrierResources {
  std::unique_ptr<IpcMemHandler> ipcMemHandler;
  std::unique_ptr<DeviceBuffer> selfMailboxBuf;
};

class IpcGpuBarrier {
 public:
  using FlagType = DeviceMailbox::FlagType;
  __host__ IpcGpuBarrier() = default;

  // The returned IpcGpuBarrierResources should be kept alive throughout the
  // lifetime of IpcGpuBarrier; the returned IpcGpuBarrier* is a pointer to
  // IpcGpuBarrier object in device memory, so you can pass this IpcGpuBarrier*
  // to kernel function.
  static __host__
      std::pair<std::unique_ptr<IpcGpuBarrierResources>, IpcGpuBarrier>
      mallocAndInit(
          int nRanks,
          int nBlocks,
          int selfRank,
          std::shared_ptr<IBootstrap> bootstrap);

  // If the kernel has peer rank mem access before the barrier, set
  // hasPreviousMemAccess=true; If the kernel has peer rank mem access after the
  // barrier, set hasSubsequentMemAccess=true. Usually, the first barrier in the
  // kernel uses hasPreviousMemAccess=false; the last barrier in the kernel uses
  // hasSubsequentMemAccess=false.
  template <bool hasPreviousMemAccess, bool hasSubsequentMemAccess>
  __device__ __forceinline__ void syncOnSameBlockIdx() {
    enum class MemFenceType {
      RELEASE_ACQUIRE,
      // Do release set but relaxed read.
      // This is usually used in the end barrier in a kernel. For example, if
      // a consumer rank reads from its peer rank (the producer rank)'s ipcbuf
      // before the barrier, then you want to guarantee that the read
      // transaction is completed before the barrier, so the producer rank won't
      // exit the kernel and overwrite the ipcbuf data. Acquire pattern is not
      // needed in this case because there is no mem access after the end
      // barrier.
      RELEASE_ONLY,
      // Do acquire load but relaxed set.
      /// This is usually used in the start barrier in a kernel. For example, if
      // a consumer rank reads from its peer rank (the producer rank)'s ipcbuf
      // after the barrier, then you want to guarantee that the read transaction
      // won't be reordered before the barrier, so the consumer rank won't read
      // the peer rank's ipcbuf until the data is ready. Release pattern is not
      // needed in this case because there is no mem access before the start
      // barrier.
      ACQUIRE_ONLY,
    };

    // It doesn't make sense to have a barrier without any mem access around it.
    static_assert(hasPreviousMemAccess || hasSubsequentMemAccess);

    constexpr MemFenceType fenceType =
        hasPreviousMemAccess && hasSubsequentMemAccess
        ? MemFenceType::RELEASE_ACQUIRE
        : (!hasPreviousMemAccess ? MemFenceType::ACQUIRE_ONLY
                                 : MemFenceType::RELEASE_ONLY);

    if constexpr (hasPreviousMemAccess) {
      // make sure all the threads in the block have made to this point before
      // we notify other peer ranks that I'm ready.
      __syncthreads();
    }
    if (threadIdx.x < NRANKS) {
      auto peerRank = threadIdx.x;
      // set the flag in peer rank's mailbox and then wait for self mailbox's
      // flag to be updated by peers
      if constexpr (fenceType == MemFenceType::ACQUIRE_ONLY) {
        allMailboxes_[peerRank].setFlagNoMemFence(selfRank_, blockIdx.x);
      } else {
        allMailboxes_[peerRank].setFlagWithMemFence(selfRank_, blockIdx.x);
      }

      if constexpr (fenceType == MemFenceType::RELEASE_ONLY) {
        allMailboxes_[selfRank_].waitFlagNoMemFence(peerRank, blockIdx.x);
      } else {
        allMailboxes_[selfRank_].waitFlagWithMemFence(peerRank, blockIdx.x);
      }
    }
    // sync remaining threads in this block so they all wait until the peer rank
    // is ready
    if constexpr (hasSubsequentMemAccess) {
      __syncthreads();
    }
  }

 private:
  int nBlocks_{-1};
  int selfRank_{-1};
  // allMailboxes_ is an arrary of DeviceMailbox in device memory
  std::array<DeviceMailbox, NRANKS> allMailboxes_;

  __host__ IpcGpuBarrier(
      int nRanks,
      int nBlocks,
      int selfRank,
      const std::array<DeviceMailbox, NRANKS>& allMailboxes);
};

} // namespace meta::comms
