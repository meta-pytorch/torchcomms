// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

// Forward declarations to avoid pulling MultiPeerNvlTransport.h /
// GpuMemHandler.h (and their `meta::comms::DeviceBuffer` reference) into
// every consumer's translation unit. The .cc file owns the heavy includes.
namespace meta::comms {
class IBootstrap;
} // namespace meta::comms

namespace comms::prims {
class GpuMemHandler;
class MultiPeerNvlTransport;
} // namespace comms::prims

namespace comms::prims::moe_ep {

/**
 * IntranodeRuntime — host-side runtime for Phase 1 (intranode dispatch /
 * combine over NVLink).
 *
 * Owns:
 *  - `comms::prims::GpuMemHandler` (mode `kCudaIpcUncached`) for the
 *    NVL data buffer + barrier/ptr-table block.
 *  - `comms::prims::MultiPeerNvlTransport` for the per-peer NVL P2P state +
 *    signal/barrier slots that the dispatch/combine kernels use.
 *  - Persistent workspace (`NUM_WORKSPACE_BYTES` = 32 MiB) for atomic
 *    counters and per-expert state.
 *
 * One IntranodeRuntime per `Buffer` instance. Constructed from `Buffer::sync`
 * after device IDs + IPC handles have been gathered Python-side.
 */
class IntranodeRuntime {
 public:
  IntranodeRuntime(
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      int rank,
      int numRanks,
      std::size_t numNvlBytes);

  /**
   * Pre-allocated-buffer ctor — used by `Buffer::sync` when the local NVL
   * buffer was already allocated by the `Buffer` ctor and the per-peer
   * IPC handles were gathered Python-side via `dist.all_gather_object`.
   * Skips `GpuMemHandler` entirely because we already have everything it
   * would have produced.
   *
   * `peerDataPtrs` is a host-side vector of size `numRanks` where entry
   * `i` is the local-process pointer to peer `i`'s NVL data buffer
   * (entry `rank` == `localBuffer`; other entries come from
   * `cudaIpcOpenMemHandle`). The runtime takes ownership of opening the
   * peer handles before this ctor is called and closing them in its
   * destructor.
   */
  IntranodeRuntime(
      int rank,
      int numRanks,
      std::size_t numNvlBytes,
      void* localBuffer,
      const cudaIpcMemHandle_t& localHandle,
      std::vector<void*> peerDataPtrs);

  ~IntranodeRuntime();

  // Non-copyable, non-movable.
  IntranodeRuntime(const IntranodeRuntime&) = delete;
  IntranodeRuntime& operator=(const IntranodeRuntime&) = delete;
  IntranodeRuntime(IntranodeRuntime&&) = delete;
  IntranodeRuntime& operator=(IntranodeRuntime&&) = delete;

  /** Local rank within the group [0, numRanks). */
  int rank() const noexcept {
    return rank_;
  }

  /** Total number of ranks (== num NVL peers in intranode-only mode). */
  int numRanks() const noexcept {
    return numRanks_;
  }

  /** Local pointer to the NVL data buffer (data + barrier slots + per-peer
   *  ptr table; layout owned by callers, computed as offsets on this base
   *  pointer). */
  void* getLocalDataPtr() const;

  /** Per-peer remote pointer for the NVL data buffer (peer-mapped via IPC). */
  void* getPeerDataPtr(int peerRank) const;

  /** Persistent workspace pointer — `NUM_WORKSPACE_BYTES` (32 MiB) of
   *  GPU-uncached memory for atomic counters + per-expert state. */
  void* getWorkspacePtr() const noexcept {
    return workspace_;
  }

  /** Per-runtime atomic counter used by combine's grid_barrier (AMD path).
   *  4-byte device buffer. */
  int* getDispatchGlobalAtomicCounter() const noexcept {
    return dispatchGlobalAtomicCounter_;
  }
  int* getCombineGlobalAtomicCounter() const noexcept {
    return combineGlobalAtomicCounter_;
  }

  /** Communication stream used by the runtime's kernel launches. */
  cudaStream_t commStream() const noexcept {
    return commStream_;
  }

  /** Underlying NVLink multi-peer transport (per-peer signal/barrier slots
   *  + state buffers). Defined in the .cc to keep the heavy include out of
   *  consumers' translation units. */
  comms::prims::MultiPeerNvlTransport& transport();
  const comms::prims::MultiPeerNvlTransport& transport() const;

  /** GpuMemHandler that owns the kCudaIpcUncached data buffer. */
  comms::prims::GpuMemHandler& memHandler();

  /** `void**` device array of size `numRanks` with peer-mapped NVL data
   *  buffer pointers (entry `i` = `getPeerDataPtr(i)`; entry == self points
   *  to local). Passed to the dispatch / combine kernels as `buffer_ptrs`. */
  void** getPeerDataPtrsDevice() const noexcept {
    return peerDataPtrsDevice_;
  }

  /** `int**` device array of size `numRanks` with peer-mapped barrier-signal
   *  pointers (carved out of the workspace). Passed to `notify_dispatch` /
   *  `cached_notify_dispatch` / `cached_notify_combine` as `task_fifo_ptrs`. */
  int** getBarrierSignalPtrsDevice() const noexcept {
    return barrierSignalPtrsDevice_;
  }

  /** Host-pinned `volatile int*` mapped into device address space. The
   *  notify_dispatch kernel writes the local rank's recv-token count here;
   *  Buffer::intranode_dispatch reads it from the host side to size the
   *  output tensors. */
  int* getMoeRecvCounterHost() const noexcept {
    return moeRecvCounterHost_;
  }
  int* getMoeRecvCounterDevice() const noexcept {
    return moeRecvCounterDevice_;
  }
  int* getMoeRecvExpertCounterHost() const noexcept {
    return moeRecvExpertCounterHost_;
  }
  int* getMoeRecvExpertCounterDevice() const noexcept {
    return moeRecvExpertCounterDevice_;
  }

  /** Rotating FIFO head used by notify_dispatch / cached_notify_dispatch /
   *  cached_notify_combine to avoid stomping on prior barrier slots. */
  int head() const noexcept {
    return head_;
  }
  void moveFifoSlots(int n);

 private:
  const int rank_;
  const int numRanks_;
  const std::size_t numNvlBytes_;

  std::unique_ptr<comms::prims::MultiPeerNvlTransport> transport_;
  std::unique_ptr<comms::prims::GpuMemHandler> memHandler_;

  // Persistent workspace — NUM_WORKSPACE_BYTES (see KernelConfigs.cuh).
  void* workspace_{nullptr};
  int* dispatchGlobalAtomicCounter_{nullptr};
  int* combineGlobalAtomicCounter_{nullptr};

  cudaStream_t commStream_{nullptr};

  // Peer-mapped pointer arrays (allocated on the device, populated from
  // `memHandler_->getPeerDeviceMemPtr(i)` in the ctor).
  void** peerDataPtrsDevice_{nullptr};
  int** barrierSignalPtrsDevice_{nullptr};

  // Host-pinned counters whose device shadow is the kernel-visible one.
  int* moeRecvCounterHost_{nullptr};
  int* moeRecvCounterDevice_{nullptr};
  int* moeRecvExpertCounterHost_{nullptr};
  int* moeRecvExpertCounterDevice_{nullptr};

  // Rotating barrier head — bumped after every notify_*.
  int head_{0};

  // Fallback storage when memHandler_ is null (pre-allocated-buffer ctor
  // path). getLocalDataPtr() / getPeerDataPtr() consult these instead of
  // memHandler_ when the runtime was constructed with caller-owned buffers.
  void* localBufferPtr_{nullptr};
  std::vector<void*> peerDataPtrsHost_;
};

} // namespace comms::prims::moe_ep
