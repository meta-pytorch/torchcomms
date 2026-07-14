// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "comms/prims/collectives/moe_ep/cpp/low_latency/Layout.h"

// Forward declarations to avoid pulling MultipeerIbgdaTransport.h into
// every consumer's translation unit. The .cc file owns the heavy includes.
namespace meta::comms {
class IBootstrap;
} // namespace meta::comms

namespace comms::prims {
class MultipeerIbgdaTransport;
class MultiPeerNvlTransport;
class GpuMemHandler;
struct MultipeerIbgdaDeviceTransport;
struct IbgdaLocalBuffer;
struct IbgdaRemoteBuffer;
} // namespace comms::prims

namespace comms::prims::moe_ep {

/**
 * LowLatencyRuntime — host-side runtime for low-latency dispatch /
 * combine over IBGDA RDMA.
 *
 * Owns:
 *  - `comms::prims::MultipeerIbgdaTransport` for the cross-host RDMA path
 *    (BNXT on AMD, mlx5 on NVIDIA).
 *  - One symmetric LL data buffer (`numRdmaBytes` bytes) allocated via
 *    `hipExtMallocWithFlags(hipDeviceMallocUncached)` on AMD /
 *    `cudaMalloc` on NVIDIA, registered with the IBGDA transport.
 *  - Optionally `comms::prims::MultiPeerNvlTransport` when
 *    `allow_nvlink_for_low_latency_mode=True` and peers are intra-node.
 *  - Per-(local-expert × src-rank) atomic counters in workspace.
 *  - LowLatencyLayout — buffer offset table.
 *
 * One LowLatencyRuntime per `Buffer` instance with `low_latency_mode=True`.
 * Constructed from `Buffer::sync` after `nvshmem_unique_id` (placeholder)
 * has been gathered Python-side.
 */
class LowLatencyRuntime {
 public:
  /** @param externalRdmaBuffer  If non-null, use this pre-allocated buffer
   *   instead of allocating a new one. Caller retains ownership. */
  LowLatencyRuntime(
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      int rank,
      int numRanks,
      std::size_t numRdmaBytes,
      int numMaxDispatchTokensPerRank,
      int hidden,
      int numExperts,
      int numQpsPerRank,
      void* externalRdmaBuffer = nullptr);

  ~LowLatencyRuntime();

  LowLatencyRuntime(const LowLatencyRuntime&) = delete;
  LowLatencyRuntime& operator=(const LowLatencyRuntime&) = delete;
  LowLatencyRuntime(LowLatencyRuntime&&) = delete;
  LowLatencyRuntime& operator=(LowLatencyRuntime&&) = delete;

  int rank() const noexcept {
    return rank_;
  }
  int numRanks() const noexcept {
    return numRanks_;
  }
  int numExperts() const noexcept {
    return numExperts_;
  }
  int numLocalExperts() const noexcept {
    return numExperts_ / numRanks_;
  }
  int numMaxDispatchTokensPerRank() const noexcept {
    return numMaxDispatchTokensPerRank_;
  }

  const LowLatencyLayout& layout() const noexcept {
    return layout_;
  }

  /** Local LL buffer base pointer (shared with all peers via IBGDA
   *  symmetric heap). */
  void* getLocalRdmaBufferPtr() const noexcept {
    return rdmaBufferPtr_;
  }

  /** Persistent global atomic counter for combine grid_barrier. */
  int* getGlobalAtomicCounter() const noexcept {
    return globalAtomicCounter_;
  }

  /** Atomic counter array per local expert (numRanks × numLocalExperts). */
  int* getAtomicCounterPerExpert() const noexcept {
    return atomicCounterPerExpert_;
  }
  int* getAtomicFinishCounterPerExpert() const noexcept {
    return atomicFinishCounterPerExpert_;
  }

  /** Persistent buffer used by the next-iteration cleaner. */
  std::int64_t* getNextCleanBuffer() const noexcept {
    return nextCleanBuffer_;
  }
  int getNextCleanBufferIntCount() const noexcept {
    return nextCleanBufferIntCount_;
  }

  /** Workspace for combine kernel's atomic_clean_flag. */
  int* getCombineWorkspace() const noexcept {
    return combineWorkspace_;
  }

  /** Communication stream. */
  cudaStream_t commStream() const noexcept {
    return commStream_;
  }

  /** Device-side peer data pointer array for NVLink IPC writes. */
  void** getPeerDataPtrsDevice() const noexcept {
    return peerDataPtrsDevice_;
  }

  /** Set peer data pointers from the Buffer's IPC handles. Called by
   *  Buffer after sync() to wire the NVLink peer-mapped buffers. */
  void setPeerDataPtrs(const std::vector<void*>& peerPtrs);

  /**
   * Set up MultipeerIbgdaTransport for the cross-node IBGDA path.
   *
   * Constructs the transport with the given bootstrap, calls exchange()
   * (which uses bootstrap->allGather), registers the LL RDMA buffer, and
   * exchanges per-peer remote buffer descriptors for the four LL regions.
   *
   * Caller responsibility: provide a bootstrap that can perform the
   * required allGathers. PreExchangedBootstrap is the typical choice when
   * Python pre-gathers QP info via dist.all_gather_object.
   *
   * Idempotent in the sense that it must only be called once (transport
   * setup is destructive). Subsequent dispatch/combine calls will see
   * non-null device transport + per-peer remote buffers.
   *
   * Skipped silently when called on a single-rank setup.
   */
  void setupIbgda(std::shared_ptr<meta::comms::IBootstrap> bootstrap);

  /** Whether IBGDA is set up and the device transport is available. */
  bool hasIbgda() const noexcept {
    return ibgdaDeviceTransportPtr_ != nullptr;
  }

  /** Returns nullptr if IBGDA wasn't set up. */
  comms::prims::MultipeerIbgdaDeviceTransport* getIbgdaDeviceTransport()
      const noexcept {
    return ibgdaDeviceTransportPtr_;
  }

  /** Per-peer arrays (numRanks entries; self-rank entry zero-init). nullptr
   *  if IBGDA not set up. */
  const comms::prims::IbgdaLocalBuffer* getLocalRdmaXBufDevice()
      const noexcept {
    return ibgdaLocalRdmaXBufDevice_;
  }
  const comms::prims::IbgdaLocalBuffer* getLocalCombineXBufDevice()
      const noexcept {
    return ibgdaLocalCombineXBufDevice_;
  }
  const comms::prims::IbgdaRemoteBuffer* getPeerRemoteRecvXDevice()
      const noexcept {
    return ibgdaPeerRemoteRecvXDevice_;
  }
  const comms::prims::IbgdaRemoteBuffer* getPeerRemoteRecvCountDevice()
      const noexcept {
    return ibgdaPeerRemoteRecvCountDevice_;
  }
  const comms::prims::IbgdaRemoteBuffer* getPeerRemoteCombineRecvXDevice()
      const noexcept {
    return ibgdaPeerRemoteCombineRecvXDevice_;
  }
  const comms::prims::IbgdaRemoteBuffer* getPeerRemoteCombineRecvFlagDevice()
      const noexcept {
    return ibgdaPeerRemoteCombineRecvFlagDevice_;
  }

 private:
  const int rank_;
  const int numRanks_;
  const int numExperts_;
  const int numMaxDispatchTokensPerRank_;
  const int hidden_;
  const int numQpsPerRank_;
  const std::size_t numRdmaBytes_;
  LowLatencyLayout layout_;

  std::unique_ptr<comms::prims::MultipeerIbgdaTransport> ibgdaTransport_;
  // Device-resident IBGDA descriptors. Populated by setupIbgda().
  // Per-peer arrays are device-allocated copies of the corresponding host
  // vectors returned by transport.exchangeBuffer(). Indexed by peer rank
  // [0..numRanks); self-rank entry is zero-initialized.
  comms::prims::MultipeerIbgdaDeviceTransport* ibgdaDeviceTransportPtr_{
      nullptr};
  comms::prims::IbgdaLocalBuffer* ibgdaLocalRdmaXBufDevice_{nullptr};
  comms::prims::IbgdaLocalBuffer* ibgdaLocalCombineXBufDevice_{nullptr};
  comms::prims::IbgdaRemoteBuffer* ibgdaPeerRemoteRecvXDevice_{nullptr};
  comms::prims::IbgdaRemoteBuffer* ibgdaPeerRemoteRecvCountDevice_{nullptr};
  comms::prims::IbgdaRemoteBuffer* ibgdaPeerRemoteCombineRecvXDevice_{nullptr};
  comms::prims::IbgdaRemoteBuffer* ibgdaPeerRemoteCombineRecvFlagDevice_{
      nullptr};

  // The big LL data buffer — uncached on AMD (BNXT dma-buf parity).
  // If ownsRdmaBuffer_ is false, the buffer is externally owned.
  void* rdmaBufferPtr_{nullptr};
  bool ownsRdmaBuffer_{true};

  // Persistent atomic counters (live in workspace; persistent across
  // dispatch / combine invocations).
  int* globalAtomicCounter_{nullptr};
  int* atomicCounterPerExpert_{nullptr};
  int* atomicFinishCounterPerExpert_{nullptr};
  std::int64_t* nextCleanBuffer_{nullptr};

  // Device-side peer data pointer array for NVLink IPC writes
  void** peerDataPtrsDevice_{nullptr};
  int* combineWorkspace_{nullptr};
  int nextCleanBufferIntCount_{0};

  cudaStream_t commStream_{nullptr};
};

} // namespace comms::prims::moe_ep
