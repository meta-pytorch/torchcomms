// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/P2pIbgdaTransportState.h"

namespace comms::pipes {

// Forward declaration — avoid pulling DOCA headers
class MultipeerIbgdaTransport;

/**
 * Configuration for MultiPeerIbgdaTransportSetup.
 *
 * Controls staging buffer sizing and pipelining depth.
 * These mirror the NVLink config in MultiPeerNvlTransportConfig.
 */
struct MultiPeerIbgdaTransportSetupConfig {
  // Per-pipeline-slot staging buffer size in bytes
  size_t dataBufferSize{2048};

  // Chunk size for sub-pipeline RDMA puts (bytes)
  size_t chunkSize{512};

  // Number of pipeline slots for latency hiding
  int pipelineDepth{4};
};

/**
 * MultiPeerIbgdaTransportSetup - Host-side IBGDA staging + signal buffer
 * manager for device-initiated collectives.
 *
 * Allocates, registers, and exchanges ALL communication infrastructure
 * (staging data buffers, signal buffers) for IBGDA-based device collectives.
 * Fills the gap where MultiPeerNvlTransport handles this internally for
 * NVLink, but MultipeerIbgdaTransport follows a caller-provides-buffers model.
 *
 * Owned by MultiPeerTransport alongside the existing ibgdaTransport_.
 *
 * LIFETIME:
 *   - ibgdaTransport must outlive this object
 *   - All GPU memory is freed in the destructor
 *   - NIC registrations are deregistered in the destructor
 */
class MultiPeerIbgdaTransportSetup {
 public:
  /**
   * Constructor. Allocates a device-side iteration counter (zeroed).
   *
   * @param ibgdaTransport  Existing MultipeerIbgdaTransport (must outlive)
   * @param myRank          This rank's global ID
   * @param nRanks          Total number of ranks
   * @param config          Staging buffer and pipelining configuration
   * @param stream          CUDA stream for async memset operations
   */
  MultiPeerIbgdaTransportSetup(
      MultipeerIbgdaTransport& ibgdaTransport,
      int myRank,
      int nRanks,
      const MultiPeerIbgdaTransportSetupConfig& config,
      cudaStream_t stream = nullptr);

  ~MultiPeerIbgdaTransportSetup();

  // Non-copyable, non-movable (owns GPU memory)
  MultiPeerIbgdaTransportSetup(const MultiPeerIbgdaTransportSetup&) = delete;
  MultiPeerIbgdaTransportSetup& operator=(const MultiPeerIbgdaTransportSetup&) =
      delete;
  MultiPeerIbgdaTransportSetup(MultiPeerIbgdaTransportSetup&&) = delete;
  MultiPeerIbgdaTransportSetup& operator=(MultiPeerIbgdaTransportSetup&&) =
      delete;

  /**
   * Allocate staging + signal buffers, register with NIC, exchange with peers.
   *
   * COLLECTIVE — all ranks must call. Allocates:
   *   - Per-peer staging data buffers (pipelineDepth * dataBufferSize per peer)
   *   - Signal buffers (nRanks * 2 uint64_t slots: completion + back-pressure)
   * Registers all with ibgdaTransport and exchanges to get remote rkeys.
   */
  void exchangeBuffers();

  /**
   * Get the device-side iteration counter pointer.
   * Allocated and zeroed by the constructor. The kernel reads and increments
   * this to derive expected signal values. CUDA graph safe.
   */
  uint64_t* getIterationCounter() const;

  /**
   * Get the host-side per-peer state vector.
   * Used by MultiPeerTransport::build_device_handle() to construct
   * fully-formed P2pIbgdaTransportDevice objects with embedded staging state.
   *
   * @return Reference to host-side vector of per-peer staging state.
   */
  const std::vector<P2pIbgdaTransportState>& getHostPeerStates() const;

 private:
  MultipeerIbgdaTransport& ibgdaTransport_;
  int myRank_;
  int nRanks_;
  MultiPeerIbgdaTransportSetupConfig config_;
  cudaStream_t stream_;

  // Staging data buffers (owned)
  void* stagingBuffer_{nullptr};
  IbgdaLocalBuffer localStagingBuf_;
  std::vector<IbgdaRemoteBuffer> remoteStagingBufs_;

  // Signal buffers (owned)
  void* signalBuffer_{nullptr};
  IbgdaLocalBuffer localSignalBuf_;
  std::vector<IbgdaRemoteBuffer> remoteSignalBufs_;

  // Device-side iteration counter (owned)
  uint64_t* d_iterationCounter_{nullptr};

  // Host-side peer state vector (saved from exchangeBuffers for
  // buildP2pTransportDevice)
  std::vector<P2pIbgdaTransportState> h_peerStates_;
};

} // namespace comms::pipes
