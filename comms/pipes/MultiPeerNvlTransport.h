// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/pipes/GpuMemHandler.h"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes {

/**
 * Configuration for multi-peer NVLink transport.
 *
 * IMPORTANT: All ranks must use identical configuration values.
 *
 * Memory per rank = (nRanks - 1) × pipelineDepth × dataBufferSize
 */
struct MultiPeerNvlTransportConfig {
  // Size of staging buffer per pipeline slot (bytes).
  // Larger transfers split into multiple steps of this size.
  // Typical: 1-256 MB depending on message sizes.
  std::size_t dataBufferSize{0};

  // Chunk size for parallel processing (bytes).
  // Smaller = more parallelism, larger = less sync overhead.
  // Should be multiple of 16 bytes. Typical: 8 KB - 4 MB.
  std::size_t chunkSize{0};

  // Number of pipeline slots for overlapping communication.
  // Higher = better latency hiding but more memory.
  // Typical: 2-4 for most workloads.
  std::size_t pipelineDepth{0};

  // Number of signal slots per peer for signal/wait communication.
  // Larger = more signal available for parallelism.
  // Typical: 1-num of block for most workloads.
  std::size_t signalCount{1};
};

/**
 * Host-side multi-peer NVLink transport manager.
 *
 * Manages NVLink communication across multiple GPU ranks by:
 * 1. Allocating shared buffers with per-peer regions
 * 2. Exchanging memory handles for direct GPU-to-GPU access
 * 3. Providing P2pNvlTransportDevice handles for CUDA kernels
 *
 * MEMORY SHARING MODES (automatic fallback):
 * - Fabric handles (H100+, CUDA 12.3+): Enables multi-node NVLink on GB200
 * - cudaIpcMemHandle (fallback): Works on all CUDA GPUs, intra-node only
 *
 * The mode is automatically detected at construction time. Use
 * getMemSharingMode() to check which mode is active.
 *
 * COMMUNICATOR SEMANTICS:
 * - Constructor and exchange() are COLLECTIVE operations (all ranks must call)
 * - All ranks must use identical configuration
 * - exchange() must complete on all ranks before getP2pTransportDevice()
 *
 * USAGE:
 *   MultiPeerNvlTransport transport(myRank, nRanks, bootstrap, config);
 *   transport.exchange();  // Collective
 *   auto device = transport.getP2pTransportDevice(peerRank);
 *   myKernel<<<...>>>(device, data, size);
 *
 * BUFFER LAYOUT (4 ranks):
 *   Rank 0: [peer1 | peer2 | peer3]
 *   Rank 1: [peer0 | peer2 | peer3]
 *   Rank 2: [peer0 | peer1 | peer3]
 *   Rank 3: [peer0 | peer1 | peer2]
 */
class MultiPeerNvlTransport {
 public:
  /**
   * Constructor - Initialize multi-peer NVLink transport
   *
   * Allocates local GPU buffers for multi-peer communication:
   * - Data buffer: for staging data transfers (nRanks-1 peer regions)
   * - State buffer: for chunk-level synchronization states
   * - P2pNvlTransportDevice array: preallocated on device memory for all peers
   *   to allow stateful tranport and reduce kernel launch latency (avoids
   *   per-launch H2D copy)
   *
   * Memory sharing mode is automatically detected:
   * - Fabric handles (H100+, CUDA 12.3+): Enables GB200 multi-node NVLink
   * - cudaIpcMemHandle (fallback): Works on all CUDA GPUs, intra-node only
   *
   * Does NOT exchange memory handles - call exchange() after construction.
   * The preallocated P2pNvlTransportDevice array is populated in exchange()
   * after peer buffer pointers are available.
   *
   * @param myRank This rank's ID in the communicator (0 to nRanks-1)
   * @param nRanks Total number of ranks in the communicator
   * @param bootstrap Bootstrap interface for collective handle exchange
   * @param multiPeerNvlTransportConfig Buffer configuration (must match across
   * all ranks)
   *
   * @throws std::runtime_error if buffer allocation fails
   */
  MultiPeerNvlTransport(
      int myRank,
      int nRanks,
      std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
      const MultiPeerNvlTransportConfig& multiPeerNvlTransportConfig);

  /**
   * Destructor - Cleanup device memory
   *
   * Frees the preallocated P2pNvlTransportDevice array on device memory.
   */
  ~MultiPeerNvlTransport();

  // Non-copyable and non-movable
  MultiPeerNvlTransport(const MultiPeerNvlTransport&) = delete;
  MultiPeerNvlTransport& operator=(const MultiPeerNvlTransport&) = delete;
  MultiPeerNvlTransport(MultiPeerNvlTransport&&) = delete;
  MultiPeerNvlTransport& operator=(MultiPeerNvlTransport&&) = delete;

  /**
   * exchange - Exchange memory handles across all ranks
   *
   * COLLECTIVE OPERATION: All ranks MUST call this before using
   * getP2pTransportDevice().
   *
   * Performs collective handle exchange using the bootstrap interface:
   * 1. Each rank shares its local buffer's handle with all other ranks
   * 2. Each rank receives handles from all other ranks
   * 3. Builds and copies P2pNvlTransportDevice for each peer to the
   *    preallocated device array (allocated in constructor)
   * 4. Implicit barrier ensures all ranks complete before returning
   *
   * The type of handle (fabric or cudaIpc) depends on the automatically
   * detected memory sharing mode.
   *
   * After this call completes, all ranks can access each other's buffers via
   * NVLink using the handles obtained from getP2pTransportDevice().
   *
   * @throws May throw if handle exchange fails or communicator errors occur
   */
  void exchange();

  /**
   * getP2pTransportDevice - Get device handle for P2P communication with a peer
   *
   * Returns a pointer to the device-side transport handle configured for
   * communication with the specified peer rank. This handle can be passed to
   * CUDA kernels. The P2pNvlTransportDevice is preallocated on device memory
   * in the constructor to reduce kernel launch latency.
   *
   * PRECONDITION: exchange() must have been called by all ranks first.
   *
   * The returned device handle contains:
   * - Local buffer pointers (this rank's buffers, accessible locally)
   * - Remote buffer pointers (peer's buffers, accessible via NVLink)
   * - Configuration parameters (chunk size, pipeline depth, etc.)
   *
   * @param peerRank Target peer rank ID (must be in range [0, nRanks) and !=
   * myRank)
   * @return Pointer to P2pNvlTransportDevice on device memory for use in CUDA
   * kernels
   *
   * @note Thread-safe after exchange() completes
   * @note Can be called multiple times for the same or different peers
   * @note The returned pointer is valid until this MultiPeerNvlTransport is
   * destroyed
   */
  P2pNvlTransportDevice* getP2pTransportDevice(int peerRank);

  /**
   * getTransportsArray - Get preallocated Transport array on device memory
   *
   * Returns a pointer to the device-side Transport array containing all
   * transport handles (P2pSelfTransportDevice for self, P2pNvlTransportDevice
   * for all peers). This array can be passed directly to CUDA kernels for
   * all-to-all communication patterns without per-launch H2D copies.
   *
   * PRECONDITION: exchange() must have been called by all ranks first.
   *
   * Array layout: [Transport for rank 0, Transport for rank 1, ..., Transport
   * for rank nRanks-1]
   * - transports_d_[myRank_] contains P2pSelfTransportDevice
   * - transports_d_[peerRank] contains P2pNvlTransportDevice for peer
   *
   * @return Pointer to Transport array on device memory (size = nRanks)
   *
   * @note Thread-safe after exchange() completes
   * @note The returned pointer is valid until this MultiPeerNvlTransport is
   * destroyed
   */
  Transport* getTransportsArray();

  /**
   * getNRanks - Get total number of ranks
   *
   * @return Total number of ranks in the communicator
   */
  int getNRanks() const {
    return nRanks_;
  }

  /**
   * buildP2pTransportDevice - Build host-side P2pNvlTransportDevice
   *
   * Constructs a P2pNvlTransportDevice on the host for the specified peer.
   *
   * This method is used in tests to access const members and when building
   * Transport objects on the host side.
   *
   * PRECONDITION: exchange() must have been called first (for external use).
   *
   * @param peerRank Target peer rank ID (must be in range [0, nRanks) and !=
   * myRank)
   * @return P2pNvlTransportDevice constructed on the host
   */
  P2pNvlTransportDevice buildP2pTransportDevice(int peerRank);

  /**
   * Check if fabric-based transport is supported (H100+, CUDA 12.3+).
   *
   * Note: Even if this returns false, the transport will still work using
   * cudaIpcMemHandle fallback (intra-node only).
   *
   * @return true if fabric handles are available for multi-node NVLink
   */
  static bool isFabricSupported() {
    return GpuMemHandler::isFabricHandleSupported();
  }

  /**
   * Get the memory sharing mode being used by this transport.
   *
   * @return kFabric for H100+/CUDA12.3+, kCudaIpc for fallback
   */
  MemSharingMode getMemSharingMode() const {
    return dataBufferHandler_->getMode();
  }

 private:
  const int myRank_{-1};
  const int nRanks_{-1};
  std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap_;
  const MultiPeerNvlTransportConfig config_;

  // GpuMemHandler-based memory for data, state, and signal buffers
  // Automatically uses fabric handles on H100+/CUDA12.3+, falls back to cudaIpc
  std::unique_ptr<GpuMemHandler> dataBufferHandler_;
  std::unique_ptr<GpuMemHandler> stateBufferHandler_;
  std::unique_ptr<GpuMemHandler> signalBufferHandler_;

  // Per-peer buffer sizes for offset calculation
  std::size_t perPeerDataBufferSize_{0};
  std::size_t perPeerChunkStateBufferSize_{0};
  std::size_t perPeerSignalBufferSize_{0};

  // Preallocated Transport array on device memory
  // Indexed by rank: transports_d_[myRank_] contains P2pSelfTransportDevice,
  // transports_d_[peerRank] contains P2pNvlTransportDevice for that peer.
  // Allocated in constructor, populated in exchange() after peer pointers
  // available
  Transport* transports_d_{nullptr};
};

} // namespace comms::pipes
