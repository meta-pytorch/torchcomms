// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>
#include <optional>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/pipes/GpuMemHandler.h"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/utils/CudaRAII.h"

namespace comms::pipes {

// Forward declarations for multi-peer device transport types
struct Transport;

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

  // Number of P2P signal slots per peer for chunk-level pipeline coordination.
  // Used by signalBufferHandler_ and P2pNvlTransportDevice for send()/recv().
  // This is separate from WindowConfig.peerSignalCount (inbox model).
  // Typical: 1-num of blocks for most workloads.
  std::size_t p2pSignalCount{1};

  // If true, use dual chunk state buffers (one on each side) for local polling
  // on both sender and receiver. If false (default), use single chunk state
  // buffer on receiver side only.
  //
  // Single State Mode (default):
  //   - 1 ChunkState per chunk, stored on receiver side
  //   - Sender polls over NVLink (slower), receiver polls locally (faster)
  //   - Lower memory usage
  //
  // Dual State Mode:
  //   - 2 ChunkStates per chunk: receiverState (for data-ready signal),
  //     senderState (for ready-to-send signal)
  //   - Both sender and receiver poll locally (faster on both sides)
  //   - Higher memory usage, better performance for high-throughput workloads
  //
  // DUAL STATE MODE - STATE MACHINE:
  //   Sender: poll local senderState for READY_TO_SEND → send data →
  //           mark local senderState as UNREADY → signal receiver's
  //           receiverState
  //   Receiver: poll local receiverState for stepId → read data →
  //           mark local receiverState as UNREADY → signal sender's senderState
  //                                                  as READY_TO_SEND
  //
  // DUAL STATE MODE - STRIDED CHUNK ASSIGNMENT REQUIREMENT:
  //   Dual state mode MUST use for_each_item_strided for chunk
  //   distribution. The UNREADY state uses plain write + group.sync() for
  //   efficiency (st.release.gpu is too slow). This write is only visible
  //   within the same thread group. Strided ensures chunk K is ALWAYS
  //   assigned to group (K % total_groups), making the unready write visible
  //   after group.sync().
  bool useDualStateBuffer{false};

  // Size of LL128 packet buffer per peer (bytes).
  // When > 0, allocates LL128 buffers and enables ll128_send/recv/forward
  // on P2pNvlTransportDevice. When 0 (default), LL128 is disabled.
  // Use ll128_buffer_size() from Ll128Packet.cuh to compute from message size.
  std::size_t ll128BufferSize{0};
};

/**
 * Pre-exchanged data buffer pointers for reuse by MultiPeerNvlTransport.
 *
 * Both vectors are indexed by NVL local rank. The entry at self rank is
 * ignored (may be default-constructed / empty).
 *
 * - localBuffers[peerNvlRank]: this rank's data buffer region for receiving
 *   from peerNvlRank (the peer writes here via NVLink).
 * - remoteBuffers[peerNvlRank]: IPC-mapped pointer to peerNvlRank's data
 *   buffer region that this rank writes into via NVLink.
 *
 * SIZE REQUIREMENTS:
 *   Each per-peer buffer must be at least (pipelineDepth * dataBufferSize)
 *   bytes, matching the config passed to MultiPeerNvlTransport.
 *   setExternalDataBuffers() validates this and throws on mismatch.
 *
 * OWNERSHIP:
 *   The caller retains ownership of the underlying GPU memory. The buffers
 *   must remain valid and exclusively reserved for pipes' use for the
 *   lifetime of the MultiPeerNvlTransport. Pipes will read/write these
 *   buffers during send()/recv() operations — concurrent access by other
 *   subsystems will cause data corruption.
 */
struct ExternalStagingBuffers {
  std::vector<DeviceSpan<char>> localBuffers;
  std::vector<DeviceSpan<char>> remoteBuffers;
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
  // Non-copyable and non-movable (CUDA resources cannot be safely moved)
  MultiPeerNvlTransport(const MultiPeerNvlTransport&) = delete;
  MultiPeerNvlTransport& operator=(const MultiPeerNvlTransport&) = delete;
  MultiPeerNvlTransport(MultiPeerNvlTransport&&) = delete;
  MultiPeerNvlTransport& operator=(MultiPeerNvlTransport&&) = delete;

  /**
   * Constructor - Initialize multi-peer NVLink transport
   *
   * Allocates local GPU buffers for multi-peer communication:
   * - State buffer: for chunk-level synchronization states
   * - Signal buffer: for P2P signal coordination
   * - P2pNvlTransportDevice array: preallocated on device memory for all peers
   *   to allow stateful transport and reduce kernel launch latency (avoids
   *   per-launch H2D copy)
   *
   * Data buffers are NOT allocated in the constructor. They are either:
   * - Allocated internally in exchange() (default behavior)
   * - Provided externally via setExternalDataBuffers() before exchange()
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
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      const MultiPeerNvlTransportConfig& multiPeerNvlTransportConfig);

  /**
   * Destructor - Clean up CUDA resources
   *
   * Frees device memory allocated for Transport array.
   */
  ~MultiPeerNvlTransport() = default;

  /**
   * setExternalDataBuffers - Provide pre-exchanged data buffers
   *
   * Call this BEFORE exchange() to use externally-managed data buffers instead
   * of allocating internally. This enables reuse of data buffers that are
   * already IPC-shared (e.g., ctran's staging buffers from SharedResource).
   *
   * When external data buffers are set:
   * - exchange() will NOT allocate or exchange data buffers
   * - State and signal buffers are still allocated and exchanged internally
   * - buildP2pTransportDevice() uses the provided pointers for LocalState
   *   and RemoteState dataBuffer fields
   *
   * See ExternalStagingBuffers for size requirements and ownership semantics.
   */
  void setExternalDataBuffers(ExternalStagingBuffers externalStagingBuffers);

  /**
   * exchange - Exchange memory handles across all ranks
   *
   * COLLECTIVE OPERATION: All ranks MUST call this before using
   * getP2pTransportDevice().
   *
   * If setExternalDataBuffers() was called before exchange(), data buffers
   * are used as-is (no allocation or exchange). Otherwise, data buffers are
   * allocated internally and exchanged.
   *
   * In both cases:
   * 1. State and signal buffer handles are exchanged across all ranks
   * 2. Builds and copies P2pNvlTransportDevice for each peer to the
   *    preallocated device array (allocated in constructor)
   * 3. Implicit barrier ensures all ranks complete before returning
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
  P2pNvlTransportDevice getP2pTransportDevice(int peerRank);

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
   * getDeviceTransports - Get device-accessible array of Transport objects
   *
   * Returns a DeviceSpan of Transport objects indexed by global rank.
   * Each element is either a P2pNvlTransportDevice (for peer ranks) or a
   * P2pSelfTransportDevice (for myRank).
   *
   * PRECONDITION: exchange() must have been called by all ranks first.
   *
   * @return DeviceSpan<Transport> of size nRanks, indexed by global rank
   *
   * @note Thread-safe after exchange() completes
   * @note The span points to device memory owned by this transport
   */
  DeviceSpan<Transport> getDeviceTransports();

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
    return memSharingMode_;
  }

 private:
  // ==========================================================================
  // Private helpers
  // ==========================================================================

  // Initialize transports array on device (both P2P and SELF transports)
  void initializeTransportsArray();

  // ==========================================================================
  // Member variables
  // ==========================================================================

  const int myRank_{-1};
  const int nRanks_{-1};
  std::shared_ptr<meta::comms::IBootstrap> bootstrap_;
  const MultiPeerNvlTransportConfig config_;

  // GpuMemHandler-based memory for data, state, signal, and LL128 buffers
  // Automatically uses fabric handles on H100+/CUDA12.3+, falls back to cudaIpc
  // dataBufferHandler_ is only allocated when external data buffers are NOT
  // used. It is allocated lazily in exchange() rather than in the constructor.
  std::unique_ptr<GpuMemHandler> dataBufferHandler_;
  std::unique_ptr<GpuMemHandler> stateBufferHandler_;
  std::unique_ptr<GpuMemHandler> signalBufferHandler_;
  std::unique_ptr<GpuMemHandler>
      ll128BufferHandler_; // nullptr when ll128BufferSize == 0

  // External data buffer pointers (set via setExternalDataBuffers()).
  // When set, exchange() skips data buffer allocation/exchange.
  std::optional<ExternalStagingBuffers> externalStagingBuffers_;

  // Device-accessible Transport array for multi-peer transport
  // Allocated on device and populated in initializeTransportsArray()
  // Uses DeviceBuffer instead of GpuMemHandler since no exchange is needed
  std::unique_ptr<meta::comms::DeviceBuffer> transportsDevice_;

  // Per-peer buffer sizes for offset calculation
  std::size_t perPeerDataBufferSize_{0};
  std::size_t perPeerChunkStateBufferSize_{0};
  std::size_t perPeerSignalBufferSize_{0};
  std::size_t perPeerLl128BufferSize_{0};

  // Flag to track if multi-peer device arrays have been initialized
  bool multiPeerInitialized_{false};

  // Cached memory sharing mode (detected once in constructor)
  MemSharingMode memSharingMode_;
};

} // namespace comms::pipes
