// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/common/IpcMemHandler.h"
#include "comms/ctran/interfaces/IBootstrap.h"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/utils/CudaRAII.h"

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
};

/**
 * Host-side multi-peer NVLink transport manager.
 *
 * Manages NVLink communication across multiple GPU ranks by:
 * 1. Allocating shared buffers with per-peer regions
 * 2. Exchanging IPC handles for direct GPU-to-GPU access
 * 3. Providing P2pNvlTransportDevice handles for CUDA kernels
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
   *
   * Does NOT exchange IPC handles - call exchange() after construction.
   *
   * @param myRank This rank's ID in the communicator (0 to nRanks-1)
   * @param nRanks Total number of ranks in the communicator
   * @param bootstrap Bootstrap interface for collective IPC handle exchange
   * @param multiPeerNvlTransportConfig Buffer configuration (must match across
   * all ranks)
   *
   * @throws std::runtime_error if CUDA buffer allocation fails
   */
  MultiPeerNvlTransport(
      int myRank,
      int nRanks,
      std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
      const MultiPeerNvlTransportConfig& multiPeerNvlTransportConfig);

  /**
   * exchange - Exchange IPC memory handles across all ranks
   *
   * COLLECTIVE OPERATION: All ranks MUST call this before using
   * getP2pTransportDevice().
   *
   * Performs collective IPC handle exchange using the bootstrap
   * interface:
   * 1. Each rank shares its local buffer's IPC handle with all other ranks
   * 2. Each rank receives IPC handles from all other ranks
   * 3. Implicit barrier ensures all ranks complete before returning
   *
   * After this call completes, all ranks can access each other's buffers via
   * NVLink using the handles obtained from getP2pTransportDevice().
   *
   * @throws May throw if IPC handle exchange fails or communicator errors occur
   */
  void exchange();

  /**
   * getP2pTransportDevice - Get device handle for P2P communication with a peer
   *
   * Returns a device-side transport handle configured for communication with
   * the specified peer rank. This handle can be passed to CUDA kernels.
   *
   * PRECONDITION: exchange() must have been called by all ranks first.
   *
   * The returned device handle contains:
   * - Local buffer pointers (this rank's buffers, accessible locally)
   * - Remote buffer pointers (peer's buffers, accessible via NVLink IPC)
   * - Configuration parameters (chunk size, pipeline depth, etc.)
   *
   * @param peerRank Target peer rank ID (must be in range [0, nRanks) and !=
   * myRank)
   * @return P2pNvlTransportDevice handle for use in CUDA kernels
   *
   * @note Thread-safe after exchange() completes
   * @note Can be called multiple times for the same or different peers
   * @note The returned handle is copyable and can be passed to multiple kernels
   */
  P2pNvlTransportDevice getP2pTransportDevice(int peerRank);

 private:
  const int myRank_{-1};
  const int nRanks_{-1};
  std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap_;
  const MultiPeerNvlTransportConfig config_;

  // data buffer: staging buffer for send/recv data
  // state buffer: buffer for signaling
  std::unique_ptr<meta::comms::DeviceBuffer> dataBuffer_d_;
  std::unique_ptr<meta::comms::DeviceBuffer> stateBuffer_d_;

  // TODO: refactor IpcMemHandler to handle multiple ipcHandles exchange and
  // merge into one IpcMemHandler
  std::unique_ptr<meta::comms::IpcMemHandler> dataBufferHandler_;
  std::unique_ptr<meta::comms::IpcMemHandler> stateBufferHandler_;

  // Per-peer buffer sizes for offset calculation
  std::size_t perPeerDataBufferSize_{0};
  std::size_t perPeerStateBufferSize_{0};
};

} // namespace comms::pipes
