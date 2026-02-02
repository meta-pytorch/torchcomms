// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "comms/ctran/interfaces/IBootstrap.h"
#include "comms/pipes/IbgdaBuffer.h"

// Forward declarations for device types (defined in .cuh files)
namespace comms::pipes {
class P2pIbgdaTransportDevice;
struct MultipeerIbgdaDeviceTransport;
} // namespace comms::pipes

namespace comms::pipes {

/**
 * Configuration for MultipeerIbgdaTransport.
 *
 * IMPORTANT: All ranks must use identical configuration values.
 */
struct MultipeerIbgdaTransportConfig {
  // CUDA device index for GPU operations
  int cudaDevice{0};

  // Override NIC device name (e.g., "mlx5_0").
  // If empty, auto-discovers the NIC closest to the GPU.
  std::optional<std::string> nicDeviceName;

  // Override GID index for RoCE.
  // If not set, auto-discovers a valid RoCEv2 GID.
  std::optional<int> gidIndex;

  // Per-peer data buffer size in bytes.
  // This determines the maximum transfer size per put_signal call.
  std::size_t dataBufferSize{0};

  // Number of signal slots per peer.
  // Each slot is a 64-bit counter for signaling.
  std::size_t signalCount{1};

  // Queue pair depth (number of outstanding WQEs).
  // Higher values allow more pipelining but use more memory.
  uint32_t qpDepth{1024};
};

/**
 * Exchange information for RDMA connection setup.
 *
 * This struct is exchanged between peers during the bootstrap phase
 * to establish RDMA connectivity.
 */
struct IbgdaExchInfo {
  // Queue Pair Number for RDMA connection
  uint32_t qpn{0};

  // Remote key for RDMA access to data buffer
  uint32_t dataRkey{0};

  // Remote key for RDMA access to signal buffer
  uint32_t signalRkey{0};

  // Remote address of data buffer
  uint64_t dataAddr{0};

  // Remote address of signal buffer
  uint64_t signalAddr{0};

  // Global Identifier for RoCE routing (16 bytes)
  uint8_t gid[16]{};

  // GID index used
  int gidIndex{0};

  // Local Identifier (for IB, not used in RoCE)
  uint16_t lid{0};

  // Path MTU (4096 = IBV_MTU_4096)
  uint32_t mtu{4096};
};

/**
 * MultipeerIbgdaTransport - Host-side multi-peer RDMA transport manager
 *
 * Manages GPU-initiated RDMA (IBGDA) communication across multiple ranks using
 * DOCA GPUNetIO high-level APIs. This transport enables CUDA kernels to
 * directly issue RDMA operations without CPU involvement.
 *
 * ARCHITECTURE:
 * =============
 *
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │  Host Control Path                                                  │
 *   ├─────────────────────────────────────────────────────────────────────┤
 *   │  MultipeerIbgdaTransport (this class)                               │
 *   │  ├── IbvDevice (RDMA device management)                             │
 *   │  ├── IbvPd (Protection Domain)                                      │
 *   │  ├── IbvMr[] (Memory regions - data + signal per peer)              │
 *   │  ├── doca_gpu (GPU context for DOCA)                                │
 *   │  ├── doca_gpu_verbs_qp[] (High-level QPs per peer)                  │
 *   │  └── IBootstrap (Collective exchange)                               │
 *   ├─────────────────────────────────────────────────────────────────────┤
 *   │  GPU Data Path                                                      │
 *   ├─────────────────────────────────────────────────────────────────────┤
 *   │  MultipeerIbgdaDeviceTransport (returned by getDeviceTransport())   │
 *   │  └── P2pIbgdaTransportDevice[] (per-peer handles)                   │
 *   │      ├── doca_gpu_dev_verbs_qp* (GPU QP handle)                     │
 *   │      ├── IbgdaLocalBuffer (local signal buffer)                     │
 *   │      ├── IbgdaRemoteBuffer (remote signal buffer)                   │
 *   │      └── put_signal() / wait_signal() device methods                │
 *   └─────────────────────────────────────────────────────────────────────┘
 *
 * USAGE:
 * ======
 *
 *   // Host setup
 *   MultipeerIbgdaTransportConfig config{
 *       .cudaDevice = 0,
 *       .dataBufferSize = 1 << 20,  // 1 MB per peer
 *       .signalCount = 1,
 *   };
 *   MultipeerIbgdaTransport transport(myRank, nRanks, bootstrap, config);
 *   transport.exchange();  // Collective - all ranks must call
 *
 *   // Get device handle for kernel (requires including .cuh header)
 *   auto* deviceTransportPtr = transport.getDeviceTransportPtr();
 *
 * NIC AUTO-DISCOVERY:
 * ===================
 *
 * When nicDeviceName is not specified, the transport automatically selects
 * the RDMA NIC with the closest NUMA affinity to the specified GPU.
 *
 * COMMUNICATOR SEMANTICS:
 * =======================
 *
 * - Constructor: Local operation (allocates resources)
 * - exchange(): COLLECTIVE operation (all ranks must call)
 * - getDeviceTransportPtr(): Local operation (after exchange completes)
 */
class MultipeerIbgdaTransport {
 public:
  /**
   * Constructor - Initialize multi-peer IBGDA transport
   */
  MultipeerIbgdaTransport(
      int myRank,
      int nRanks,
      std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
      const MultipeerIbgdaTransportConfig& config);

  /**
   * Destructor - Release all resources
   */
  ~MultipeerIbgdaTransport();

  // Non-copyable, non-movable
  MultipeerIbgdaTransport(const MultipeerIbgdaTransport&) = delete;
  MultipeerIbgdaTransport& operator=(const MultipeerIbgdaTransport&) = delete;
  MultipeerIbgdaTransport(MultipeerIbgdaTransport&&) = delete;
  MultipeerIbgdaTransport& operator=(MultipeerIbgdaTransport&&) = delete;

  /**
   * exchange - Exchange connection info and connect QPs
   *
   * COLLECTIVE OPERATION: All ranks MUST call this before using
   * getDeviceTransportPtr().
   */
  void exchange();

  /**
   * getDeviceTransportPtr - Get pointer to device transport array
   *
   * Returns a pointer to the GPU memory containing the per-peer transport
   * handles. Each element corresponds to a peer (indexed by peer rank mapping).
   *
   * @return Pointer to P2pIbgdaTransportDevice array in GPU memory
   */
  P2pIbgdaTransportDevice* getDeviceTransportPtr() const;

  /**
   * Get number of peers (nRanks - 1)
   */
  int numPeers() const;

  /**
   * Get this rank's ID
   */
  int myRank() const;

  /**
   * Get total number of ranks
   */
  int nRanks() const;

  /**
   * getDataBuffer - Get local data buffer for a peer
   */
  IbgdaLocalBuffer getDataBuffer(int peerRank) const;

  /**
   * getRemoteDataBuffer - Get remote data buffer for a peer
   */
  IbgdaRemoteBuffer getRemoteDataBuffer(int peerRank) const;

  /**
   * Get the RDMA device name being used
   */
  std::string getNicDeviceName() const;

  /**
   * Get the GID index being used
   */
  int getGidIndex() const;

 private:
  // Pimpl idiom - implementation details are hidden
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace comms::pipes
