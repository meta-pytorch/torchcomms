// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "comms/ctran/ibverbx/Ibverbx.h"

#include <doca_gpunetio_host.h>
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

  // NOTE: Data buffers are NOT managed by the transport.
  // Users must allocate their own buffers and call registerBuffer() +
  // exchangeBuffer().

  // Number of signal slots per peer.
  // Each slot is a 64-bit counter for signaling.
  std::size_t signalCount{1};

  // Queue pair depth (number of outstanding WQEs).
  // Higher values allow more pipelining but use more memory.
  uint32_t qpDepth{1024};
};

/**
 * Transport connection information for RDMA QP setup.
 *
 * This struct is exchanged ONCE during the bootstrap phase to establish
 * RDMA connectivity between peers. Contains immutable connection parameters
 * that define how to reach a peer's QP.
 */
struct IbgdaTransportExchInfo {
  // Queue Pair Number for RDMA connection
  uint32_t qpn{0};

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
 * Combined exchange information for RDMA connection setup.
 *
 * This struct bundles transport and signal buffer information for convenient
 * exchange during the bootstrap phase.
 */
struct IbgdaExchInfo {
  IbgdaTransportExchInfo transport;
  IbgdaBufferExchInfo signal;
};

/**
 * Maximum number of ranks supported for allGather-based exchange.
 * This limit exists because we use fixed-size arrays for QPN exchange.
 */
constexpr int kMaxRanksForAllGather = 128;

/**
 * Transport exchange info for allGather-based exchange.
 *
 * Each rank contributes this structure containing:
 * - Common connection info (GID, lid, signal buffer) shared with all peers
 * - Per-target QPNs: qpnForRank[j] = QPN this rank uses to connect to rank j
 */
struct IbgdaTransportExchInfoAll {
  // Common info (same for all connections from this rank)
  uint8_t gid[16]{};
  int gidIndex{0};
  uint16_t lid{0};
  uint64_t signalAddr{0};
  HostRKey signalRkey{0};

  // Per-target-rank QPNs
  // qpnForRank[j] = QPN that this rank uses to connect to rank j
  // qpnForRank[myRank] is unused (set to 0)
  uint32_t qpnForRank[kMaxRanksForAllGather]{};
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
   * getDeviceTransport - Get multi-peer device transport wrapper
   *
   * Returns a MultipeerIbgdaDeviceTransport wrapper that provides convenient
   * access to per-peer transport handles with rank-to-index mapping.
   * Use .get(peerRank) to get the transport for a specific peer.
   *
   * NOTE: Requires including MultipeerIbgdaDeviceTransport.cuh in CUDA files.
   * For non-CUDA code, use getP2pTransportDevice(peerRank) instead.
   *
   * @return MultipeerIbgdaDeviceTransport wrapper (include .cuh header to use)
   */
  MultipeerIbgdaDeviceTransport getDeviceTransport() const;

  /**
   * getP2pTransportDevice - Get P2P transport for a specific peer rank
   *
   * Returns a pointer to the P2pIbgdaTransportDevice for the given peer rank.
   * This method handles the rank-to-index mapping internally and provides
   * explicit peer selection without requiring CUDA headers.
   *
   * @param peerRank Global rank of the peer (must be != myRank and < nRanks)
   * @return Pointer to P2pIbgdaTransportDevice for the specified peer
   */
  P2pIbgdaTransportDevice* getP2pTransportDevice(int peerRank) const;

  /**
   * getDeviceTransportPtr - Get pointer to device transport array
   *
   * Returns a pointer to the GPU memory containing the per-peer transport
   * handles. Each element corresponds to a peer (indexed by peer rank mapping).
   * Prefer getP2pTransportDevice(peerRank) for explicit peer selection.
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
   * registerBuffer - Register a user-provided buffer for RDMA access
   *
   * Registers the specified GPU memory buffer with the RDMA device, making it
   * accessible for RDMA operations. The buffer must remain valid until
   * deregisterBuffer() is called or the transport is destroyed.
   *
   * @param ptr Pointer to GPU memory (must be cudaMalloc'd or similar)
   * @param size Size of the buffer in bytes
   * @return IbgdaLocalBuffer with valid lkey for local RDMA operations
   * @throws std::runtime_error if registration fails
   */
  IbgdaLocalBuffer registerBuffer(void* ptr, std::size_t size);

  /**
   * deregisterBuffer - Deregister a previously registered buffer
   *
   * Releases the RDMA memory registration for the specified buffer.
   * The buffer pointer must match a previously registered buffer.
   *
   * @param ptr Pointer to the buffer to deregister
   */
  void deregisterBuffer(void* ptr);

  /**
   * exchangeBuffer - Collectively exchange buffer info with all peers
   *
   * COLLECTIVE OPERATION: All ranks MUST call this with their local buffer.
   * Returns remote buffer info for all peers, indexed by peer rank.
   *
   * @param localBuf Local buffer that was registered with registerBuffer()
   * @return Vector of remote buffers, one per peer (size = nRanks - 1)
   *         Index i corresponds to peerIndexToRank(i)
   */
  std::vector<IbgdaRemoteBuffer> exchangeBuffer(
      const IbgdaLocalBuffer& localBuf);

  /**
   * Get the RDMA device name being used
   */
  std::string getNicDeviceName() const;

  /**
   * Get the GID index being used
   */
  int getGidIndex() const;

 private:
  // Helper methods
  void initDocaGpu();
  void openIbDevice();
  void allocateResources();
  void registerMemory();
  void createQps();
  void connectQp(
      doca_gpu_verbs_qp_hl* qpHl,
      const IbgdaTransportExchInfo& peerInfo);
  int rankToPeerIndex(int rank) const;
  int peerIndexToRank(int peerIndex) const;

  // Rank information
  const int myRank_{-1};
  const int nRanks_{0};

  // Bootstrap for collective operations
  std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap_;

  // Configuration
  MultipeerIbgdaTransportConfig config_;

  // DOCA GPU context
  doca_gpu* docaGpu_{nullptr};

  // IB verbs resources (ibverbx RAII wrappers)
  std::optional<ibverbx::IbvDevice> ibvDevice_;
  std::optional<ibverbx::IbvPd> ibvPd_;
  doca_verbs_ah_attr* ahAttr_{nullptr};
  ibverbx::ibv_gid localGid_{};

  // High-level QPs (one per peer)
  std::vector<doca_gpu_verbs_qp_hl*> qpHlList_;

  // GPU memory (signal buffer is transport-managed)
  void* signalBuffer_{nullptr};
  std::size_t signalBufferSize_{0};

  // Memory regions for signal buffer
  std::optional<ibverbx::IbvMr> signalMr_;

  // User-registered buffers (maps ptr -> IbvMr)
  std::unordered_map<void*, ibverbx::IbvMr> registeredBuffers_;

  // GPU PCIe bus ID and NIC device name
  std::string gpuPciBusId_;
  std::string nicDeviceName_;
  int gidIndex_{3}; // Default GID index

  // Per-peer device transports (GPU accessible)
  P2pIbgdaTransportDevice* peerTransportsGpu_{nullptr};
  std::size_t peerTransportSize_{0};

  // Exchange info received from peers
  std::vector<IbgdaExchInfo> peerExchInfo_;
};

} // namespace comms::pipes
