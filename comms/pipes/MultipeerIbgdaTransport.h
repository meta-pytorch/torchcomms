// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <doca_gpunetio_host.h>
#include "doca_verbs_net_wrapper.h"

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/IbverbsLazy.h"
#include "comms/utils/CudaRAII.h"

// Forward declarations for device types (defined in .cuh files)
namespace comms::pipes {
class P2pIbgdaTransportDevice;
struct MultipeerIbgdaDeviceTransport;
} // namespace comms::pipes

namespace comms::pipes {

/**
 * IP address family for RoCE GID selection.
 * Similar to NCCL_IB_ADDR_FAMILY.
 */
enum class AddressFamily {
  IPV4, // IPv4
  IPV6, // IPv6
};

/**
 * Configuration for MultipeerIbgdaTransport.
 *
 * IMPORTANT: All ranks must use identical configuration values.
 */
struct MultipeerIbgdaTransportConfig {
  // CUDA device index for GPU operations
  int cudaDevice{0};

  // Override GID index for RoCE.
  // If not set, auto-discovers a valid RoCEv2 GID.
  std::optional<int> gidIndex;

  // IP address family for the InfiniBand GID (similar to NCCL_IB_ADDR_FAMILY).
  // Used to determine the address type for RoCE connections when gidIndex is
  // not explicitly set. Has no effect on InfiniBand (non-RoCE) links.
  // Default is IPV6 (IPv6).
  AddressFamily addressFamily{AddressFamily::IPV6};

  // NOTE: Data buffers are NOT managed by the transport.
  // Users must allocate their own buffers and call registerBuffer() +
  // exchangeBuffer().
  // GPU-to-NIC mapping for RDMA device selection.
  // Maps CUDA device index to a list of NIC names (first element is preferred).
  // If empty, uses topology-aware auto-discovery.
  std::map<int, std::vector<std::string>> gpuNicMap;

  // IB HCA filter string (NCCL_IB_HCA format) for NIC filtering during
  // auto-discovery. If empty, all discovered NICs are considered.
  // Only used during auto-discovery (not when gpuNicMap has a mapping for the
  // GPU).
  std::string ibHca;

  // Per-peer data buffer size in bytes.
  //
  // Raw put()/signal() users interpret this as the exported per-peer RDMA
  // buffer size. send()/recv() users interpret it as the size of one logical
  // staging slot. The send/recv ring therefore has:
  //   pipelineDepth slots
  //   each slot is dataBufferSize bytes
  //   each slot is partitioned across active_blocks block-groups at runtime
  //
  // For one send()/recv() call:
  //   perBlockSlot = (dataBufferSize / active_blocks) & ~15ULL
  //
  // In the benchmark, one "section" is exactly one dataBufferSize-sized slot.
  std::size_t dataBufferSize{0};

  // Number of signal slots managed by the transport (per peer).
  // Used by the slot-index API (put/signal/wait_signal by slot ID).
  // Independent of send/recv which uses its own private signal buffers.
  int numSignalSlots{0};

  // Number of counter slots managed by the transport (per peer).
  // Used by the slot-index API (wait_counter by slot ID).
  // Independent of send/recv which uses its own private counter buffers.
  int numCounterSlots{0};

  // Send/recv configuration. When set, the transport allocates a private
  // pipelined staging ring plus private signal/counter state for send()/recv().
  // When nullopt (default), send/recv is disabled and only the raw put/signal
  // APIs are available.
  struct SendRecvConfig {
    // Maximum number of block-groups that may participate in one send()/recv()
    // call. This sizes the private signal/counter/step arrays and defines the
    // maximum active_blocks accepted at runtime.
    int maxGroups{128};

    // Number of logical slots in the send/recv staging ring.
    // Total staging bytes per peer per direction:
    //   pipelineDepth * dataBufferSize
    int pipelineDepth{2};
  };
  std::optional<SendRecvConfig> sendRecv;

  // Queue pair depth (number of outstanding WQEs per peer).
  // Higher values allow more pipelining but use more memory.
  uint32_t qpDepth{1024};

  // Number of QP sets per peer (each set = main QP + companion QP + loopback).
  // Multiple QPs per peer allow different GPU blocks to use independent QPs,
  // eliminating O(N) cross-block WQE serialization in DOCA's mark_wqes_ready.
  // Block-to-QP mapping: blockIdx.x % numQpsPerPeer.
  // Default 1 preserves current single-QP-per-peer behavior.
  int numQpsPerPeer{1};

  // InfiniBand Verbs Timeout for QP ACK timeout.
  // Timeout is computed as 4.096 µs * 2^timeout.
  // Increasing this value can help on very large networks (e.g., if
  // ibv_poll_cq returns error 12). See InfiniBand specification Volume 1,
  // section 12.7.34 (Local Ack Timeout).
  // Valid values: 1-31. A value of 0 or >= 32 results in infinite timeout.
  // Default is 20 (similar to NCCL_IB_TIMEOUT).
  uint8_t timeout{20};

  // InfiniBand retry count for QP transport errors.
  // See InfiniBand specification Volume 1, section 12.7.38.
  // Default is 7 (similar to NCCL_IB_RETRY_CNT).
  uint8_t retryCount{7};

  // InfiniBand traffic class field (similar to NCCL_IB_TC).
  // See InfiniBand specification Volume 1 or vendor documentation.
  // Default is 224.
  uint8_t trafficClass{224};

  // InfiniBand Service Level (similar to NCCL_IB_SL).
  // See InfiniBand specification Volume 1, section 4.3.1.
  // Default is 0.
  uint8_t serviceLevel{0};

  // Minimum RNR NAK Timer field value (similar to ibv_qp_attr.min_rnr_timer).
  // Controls the delay before a receiver sends a RNR NAK.
  // See InfiniBand specification Volume 1, Table 46.
  // Default is 12 (matching NCCL IbvQpUtils).
  uint8_t minRnrTimer{12};

  // RNR retry count (similar to ibv_qp_attr.rnr_retry).
  // Number of times to retry after receiving an RNR NAK.
  // 7 means infinite retry.
  // Default is 7 (matching NCCL IbvQpUtils).
  uint8_t rnrRetry{7};
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

  // Port active MTU. Used to negotiate path MTU: min(local, remote).
  enum ibv_mtu mtu { IBV_MTU_4096 };
};

/**
 * Maximum number of ranks supported for allGather-based exchange.
 * This limit exists because we use fixed-size arrays for QPN exchange.
 */
constexpr int kMaxRanksForAllGather = 128;

/**
 * Maximum number of QP sets per peer for multi-QP support.
 */
constexpr int kMaxQpsPerPeer = 128;

/**
 * Transport exchange info for allGather-based exchange.
 *
 * Each rank contributes this structure containing:
 * - Common connection info (GID, lid) shared with all peers
 * - Per-target QPNs: qpnForRank[j] = QPN this rank uses to connect to rank j
 */
struct IbgdaTransportExchInfoAll {
  // Common info (same for all connections from this rank)
  uint8_t gid[16]{};
  int gidIndex{0};
  uint16_t lid{0};

  // Port active MTU.
  enum ibv_mtu mtu { IBV_MTU_4096 };

  // Number of QPs per peer used by this rank
  int numQpsPerPeer{1};

  // Per-target-rank QPNs
  // qpnForRank[j][q] = QPN of q-th QP that this rank uses to connect to rank j
  // qpnForRank[myRank][*] is unused (set to 0)
  uint32_t qpnForRank[kMaxRanksForAllGather][kMaxQpsPerPeer]{};
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
 *   │  ├── IbvMr[] (Memory regions - data per peer)                       │
 *   │  ├── doca_gpu (GPU context for DOCA)                                │
 *   │  ├── doca_gpu_verbs_qp[] (High-level QPs per peer)                  │
 *   │  └── IBootstrap (Collective exchange)                               │
 *   ├─────────────────────────────────────────────────────────────────────┤
 *   │  GPU Data Path                                                      │
 *   ├─────────────────────────────────────────────────────────────────────┤
 *   │  MultipeerIbgdaDeviceTransport (returned by getDeviceTransport())   │
 *   │  └── P2pIbgdaTransportDevice[] (per-peer handles)                   │
 *   │      ├── doca_gpu_dev_verbs_qp* (GPU QP handle)                     │
 *   │      └── put() / wait_local() device methods                        │
 *   └─────────────────────────────────────────────────────────────────────┘
 *
 * USAGE:
 * ======
 *
 *   // Host setup
 *   MultipeerIbgdaTransportConfig config{
 *       .cudaDevice = 0,
 *       .dataBufferSize = 1 << 20,  // 1 MB per peer
 *   };
 *   MultipeerIbgdaTransport transport(myRank, nRanks, bootstrap, config);
 *   transport.exchange();  // Collective - all ranks must call
 *
 *   // Get device handle for kernel (requires including .cuh header)
 *   auto* deviceTransportPtr = transport.getDeviceTransportPtr();
 *
 * NIC SELECTION:
 * ==============
 *
 * The transport selects the RDMA NIC in order of priority:
 * 1. config.gpuNicMap - explicit GPU-to-NIC mapping (map from GPU index to NIC
 * names)
 * 2. Auto-discovery - selects the NIC with closest NUMA affinity to the GPU
 *    (optionally filtered by config.ibHca allowlist)
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
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
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
   * Get the number of QP sets per peer
   */
  int numQpsPerPeer() const;

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
  void createQpGroups();
  void createLoopbackCompanionQps();
  void allocate_send_recv_buffers();
  void exchange_send_recv_buffers();
  void cleanup_send_recv_buffers();
  void cleanup();
  void connectQp(
      doca_gpu_verbs_qp_hl* qpHl,
      const IbgdaTransportExchInfo& peerInfo);
  int rankToPeerIndex(int rank) const;
  int peerIndexToRank(int peerIndex) const;

  // Rank information
  const int myRank_{-1};
  const int nRanks_{0};

  // Bootstrap for collective operations
  std::shared_ptr<meta::comms::IBootstrap> bootstrap_;

  // Configuration
  MultipeerIbgdaTransportConfig config_;

  // DOCA GPU context
  doca_gpu* docaGpu_{nullptr};

  // IB verbs resources (raw rdma-core)
  ibv_context* ibvCtx_{nullptr};
  ibv_pd* ibvPd_{nullptr};
  doca_verbs_ah_attr* ahAttr_{nullptr};
  union ibv_gid localGid_{};

  // QP groups (one per peer): main QP + companion QP with shared UAR.
  // The companion QP is created with core_direct=true (required for WAIT WQE).
  std::vector<doca_gpu_verbs_qp_group_hl*> qpGroupHlList_;

  // Self-loop responder companion QPs (one per peer, passive endpoints)
  // These are connected to the active companion QPs in each group,
  // forming loopback pairs for counter-based completion tracking.
  std::vector<doca_gpu_verbs_qp_hl*> loopbackCompanionQpHlList_;

  // Sink buffer for RDMA atomic return values (discarded).
  // DOCA's OPCODE_ATOMIC_FA requires a local address for the fetch-add
  // return value. We don't need it, so we use a small "sink" buffer.
  // Allocated via cuMemCreate with gpuDirectRDMACapable=1 so it can be
  // registered as an IB MR on all platforms (including aarch64/SMMU).
  void* sinkBuffer_{nullptr};
  std::size_t sinkBufferSize_{0};
  std::size_t sinkBufferAllocSize_{0};
  std::uint64_t sinkBufferHandle_{0};
  ibv_mr* sinkMr_{nullptr};

  // Cached MR entry: one MR per CUDA allocation, refcounted.
  // Multiple user buffers within the same cudaMalloc allocation share one MR.
  struct CachedMr {
    ibv_mr* mr;
    size_t allocSize;
    int refs;
  };

  // Maps CUDA allocation base address -> cached MR covering the full
  // allocation. Keyed by allocBase from cuMemGetAddressRange, not by user
  // pointer. Ordered map enables O(log n) containment lookup via
  // upper_bound — used by deregisterBuffer to find the owning allocation
  // without calling cuMemGetAddressRange (which fails if CUDA already freed
  // the memory).
  std::map<uintptr_t, CachedMr> registeredBuffers_;

  // GPU PCIe bus ID and NIC device name
  std::string gpuPciBusId_;
  std::string nicDeviceName_;
  int gidIndex_{3}; // Default GID index
  enum ibv_mtu localMtu_ { IBV_MTU_4096 };

  // Per-peer device transports (GPU accessible)
  P2pIbgdaTransportDevice* peerTransportsGpu_{nullptr};
  std::size_t peerTransportSize_{0};

  // All GPU allocations from buildDeviceTransportsOnGpu (freed in cleanup)
  std::vector<void*> gpuAllocations_;

  // Exchange info received from peers
  std::vector<IbgdaTransportExchInfo> peerExchInfo_;

  // Transport-owned signal buffers (allocated if numSignalSlots > 0)
  void* signalInboxGpu_{nullptr};
  std::vector<IbgdaRemoteBuffer> signalRemoteViews_;
  std::vector<IbgdaLocalBuffer> signalLocalViews_;

  // Transport-owned counter buffers (allocated if numCounterSlots > 0)
  void* counterGpu_{nullptr};
  std::vector<IbgdaLocalBuffer> counterViews_;

  void* discardSignalGpu_{nullptr};
  std::vector<IbgdaRemoteBuffer> discardSignalRemoteViews_;

  // Send/recv buffers (allocated when maxGroups > 0)
  struct SendRecvPeerBuffers {
    IbgdaLocalBuffer sendStaging;
    IbgdaLocalBuffer recvStaging;
    IbgdaLocalBuffer signal;
    IbgdaLocalBuffer counter;
    int64_t* stepState{nullptr};
    IbgdaRemoteBuffer remoteRecvStaging;
    IbgdaRemoteBuffer remoteSignal;
  };
  std::vector<SendRecvPeerBuffers> sendRecvPeerBuffers_;

  std::unique_ptr<meta::comms::DeviceBuffer> sendStagingBulk_;
  std::unique_ptr<meta::comms::DeviceBuffer> recvStagingBulk_;
  std::unique_ptr<meta::comms::DeviceBuffer> signalBulk_;
  std::unique_ptr<meta::comms::DeviceBuffer> counterBulk_;
  std::unique_ptr<meta::comms::DeviceBuffer> stepStateBulk_;

  IbgdaLocalBuffer recvStagingBulkReg_;
  IbgdaLocalBuffer signalBulkReg_;
};

} // namespace comms::pipes
