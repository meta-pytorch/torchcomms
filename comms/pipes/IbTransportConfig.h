// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace comms::pipes {

enum class IbBackendMode {
  kIbgda,
  kIbrcProxy,
};

/**
 * IP address family for RoCE GID selection.
 * Similar to NCCL_IB_ADDR_FAMILY.
 */
enum class AddressFamily {
  IPV4, // IPv4
  IPV6, // IPv6
};

/**
 * Common InfiniBand transport configuration shared by the GPU-direct IBGDA
 * backend and the CPU-proxy IBRC backend.
 *
 * IMPORTANT: All ranks must use identical configuration values.
 */
struct MultipeerIbTransportConfig {
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

  // Number of QP sets per (peer, NIC). Each set = main QP + companion QP +
  // loopback. With multi-NIC, total QPs to a peer = numQpsPerPeerPerNic *
  // numNics. Multiple QPs allow different GPU blocks to use independent QPs,
  // eliminating O(N) cross-block WQE serialization in DOCA's mark_wqes_ready.
  // Block-to-QP mapping: blockIdx.x % numQpsPerPeerPerNic.
  // Default 1 preserves current single-QP-per-(peer, NIC) behavior.
  int numQpsPerPeerPerNic{1};

  // InfiniBand Verbs Timeout for QP ACK timeout.
  // Timeout is computed as 4.096 us * 2^timeout.
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

  // When true, defer per-peer state (QPs, staging, signal buffers) to
  // first use via materializePeer(). When false (default), allocate
  // eagerly at exchange() time.
  bool ibLazyConnect{false};

  // Timeout (ms) for the bilateral exchange in materializePeer().
  // On timeout, materializePeer throws rather than hanging.
  uint32_t materializePeerTimeoutMs{30000};
};

// Preserve the existing IBGDA-facing spelling while new IB backends share the
// neutral config type.
using MultipeerIbgdaTransportConfig = MultipeerIbTransportConfig;

} // namespace comms::pipes
