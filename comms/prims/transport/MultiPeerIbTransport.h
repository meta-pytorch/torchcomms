// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <folly/Synchronized.h>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/prims/memory/DeviceSpan.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/prims/transport/rdma/DataDirectMode.h"
#include "comms/utils/memtrace/GpuMemoryTracker.h"

namespace meta::comms {
class DeviceBuffer;
} // namespace meta::comms

namespace comms::prims {

/**
 * IP address family for RoCE GID selection (similar to NCCL_IB_ADDR_FAMILY).
 */
enum class AddressFamily {
  IPV4, // IPv4
  IPV6, // IPv6
};

/**
 * The *resolved* NIC data-placement ordering mode for an IB QP (mlx5 QPC
 * dp_ordering) -- what actually gets written to the QPC, after the requested
 * IbQpOrderingPolicy below has been checked against the NIC.
 *
 * The NIC encodes this as a 2-bit tier plus a "force" bit that overrides the
 * HCA's mlxreg adaptive-routing admin default:
 *   tier 0 IBTA    - strict IBTA ordering
 *   tier 1 OOO_RW  - out-of-order placement for RDMA Read/Write
 *   tier 2 OOO_ALL - out-of-order placement for all supported opcodes
 * This enum flattens the (tier, force) pair into the four combinations worth
 * selecting; ibQpOrderingTier()/ibQpOrderingForce() below split it back out.
 *
 * Ibta leaves both QPC bits and the force bit at zero, which is the pre-patch
 * wire behaviour. IbtaForced is deliberately distinct: it pins the QP to strict
 * ordering even when the NIC's admin default enables adaptive routing, which is
 * how you find out whether a QP is being sprayed today.
 *
 * Note that Ibta does NOT mean "the QP runs strict IBTA". It means "we wrote
 * nothing and the firmware chose". On a ConnectX-8 rail with adaptive routing
 * enabled the firmware chooses OOO_RW on its own, so Ibta and OooRw describe
 * the same QP there; on ConnectX-7 (AR off) Ibta really is tier 0.
 */
enum class IbQpOrderingSemantic {
  Ibta, // tier IBTA, force off (default; no QPC write at all)
  IbtaForced, // tier IBTA, force on
  OooRw, // tier OOO_RW, force on
  OooAll, // tier OOO_ALL, force on
};

// The 2-bit dp_ordering tier the NIC expects (matches DOCA's
// doca_verbs_qp_ordering_semantic and UCX's uct_ib_mlx5_dp_ordering_t).
constexpr int ibQpOrderingTier(IbQpOrderingSemantic mode) {
  switch (mode) {
    case IbQpOrderingSemantic::OooRw:
      return 1;
    case IbQpOrderingSemantic::OooAll:
      return 2;
    case IbQpOrderingSemantic::Ibta:
    case IbQpOrderingSemantic::IbtaForced:
      break;
  }
  return 0;
}

// Whether the QPC dp_ordering_force bit is set (requires HCA
// cmd_hca_cap_2.dp_ordering_force).
constexpr bool ibQpOrderingForce(IbQpOrderingSemantic mode) {
  return mode != IbQpOrderingSemantic::Ibta;
}

// True when the mode writes nothing to the QPC, i.e. is a wire-level no-op and
// the QP keeps whatever tier the firmware picked for it.
constexpr bool ibQpOrderingIsWireNoOp(IbQpOrderingSemantic mode) {
  return mode == IbQpOrderingSemantic::Ibta;
}

constexpr const char* ibQpOrderingSemanticName(IbQpOrderingSemantic mode) {
  switch (mode) {
    case IbQpOrderingSemantic::IbtaForced:
      return "ibta_forced";
    case IbQpOrderingSemantic::OooRw:
      return "ooo_rw";
    case IbQpOrderingSemantic::OooAll:
      return "ooo_all";
    case IbQpOrderingSemantic::Ibta:
      break;
  }
  return "ibta";
}

// Name for an ordering mode that arrived over the wire from a peer, which may
// be out of range if that peer is running a different build. Never casts the
// value to the enum, so an unknown value is reported rather than aliased onto a
// real mode.
constexpr const char* ibQpOrderingSemanticNameFromWire(int value) {
  switch (value) {
    case static_cast<int>(IbQpOrderingSemantic::Ibta):
      return ibQpOrderingSemanticName(IbQpOrderingSemantic::Ibta);
    case static_cast<int>(IbQpOrderingSemantic::IbtaForced):
      return ibQpOrderingSemanticName(IbQpOrderingSemantic::IbtaForced);
    case static_cast<int>(IbQpOrderingSemantic::OooRw):
      return ibQpOrderingSemanticName(IbQpOrderingSemantic::OooRw);
    case static_cast<int>(IbQpOrderingSemantic::OooAll):
      return ibQpOrderingSemanticName(IbQpOrderingSemantic::OooAll);
    default:
      break;
  }
  return "unknown";
}

// Parse the MCCL_IBGDA_QP_ORDERING_SEMANTIC spelling. Returns nullopt on an
// unrecognized value so the caller can fail loudly instead of silently running
// the default arm of an A/B.
inline std::optional<IbQpOrderingSemantic> parseIbQpOrderingSemantic(
    const std::string& value) {
  if (value == "ibta") {
    return IbQpOrderingSemantic::Ibta;
  }
  if (value == "ibta_forced") {
    return IbQpOrderingSemantic::IbtaForced;
  }
  if (value == "ooo_rw") {
    return IbQpOrderingSemantic::OooRw;
  }
  if (value == "ooo_all") {
    return IbQpOrderingSemantic::OooAll;
  }
  return std::nullopt;
}

/**
 * What the *caller* asks for, before the NIC gets a say. Resolved into an
 * IbQpOrderingSemantic at transport init by consulting the HCA capabilities.
 *
 * The distinction exists because the two kinds of request want opposite
 * failure behaviour:
 *
 *   Auto      - a fleet default. Walks OooAll -> OooRw -> Ibta and takes the
 *               strongest tier the NIC reports, silently, falling all the way
 *               to Ibta when the capability query fails outright (a non-mlx5
 *               NIC has no such query). A default must never be the reason a
 *               job refuses to start.
 *   explicit  - somebody is running an experiment and named a tier. Fail
 *               closed if the NIC cannot deliver it, because a silent
 *               downgrade turns an A/B into a no-op that reads as "OOO does
 *               not help". NVIDIA's GDAKI refuses here for the same reason.
 *
 * Auto aims at OooAll because that is the rung that pays: it is the only tier
 * that lifts the ordering fence in front of an atomic, and every prims signal
 * is an ATOMIC_FETCH_AND_ADD. OooRw relaxes Read/Write placement only, leaves
 * that fence standing, and on a ConnectX-8 rail is close to a no-op anyway --
 * the firmware already programs OOO_RW at INIT2RTR under adaptive routing with
 * nothing in userspace asking for it.
 *
 * The ladder is not decoration. ConnectX-8 reports OooAll; ConnectX-7 reports
 * OooRw but not OooAll (measured: cmd_hca_cap.dp_ordering_ooo_all_rc = 0), so a
 * flat "ask for OooAll" would fail closed on every CX7 NIC. Ranks that resolve
 * different tiers are rejected at peer connect, so a job spanning both NIC
 * generations has to pin ooo_rw explicitly.
 */
enum class IbQpOrderingPolicy {
  Auto, // strongest tier the NIC supports: OooAll, else OooRw, else Ibta
  Ibta, // explicit: write nothing, keep the firmware default
  IbtaForced, // explicit: pin to strict IBTA even under adaptive routing
  OooRw, // explicit: out-of-order placement for RDMA Read/Write
  OooAll, // explicit: also covers atomics
};

constexpr bool ibQpOrderingPolicyIsAuto(IbQpOrderingPolicy policy) {
  return policy == IbQpOrderingPolicy::Auto;
}

// The tier an explicit policy names. Auto has no fixed answer -- it depends on
// the NIC -- so it is not accepted here; callers must go through the resolver.
constexpr IbQpOrderingSemantic ibQpOrderingPolicyToSemantic(
    IbQpOrderingPolicy policy) {
  switch (policy) {
    case IbQpOrderingPolicy::IbtaForced:
      return IbQpOrderingSemantic::IbtaForced;
    case IbQpOrderingPolicy::OooRw:
      return IbQpOrderingSemantic::OooRw;
    case IbQpOrderingPolicy::OooAll:
      return IbQpOrderingSemantic::OooAll;
    case IbQpOrderingPolicy::Auto:
    case IbQpOrderingPolicy::Ibta:
      break;
  }
  return IbQpOrderingSemantic::Ibta;
}

constexpr const char* ibQpOrderingPolicyName(IbQpOrderingPolicy policy) {
  switch (policy) {
    case IbQpOrderingPolicy::Auto:
      return "auto";
    case IbQpOrderingPolicy::IbtaForced:
      return "ibta_forced";
    case IbQpOrderingPolicy::OooRw:
      return "ooo_rw";
    case IbQpOrderingPolicy::OooAll:
      return "ooo_all";
    case IbQpOrderingPolicy::Ibta:
      break;
  }
  return "ibta";
}

// Parse the MCCL_IBGDA_QP_ORDERING_SEMANTIC spelling into a policy. Returns
// nullopt on an unrecognized value so a typo cannot silently select the
// default.
inline std::optional<IbQpOrderingPolicy> parseIbQpOrderingPolicy(
    const std::string& value) {
  if (value == "auto") {
    return IbQpOrderingPolicy::Auto;
  }
  if (value == "ibta") {
    return IbQpOrderingPolicy::Ibta;
  }
  if (value == "ibta_forced") {
    return IbQpOrderingPolicy::IbtaForced;
  }
  if (value == "ooo_rw") {
    return IbQpOrderingPolicy::OooRw;
  }
  if (value == "ooo_all") {
    return IbQpOrderingPolicy::OooAll;
  }
  return std::nullopt;
}

/**
 * Shared configuration for the multi-peer IB transports (IBGDA, IBRC). Every
 * field is backend-agnostic IB transport config. IMPORTANT: all ranks must use
 * identical configuration values.
 */
struct MultipeerIbTransportConfig {
  // CUDA device index for GPU operations
  int cudaDevice{0};

  // Override GID index for RoCE. If not set, auto-discovers a valid RoCEv2 GID.
  std::optional<int> gidIndex;

  // IP address family for the InfiniBand GID (similar to NCCL_IB_ADDR_FAMILY).
  // Used to determine the address type for RoCE connections when gidIndex is
  // not explicitly set. Has no effect on InfiniBand (non-RoCE) links.
  AddressFamily addressFamily{AddressFamily::IPV6};

  // GPU-to-NIC mapping for RDMA device selection. Maps CUDA device index to a
  // list of NIC names (first element is preferred). If empty, uses
  // topology-aware auto-discovery. (Data buffers are NOT managed by the
  // transport; users allocate and register them.)
  std::map<int, std::vector<std::string>> gpuNicMap;

  // IB HCA filter string (NCCL_IB_HCA format) for NIC filtering during
  // auto-discovery. If empty, all discovered NICs are considered. Only used
  // during auto-discovery (not when gpuNicMap has a mapping for the GPU).
  std::string ibHca;

  // Per-peer data buffer size in bytes for raw put()/signal() users. When
  // perChannelSize is set for send()/recv(), the transport derives this as the
  // total fixed-channel staging size:
  //   perChannelSize * max_num_channels
  std::size_t dataBufferSize{0};

  // Fixed-channel send/recv staging window size in bytes for one channel. When
  // this is nonzero, pipelineDepth is the number of chunks within this channel
  // window and dataBufferSize is derived as the total staging size across all
  // channels.
  std::size_t perChannelSize{0};

  // Maximum number of logical IB channels per peer in the fixed-channel model.
  // A channel is selected by ThreadGroup::group_id. For IB, QP ownership is
  // scoped by (channel, direction, NIC).
  int max_num_channels{64};

  // Fixed-channel send/recv slots/chunks per channel.
  int pipelineDepth{2};

  // Number of signal slots managed by the transport (per peer), for the
  // slot-index API. Independent of send/recv's private signal buffers.
  int numSignalSlots{0};

  // Number of counter slots managed by the transport (per peer), for the
  // slot-index API. Independent of send/recv's private counter buffers.
  int numCounterSlots{0};

  // Maximum number of physical block groups that may own IB QP resources.
  // Device-side IB QP selection uses ThreadGroup::block_id and requires
  // block_id < maxGroups.
  int maxGroups{64};

  // Legacy block-owned QP count for IBRC. IBGDA send/recv uses
  // qpsPerConnection with the fixed-channel helpers below; IBRC moves to the
  // fixed-channel shape in the following stack diff.
  int qpsPerBlockPerNic{1};

  // Queue pair depth (outstanding WQEs per peer). BNXT bumps the default
  // because qpDepth also sizes msn_tbl_sz on bnxt_re.
#ifdef NIC_BNXT
  uint32_t qpDepth{2048};
#else
  uint32_t qpDepth{1024};
#endif

  // Number of main QPs per fixed-channel IB connection, where one connection is
  // a (channel, direction, NIC) tuple. IBGDA companion QPs use the same slot
  // geometry as main QPs because device-side lane selection indexes both with
  // qpsPerConnection.
  int qpsPerConnection{1};

  // IBGDA-only reliable-doorbell policy; ignored by IBRC and AMD. nullopt
  // auto-detects NIC support, true requires support, and false disables it.
  std::optional<bool> enableReliableDoorbell;

  // IBGDA-only collapsed-CQ policy; ignored by IBRC and AMD. nullopt probes
  // every selected NIC and enables the format only when all accept it, true
  // requires every NIC to accept it, and false forces ordinary ring CQs.
  std::optional<bool> enableCollapsedCq;

  int numQpsPerPeerPerNic() const {
    if (maxGroups < 0 || qpsPerBlockPerNic < 0) {
      throw std::invalid_argument(
          "maxGroups and qpsPerBlockPerNic must be >= 0");
    }
    if (maxGroups != 0 &&
        qpsPerBlockPerNic > std::numeric_limits<int>::max() / maxGroups) {
      throw std::overflow_error("maxGroups * qpsPerBlockPerNic overflows int");
    }
    return maxGroups * qpsPerBlockPerNic;
  }

  // Slot-indexed storage is reserved per (logical channel, protocol slot).
  // max_num_channels stays the LOGICAL channel count a caller selects with
  // group_id; slot p owns [p * max_num_channels, (p+1) * max_num_channels).
  // The slot count is kNumProtoSlots (IbgdaBuffer.h) rather than runtime
  // config, so host sizing and device indexing cannot disagree. QPs are NOT
  // multiplied: a channel is one QP pair shared by every protocol on it.
  int totalChannelSlots() const {
    if (max_num_channels < 0) {
      throw std::invalid_argument("max_num_channels must be >= 0");
    }
    if (max_num_channels > std::numeric_limits<int>::max() / kNumProtoSlots) {
      throw std::overflow_error(
          "max_num_channels * kNumProtoSlots overflows int");
    }
    return max_num_channels * kNumProtoSlots;
  }

  std::size_t fixedChannelDataBufferSize() const {
    const auto channels = static_cast<std::size_t>(totalChannelSlots());
    if (channels != 0 &&
        perChannelSize > std::numeric_limits<std::size_t>::max() / channels) {
      throw std::overflow_error(
          "perChannelSize * totalChannelSlots overflows size_t");
    }
    return perChannelSize * channels;
  }

  int fixedChannelMainQpsPerPeerPerNic() const {
    if (max_num_channels < 0 || qpsPerConnection < 0) {
      throw std::invalid_argument(
          "max_num_channels and qpsPerConnection must be >= 0");
    }
    const int directionCount = fixedChannelDirectionCount();
    if (max_num_channels != 0 &&
        qpsPerConnection > std::numeric_limits<int>::max() / directionCount /
                max_num_channels) {
      throw std::overflow_error(
          "max_num_channels * direction_count * qpsPerConnection overflows int");
    }
    return max_num_channels * directionCount * qpsPerConnection;
  }

  int fixedChannelCompanionQpsPerPeerPerNic() const {
    if (max_num_channels < 0 || qpsPerConnection < 0) {
      throw std::invalid_argument(
          "max_num_channels and qpsPerConnection must be >= 0");
    }
    const int directionCount = fixedChannelDirectionCount();
    if (max_num_channels != 0 &&
        qpsPerConnection > std::numeric_limits<int>::max() / directionCount /
                max_num_channels) {
      throw std::overflow_error(
          "max_num_channels * direction_count * qpsPerConnection overflows int");
    }
    return max_num_channels * directionCount * qpsPerConnection;
  }

  int fixedChannelDirectionCount() const {
    return perChannelSize > 0 ? kIbDirections : 1;
  }

  // mlx5 Data-Direct: register MRs through the NIC's data-direct (BAR1) PCIe
  // path for ~2x NIC<->HBM RDMA-write BW on GB300 (NCCL's NCCL_IB_DATA_DIRECT).
  // The single shared comms::prims::DataDirectMode (see DataDirectMode.h) — the
  // same enum NIC discovery uses — so this field both selects the discovery
  // mode and gates the registration path. Disabled disables discovery's DD
  // probing too; Only/Both take effect only on a DD-capable NIC (a no-op
  // otherwise). The caller should tunnel NCCL_IB_DATA_DIRECT (0/1/2) into this
  // field.
  DataDirectMode enableDataDirect{DataDirectMode::Only};

  // PCIe Relaxed Ordering on eligible (bulk data) MRs so NIC<->HBM DMA TLPs
  // pipeline instead of strict-ordering to ~half rate (NCCL's
  // NCCL_IB_PCI_RELAXED_ORDERING). Only applied to MRs the caller marks
  // relaxed-ordering-eligible (data, not signal/counter). The caller should
  // tunnel NCCL_IB_PCI_RELAXED_ORDERING into this field.
  enum class PciRelaxedOrderingMode {
    Disabled, // strict ordering on every MR
    Enabled, // relaxed ordering on eligible MRs
    Auto, // relaxed ordering on eligible MRs when supported (NCCL default)
  };
  PciRelaxedOrderingMode enablePciRelaxedOrdering{PciRelaxedOrderingMode::Auto};

  // NIC data-placement ordering for IB QPs (mlx5 dp_ordering, a.k.a. DDP /
  // out-of-order data placement). The PCIe knob above reorders the NIC's writes
  // across the PCIe bus; this one decides whether the NIC may place inbound
  // fabric payload out of order, which is what lets adaptive routing spray a
  // QP's packets across paths without a reassembly stall.
  //
  // Auto is the default: take the strongest tier the NIC reports, walking
  // OooAll -> OooRw -> Ibta. See IbQpOrderingPolicy for why the ladder has
  // three rungs and what each one costs.
  IbQpOrderingPolicy qpOrderingPolicy{IbQpOrderingPolicy::Auto};

  // InfiniBand Verbs Timeout for QP ACK timeout (4.096us * 2^timeout). Valid
  // 1-31; 0 or >=32 is infinite. Default 20 (similar to NCCL_IB_TIMEOUT).
  uint8_t timeout{20};

  // InfiniBand retry count for QP transport errors (NCCL_IB_RETRY_CNT).
  uint8_t retryCount{7};

  // InfiniBand traffic class field (similar to NCCL_IB_TC).
  uint8_t trafficClass{224};

  // InfiniBand Service Level (similar to NCCL_IB_SL).
  uint8_t serviceLevel{0};

  // Minimum RNR NAK Timer field (ibv_qp_attr.min_rnr_timer); NCCL
  // IbvQpUtils=12.
  uint8_t minRnrTimer{12};

  // RNR retry count (ibv_qp_attr.rnr_retry); 7 means infinite.
  uint8_t rnrRetry{7};

  // Depth of the RDMA-Read/Atomic pipeline on one QP, applied to BOTH
  // directions: max_dest_rd_atomic (responder, QPC log_rra_max, written on
  // INIT->RTR) and max_rd_atomic (initiator, QPC log_sra_max, written on
  // RTR->RTS). Must be a power of two in [1, 128] because the NIC stores
  // log2 of it.
  //
  // 1 was the historical IBGDA behaviour and NOT a considered default: the DOCA
  // modify masks never carried MAX_QP_RD_ATOMIC / MAX_DEST_RD_ATOMIC, so both
  // QPC fields stayed at their zeroed value (log2(1) == 0), i.e. exactly one
  // outstanding read/atomic per QP per direction. Every prims signal is an
  // ATOMIC_FA, so a depth of 1 lets only one signal be in flight per QP.
  //
  // 16 matches what the rest of the world does for a GPU-initiated transport:
  // NVIDIA's own GDAKI defaults to the device maximum
  // (third-party/nccl/.../net_ib/gdaki/gin_host_gdaki.cc:391-394, which is 16
  // on ConnectX-8), and prims' AMD sibling hardcodes 15/16
  // (comms/prims/transport/amd/prims_amd_gda/PrimsAmdGdaHost.cc:1188). Note
  // NCCL's *CPU-driven* RC transport uses 1 (net_ib/connect.cc:408,465) -- the
  // split is deliberate, and prims is the GPU-initiated case.
  //
  // Raising it is safe: depth bounds how many read/atomic ops may be in flight,
  // not the order they take effect. Per mlx5dv_create_qp(3), on an
  // out-of-order-placement QP "RDMA Read and RDMA Atomic operations are
  // executed on the responder side in order ... after all previous messages are
  // done executing", so signals still land after their data and in sequence
  // among themselves.
  //
  // Measured benefit on the current channel design: none. See the diff's test
  // plan -- a fixed ~28 us per-chunk cost keeps the per-QP signal interval
  // above the RTT, so one credit is always sufficient. The value is aligned
  // with the vendor default rather than tuned.
  //
  // Clamped down to the NIC's max_qp_rd_atom / max_qp_init_rd_atom.
  uint8_t maxRdAtomic{16};

  // Deprecated compatibility setting. Per-peer state is always materialized
  // on demand; false no longer enables eager all-peer allocation.
  bool ibLazyConnect{true};
};

// Whether Data-Direct MR registration applies for a NIC: Data-Direct is
// requested via config (not Disabled) and the NIC is DD-capable.
// registerBuffer() selects the Data-Direct registration path exactly when this
// holds (and the mlx5dv symbol is available). Exposed as a free function so the
// config -> registration tunnel can be unit-tested without a NIC.
inline bool dataDirectActiveForNic(
    const MultipeerIbTransportConfig& config,
    bool nicIsDataDirect) {
  return config.enableDataDirect != DataDirectMode::Disabled && nicIsDataDirect;
}

// Whether PCIe Relaxed Ordering applies for a NIC: requested via config (not
// Disabled) and the NIC accepts the IBV_ACCESS_RELAXED_ORDERING access flag
// (probed during openNics). registerBuffer() sets the flag exactly when this
// holds, so on a NIC whose driver rejects it both Auto and Enabled fall back to
// strict ordering instead of failing registration. Free function so the
// config -> registration gating is unit-testable without a NIC.
inline bool relaxedOrderingActiveForNic(
    const MultipeerIbTransportConfig& config,
    bool nicRelaxedOrderingCapable) {
  return config.enablePciRelaxedOrdering !=
      MultipeerIbTransportConfig::PciRelaxedOrderingMode::Disabled &&
      nicRelaxedOrderingCapable;
}

// Exact caller-visible user-VA registration owned by its caller. A provider MR
// may cover its page-aligned DMA-BUF mapping, but only local lkeys and the
// requested ptr/size are exposed. The registration does not own the transport
// and must be passed to deregisterIbBufferRange() while the transport is alive.
class IbBufferRegistration {
 public:
  IbBufferRegistration() = default;
  ~IbBufferRegistration();
  IbBufferRegistration(const IbBufferRegistration&) = delete;
  IbBufferRegistration& operator=(const IbBufferRegistration&) = delete;
  IbBufferRegistration(IbBufferRegistration&& other) noexcept
      : localBuffer(std::exchange(other.localBuffer, IbgdaLocalBuffer{})),
        size(std::exchange(other.size, 0)),
        relaxedOrdering(std::exchange(other.relaxedOrdering, false)),
        mrs_(
            std::exchange(
                other.mrs_,
                std::array<ibverbx::ibv_mr*, kMaxNicsPerGpu>{})),
        numNics_(std::exchange(other.numNics_, 0)) {}
  // Assignment could silently discard MRs that still require an explicit
  // release.
  IbBufferRegistration& operator=(IbBufferRegistration&&) = delete;

  bool valid() const {
    return localBuffer.ptr != nullptr && size != 0 && numNics_ != 0;
  }

  IbgdaLocalBuffer localBuffer;
  std::size_t size{0};
  bool relaxedOrdering{false};

 private:
  friend class MultiPeerIbTransportBase;

  IbBufferRegistration(
      IbgdaLocalBuffer buffer,
      std::size_t registrationSize,
      bool registrationRelaxedOrdering,
      std::array<ibverbx::ibv_mr*, kMaxNicsPerGpu> mrs,
      int numNics)
      : localBuffer(buffer),
        size(registrationSize),
        relaxedOrdering(registrationRelaxedOrdering),
        mrs_(mrs),
        numNics_(numNics) {}

  void reset() {
    localBuffer = {};
    size = 0;
    relaxedOrdering = false;
    mrs_ = {};
    numNics_ = 0;
  }

  std::array<ibverbx::ibv_mr*, kMaxNicsPerGpu> mrs_{};
  int numNics_{0};
};

inline bool reliableDoorbellActiveForNic(
    const MultipeerIbTransportConfig& config,
    bool nicReliableDoorbellCapable) {
  if (config.enableReliableDoorbell.value_or(false) &&
      !nicReliableDoorbellCapable) {
    throw std::invalid_argument(
        "enableReliableDoorbell requires reliable-doorbell NIC support");
  }
  return config.enableReliableDoorbell.value_or(nicReliableDoorbellCapable);
}

inline bool reliableDoorbellNeedsCapabilityQuery(
    const MultipeerIbTransportConfig& config) {
  return config.enableReliableDoorbell.value_or(true);
}

inline bool collapsedCqActiveForTransport(
    const MultipeerIbTransportConfig& config,
    bool allNicsAcceptCollapsedCq) {
  if (config.enableCollapsedCq.value_or(false) && !allNicsAcceptCollapsedCq) {
    throw std::invalid_argument(
        "enableCollapsedCq requires collapsed-CQ support on every selected NIC");
  }
  return config.enableCollapsedCq.value_or(allNicsAcceptCollapsedCq);
}

inline bool collapsedCqNeedsCapabilityProbe(
    const MultipeerIbTransportConfig& config) {
  return config.enableCollapsedCq.value_or(true);
}

// Whether MultipeerIbTransportConfig::maxRdAtomic is programmable as-is.
// max_rd_atomic / max_dest_rd_atomic are stored by the NIC as log2, so only
// powers of two round-trip. DOCA's own setters only reject 0 (they call
// doca_internal_utils_next_power_of_two() where they meant
// doca_internal_utils_is_power_of_two()), so a non-power-of-two would silently
// be rounded DOWN by log2 -- e.g. 15 would become 8. Reject it here instead.
constexpr bool isIbMaxRdAtomicValid(unsigned value) {
  return value >= 1 && value <= 128 && (value & (value - 1)) == 0;
}

// Order in which connectPeers() walks a rank's pending peers.
// doMaterializePeer() is a rendezvous, so an edge only progresses while both
// ends are working on each other. Deadlock freedom needs a key that is
// symmetric (k(a,b) == k(b,a)) and injective in the peer for a fixed rank: the
// globally lowest-keyed pending edge then always has both ends selecting it.
// Both properties hold for XOR; the caller-side precondition is unchanged and
// still the one documented on MultiPeerTransport::materializePeers.
//
// The key also decides how much of the graph pairs up at once. Ordering by peer
// rank is equally deadlock-free but serializes a ring into nRanks - 1 rounds,
// because rank r takes r-1 before r+1 and so edge (r, r+1) cannot start until
// (r-1, r) has finished. XOR distance instead puts every (2i, 2i+1) edge at
// key 1 and every (2i+1, 2i+2) edge above it, collapsing a contiguous ring to
// two rounds when nRanks is even and three when it is odd. Rings the collective
// layer actually builds are strided (node * nvlSize + nvlRank), and a stride
// that is not a power of two costs a round or two more; the win is that the
// round count stays a small constant instead of scaling with nRanks. XOR is not
// optimal for every graph -- a few strided and irregular shapes need one to
// three rounds more than rank order -- but no topology in comms regresses more
// than that, against O(nRanks) rounds saved on every ring.
//
// Free function so the schedule is unit-testable without a NIC.
constexpr int peerMaterializationKey(int myRank, int peerRank) {
  return myRank ^ peerRank;
}

// Order a rank's pending peers into the sequence connectPeers() materializes
// them in. Free function so the schedule the transport actually runs is
// unit-testable without a NIC.
inline void sortPendingPeers(int myRank, std::vector<int>& peers) {
  std::sort(peers.begin(), peers.end(), [myRank](int lhs, int rhs) {
    return peerMaterializationKey(myRank, lhs) <
        peerMaterializationKey(myRank, rhs);
  });
}

/**
 * Transport connection information for RDMA QP setup.
 *
 * This struct is exchanged ONCE during the bootstrap phase to establish
 * RDMA connectivity between peers. Contains immutable connection parameters
 * that define how to reach a peer's QP.
 */
struct IbTransportExchInfo {
  // Queue Pair Number for RDMA connection
  uint32_t qpn{0};

  // Global Identifier for RoCE routing (16 bytes)
  uint8_t gid[16]{};

  // GID index used
  int gidIndex{0};

  // Local Identifier (for IB, not used in RoCE)
  uint16_t lid{0};

  // Port active MTU. Used to negotiate path MTU: min(local, remote).
  ibverbx::ibv_mtu mtu{ibverbx::IBV_MTU_4096};
};

/**
 * Maximum number of ranks supported for allGather-based exchange.
 * This limit exists because we use fixed-size arrays for QPN exchange.
 */
constexpr int kMaxRanksForAllGather = 128;

// Eager allGather QPN exchange uses a compact fixed-size wire format. Larger
// block-owned QP shapes must use lazy peer materialization.
constexpr int kMaxEagerExchangeQpsPerPeerPerNic = 128;

constexpr int kMaxIbGroups = 64;
constexpr int kMaxIbQpsPerBlockPerNic = 128;
constexpr int kMaxIbQpsPerPeerPerNic = kMaxIbGroups * kMaxIbQpsPerBlockPerNic;

/**
 * Transport exchange info for allGather-based exchange.
 *
 * Each rank contributes this structure containing per-NIC GID/LID and the
 * per-(target_rank, q) QPN this rank uses on that NIC.
 */
struct IbTransportExchInfoAll {
  // Per-NIC public info shared with peers (wire format). nicInfo[n] holds
  // this rank's NIC n's GID, LID, and the QPNs it uses to connect to each
  // (target_rank, q). Indices [numNics .. kMaxNicsPerGpu) are zero-init and
  // never read by peers (both ranks must agree on numNics — validated at
  // exchange time).
  struct NicWireInfo {
    uint8_t gid[16]{};
    uint16_t lid{0};
    // QPN this rank uses on this NIC to connect to (target_rank, q).
    // qpnForRank[myRank][*] is unused (set to 0).
    uint32_t qpnForRank[kMaxRanksForAllGather]
                       [kMaxEagerExchangeQpsPerPeerPerNic]{};
  };
  NicWireInfo nicInfo[kMaxNicsPerGpu]{};

  // Common (shared across NICs on this rank).
  int gidIndex{0};
  ibverbx::ibv_mtu mtu{ibverbx::IBV_MTU_4096};

  // Number of NICs (rails) used by this rank.
  // Must match across all ranks (validated at exchange time).
  int numNics{1};

  // Number of QPs per (peer, NIC) used by this rank.
  int numQpsPerPeerPerNic{1};

  // Block-owned QP shape.
  int maxGroups{64};
  int qpsPerBlockPerNic{1};
};

// Phases within the peer-pair-specific bootstrap tag computed by
// exchangeRawWithPeer().
constexpr int kIbPeerQpExchangeTag = 0;
constexpr int kIbPeerBufferExchangeTag = 1;

// Wire formats for bilateral peer materialization. Split into two phases: QP
// info first (to connect), then buffer info (acts as QP-ready barrier).
//
// WIRE FORMAT: PeerQpPayload is exchanged between peer ranks as raw bytes by
// exchangeWithPeer(), which sends and expects exactly sizeof(PeerQpPayload).
// Its layout is therefore an ABI both ends must agree on. Only APPEND new
// fields: inserting or reordering one shifts every field after it, so two
// ranks built from different revisions would decode each other's gids, qpns
// and MTU as garbage instead of failing on the size.
struct PeerQpPayload {
  struct NicQpInfo {
    uint8_t gid[16]{};
    uint16_t lid{0};
    uint32_t qpns[kMaxIbQpsPerPeerPerNic]{};
  };
  NicQpInfo nicInfo[kMaxNicsPerGpu]{};
  int gidIndex{0};
  int mtu{0};
  int numNics{0};
  int numQpsPerPeerPerNic{0};
  int maxGroups{0};
  int qpsPerBlockPerNic{0};
  // dp_ordering mode, exchanged so a mismatch is a loud connect-time error
  // rather than a silent one-sided reassembly change. NVIDIA's GDAKI refuses to
  // connect on an ordering_semantic mismatch for the same reason.
  // static_cast<int>(IbQpOrderingSemantic); 0 == Ibta == opted out.
  int qpOrderingSemantic{0};
  // Responder/initiator RDMA-Read/Atomic depth (config maxRdAtomic after the
  // NIC-capability clamp). Exchanged for the same reason: the two ends must
  // agree or one side's log_rra_max will not cover the other's log_sra_max.
  // Defaults to the same 1 the transport resolves when nobody raises the depth.
  int maxRdAtomic{1};
};

struct PeerBufferPayload {
  IbgdaBufferExchInfo recvStaging;
  IbgdaBufferExchInfo srSignal;
  IbgdaBufferExchInfo slotSignal;
  IbgdaBufferExchInfo slotDiscard;
};

// Which memory a NIC completion counter lives in. Shared by the slot counter
// (#16) and the send/recv NIC_DONE counter:
//   Device     - GPU device memory, allocated and registered by the transport.
//                The NIC bumps it via a loopback RDMA atomic (IBGDA).
//   HostPinned - host-mapped (cudaHostAllocMapped) memory, allocated by the
//                transport; the CPU progress thread writes it and the device
//                reads via the mapped pointer (IBRC). Never MR-registered.
enum class IbCounterStorage {
  Device,
  HostPinned,
};

// Per-peer send/recv staging-ring views. Eager mode owns the bulk allocations
// and slices these; the device side reads them via channelLayoutForPeer().
struct IbSendRecvPeerBuffers {
  IbgdaLocalBuffer sendStaging;
  IbgdaLocalBuffer recvStaging;
  IbgdaLocalBuffer signal;
  IbgdaLocalBuffer counter;
  IbgdaLocalBuffer counterCompletion;
  IbgdaRemoteBuffer remoteRecvStaging;
  IbgdaRemoteBuffer remoteSignal;
};

/**
 * MultiPeerIbTransportBase - backend-agnostic host control plane shared by the
 * multi-peer IB transports (IBGDA today, IBRC next).
 *
 * This is a NON-template base so its (heavy) method bodies live in
 * MultiPeerIbTransport.cc and are compiled exactly once, reused by every
 * backend with no per-backend wiring. It owns rank state + rank<->peerIndex
 * mapping, the full refcounted MR registry
 * (registerBuffer/deregisterBuffer/exchangeBuffer), the generic per-NIC IB
 * resources (NicResources), the bilateral bootstrap exchange, and the lazy
 * materialization queue/state. It NEVER calls into a backend; the small piece
 * of control flow that does (the connect loop) lives in the CRTP layer below.
 *
 * MR registration is generic — it resolves the allocation, exports a DMA-BUF
 * via the platform helper, and registers one MR per NIC on the base-owned PDs
 * (nics_[n].ibvPd), which the backend fills during NIC bring-up. No backend
 * hook is involved.
 */
class MultiPeerIbTransportBase {
 public:
  /** @return Number of peers (nRanks - 1). */
  int numPeers() const {
    return nRanks_ - 1;
  }

  /** @return This rank's id. */
  int myRank() const {
    return myRank_;
  }

  /** @return Total number of ranks. */
  int nRanks() const {
    return nRanks_;
  }

  /**
   * @return Number of NICs (rails) in use. Resolved by the shared base during
   * construction.
   */
  int numNics() const {
    return numNics_;
  }

  /** @return Configured send/recv staging pipeline depth. */
  int pipelineDepth() const {
    return config_.pipelineDepth;
  }

  /**
   * @return Logical IB channels per peer; device code requires
   * group_id < this. Cross-validated across ranks at materialization.
   */
  int maxNumChannels() const {
    return config_.max_num_channels;
  }

  /**
   * registerBuffer - Register a user GPU buffer for RDMA, refcounted per
   * allocation. Containment fast-path returns cached per-NIC lkeys without any
   * driver call; on a miss it resolves the allocation, exports a DMA-BUF, and
   * registers one MR per NIC on the base-owned PDs.
   *
   * @return IbgdaLocalBuffer carrying one lkey per NIC.
   */
  // @param relaxedOrdering eligible for PCIe Relaxed Ordering (gated by
  //   config.enablePciRelaxedOrdering). Only bulk data (staging) MRs pass true;
  //   signal/counter MRs stay strict. Data-Direct is applied automatically on
  //   DD-capable NICs regardless of this flag.
  IbgdaLocalBuffer
  registerBuffer(void* ptr, std::size_t size, bool relaxedOrdering = false);

  /** deregisterBuffer - Decrement refcount; deregister all per-NIC MRs at 0. */
  void deregisterBuffer(void* ptr);

  /**
   * Register a local send source and expose exactly [ptr, ptr + size) without
   * allocation discovery or caching. A provider MR may be page-aligned. The
   * result exposes local keys only and is invisible to exchangeBuffer() and
   * registeredSlotMemoryExchInfo(); use registerBuffer() for memory that peers
   * write into.
   */
  IbBufferRegistration registerIbBufferRange(void* ptr, std::size_t size);

  /** Release an exact-range registration and invalidate it. */
  void deregisterIbBufferRange(IbBufferRegistration& registration);

  /**
   * exchangeBuffer - COLLECTIVE. allGather a registered buffer's addr + per-NIC
   * rkeys; return one IbgdaRemoteBuffer per peer (indexed by peerIndexToRank).
   */
  std::vector<IbgdaRemoteBuffer> exchangeBuffer(
      const IbgdaLocalBuffer& localBuf);

  /** Queue a peer for lazy materialization (no network I/O). */
  void queuePeerForMaterialization(int peerRank);

  /**
   * Report the outcome of a connectPeers() round. Defined out-of-line so this
   * header, which every IB backend includes, does not pull in glog.
   */
  void logPeersMaterialized(
      std::size_t peerCount,
      std::int64_t elapsedMs,
      bool failed) const;

  static std::int64_t elapsedMsSince(
      std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - start)
        .count();
  }

  /** @return true if the peer is materialized and ready for kernel use. */
  bool isPeerMaterialized(int peerRank) const;

 protected:
  MultiPeerIbTransportBase(
      int myRank,
      int nRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      MultipeerIbTransportConfig config);

  // Non-virtual protected dtor: the base is never owned/deleted polymorphically
  // (the dispatcher holds the concrete backend type). Defined out-of-line in
  // the .cc so the unique_ptr<DeviceBuffer> members destruct against a complete
  // type.
  ~MultiPeerIbTransportBase();

  MultiPeerIbTransportBase(const MultiPeerIbTransportBase&) = delete;
  MultiPeerIbTransportBase& operator=(const MultiPeerIbTransportBase&) = delete;

  int rankToPeerIndex(int rank) const {
    return (rank < myRank_) ? rank : (rank - 1);
  }
  int peerIndexToRank(int peerIndex) const {
    return (peerIndex < myRank_) ? peerIndex : (peerIndex + 1);
  }

  // Generic NIC bring-up: resolve NIC names (config.gpuNicMap / topology
  // auto-discovery), open device + PD, and query GID + port (active MTU, link
  // layer, port state). Fills nics_
  // (deviceName/ibvCtx/ibvPd/localGid/linkLayer) and localMtu_ (from NIC 0),
  // using gidIndex_. No backend hook — each backend builds its address handles
  // afterwards from nics_[n].linkLayer + config_.addressFamily.
  void openNics();

  // ---- shared eager-exchange scaffolding ----
  // Collective allGather of this rank's exchange info. The backend fills the
  // backend-specific QPN/GID/LID fields of localInfo; the base guards the rank
  // count, places localInfo at myRank_, allGathers, and returns all ranks'
  // info (indexed by global rank).
  std::vector<IbTransportExchInfoAll> allGatherExchInfo(
      const IbTransportExchInfoAll& localInfo);

  // Validate every peer agrees on numNics (same-rail pairing precondition) and
  // numQpsPerPeerPerNic. Throws std::runtime_error on mismatch.
  void validatePeerTopology(
      const std::vector<IbTransportExchInfoAll>& allInfo) const;

  // Bilateral bootstrap exchange of a fixed-size payload with one peer. The
  // typed wrapper is header-only (so it can instantiate with backend-private
  // payload types); the heavy logic (lower-rank-recvs-first to avoid deadlock)
  // lives in exchangeRawWithPeer in the .cc. The bootstrap implementation owns
  // timeout and cancellation so caller-owned payloads remain live until the
  // exchange completes.
  template <typename T>
  T exchangeWithPeer(int peerRank, const T& localPayload, int tag) {
    T remotePayload{};
    exchangeRawWithPeer(
        peerRank, &localPayload, &remotePayload, sizeof(T), tag);
    return remotePayload;
  }
  void exchangeRawWithPeer(
      int peerRank,
      const void* localPayload,
      void* remotePayload,
      std::size_t bytes,
      int tag);

  // ---- shared send/recv staging-ring lifecycle (eager mode) ----
  // Backend-agnostic host send/recv buffer management, shared by IBGDA (Device
  // counter, NIC loopback atomic) and IBRC (Host counter, CPU proxy). Staging
  // = max_num_channels * perChannelSize per direction; signal is sized
  // off max_num_channels. Per-peer staging + signal are device-registered;
  // recvStaging + signal are collectively exchanged so peers can RDMA into our
  // ring.
  bool sendRecvBuffersEnabled() const {
    return config_.perChannelSize > 0;
  }
  IbChannelLayout channelLayoutForPeer(int peerIndex) const;
  // Allocate + register the per-peer staging/signal bulks and slice them.
  // counterStorage selects the NIC_DONE counter: Device (transport-allocated,
  // registered) or HostPinned (transport-allocated host-mapped, never
  // registered).
  void allocateSendRecvBuffersEager(IbCounterStorage counterStorage);
  // COLLECTIVE. allGather recvStaging + signal so each peer holds our remote
  // views. Must be called after allocateSendRecvBuffersEager().
  void exchangeSendRecvBuffersEager();
  void cleanupSendRecvBuffers() noexcept;

  // ---- per-peer (lazy) send/recv: shared by IBGDA + IBRC ----
  // Allocate + register ONE peer's send/recv rings on demand (lazy connect) and
  // fill the outbound payload's recvStaging/srSignal exch info. counterStorage
  // selects the NIC_DONE counter: Device (a registered slice of the contiguous
  // per-peer buffer; NIC loopback atomic — IBGDA) or HostPinned (a separate
  // host-mapped allocation written by the CPU proxy — IBRC). The per-peer
  // buffer is dedicated to this peer pair, so no numPeers slicing is needed.
  void allocateSendRecvBufferForPeer(
      int peerIndex,
      PeerBufferPayload& payload,
      IbCounterStorage counterStorage);
  // Apply a peer's payload: remote recvStaging/signal views are used whole.
  void applyRemoteSendRecvBuffer(
      int peerIndex,
      const PeerBufferPayload& remotePayload);
  // Per-peer teardown: deregister + free this peer's lazy allocation and reset
  // its views. Safe on an unmaterialized peer.
  void cleanupSendRecvBufferForPeer(int peerIndex) noexcept;

  void validateSendRecvConfig() const;
  std::size_t sendRecvStagingBytesPerPeer() const;
  std::size_t sendRecvSignalBytesPerPeer() const;
  std::size_t sendRecvCounterBytesPerPeer() const;

  void allocateSignalCounterResources(
      IbCounterStorage counterStorage,
      bool allocateDiscardSignal);
  void cleanupSignalCounterResources() noexcept;
  void cleanupPeerSignalCounterResources(int peerIndex) noexcept;
  void allocatePeerSignalCounterResources(
      int peerIndex,
      PeerBufferPayload& payload,
      IbCounterStorage counterStorage,
      bool allocateDiscardSignal);
  void applyRemoteSignalCounterResources(
      int peerIndex,
      const PeerBufferPayload& remotePayload,
      bool hasDiscardSignal);

  IbgdaRemoteBuffer slotRemoteSignalView(int peerIndex) const;
  IbgdaLocalBuffer slotLocalSignalView(int peerIndex) const;
  IbgdaLocalBuffer slotCounterDeviceView(int peerIndex) const;
  IbgdaLocalBuffer slotCounterHostView(int peerIndex) const;
  IbgdaRemoteBuffer slotDiscardSignalRemoteView(int peerIndex) const;

  // Cached MR entry: one MR per (CUDA allocation, NIC), refcounted. Multiple
  // user buffers within the same allocation share one MR set.
  struct CachedMr {
    std::array<ibverbx::ibv_mr*, kMaxNicsPerGpu> mrs{};
    std::size_t allocSize{0};
    int refs{0};
    // Effective PCIe Relaxed Ordering the MRs were registered with (the
    // caller's request resolved against config). Part of the cache key: a
    // containment hit must resolve to the same value, else the access-flag
    // (ordering) semantics would silently differ from what the caller asked
    // for.
    bool relaxedOrdering{false};
  };

  const int myRank_{-1};
  const int nRanks_{0};
  std::shared_ptr<meta::comms::IBootstrap> bootstrap_;
  MultipeerIbTransportConfig config_;

  // Number of NICs (rails) in use; resolved by the base constructor.
  int numNics_{1};

  // Generic IB facts captured during NIC bring-up (openNics()): the RoCE GID
  // index (resolved from config in the ctor) and the negotiated active MTU
  // (NIC 0's). Read by backends when building address handles / connecting QPs.
  int gidIndex_{3};
  ibverbx::ibv_mtu localMtu_{ibverbx::IBV_MTU_4096};

  // Per-NIC generic IB resources (device name, context, PD, GID, link layer).
  // openNics() fills these; the base registers MRs on the PDs. The backend
  // keeps only its backend-specific per-NIC state (e.g. DOCA AH attrs and QP
  // groups), index-aligned with this vector.
  struct NicResources {
    std::string deviceName;
    ibverbx::ibv_context* ibvCtx{nullptr};
    ibverbx::ibv_pd* ibvPd{nullptr};
    ibverbx::ibv_gid localGid{};
    int linkLayer{0}; // ibverbx::IBV_LINK_LAYER_* (IB vs Ethernet/RoCE)
    // This NIC exposes a Data-Direct (`_dma`) variant, so data MRs can register
    // through the PCIe (BAR1) path. Copied from the discovery NicCandidate.
    bool isDataDirect{false};
    // This NIC's driver accepts IBV_ACCESS_RELAXED_ORDERING (probed once during
    // openNics). registerBuffer() applies Relaxed Ordering only when every NIC
    // is capable, so an unsupporting NIC falls back to strict ordering instead
    // of failing every data-MR registration.
    bool relaxedOrderingCapable{false};
  };
  std::vector<NicResources> nics_;

  // True iff every opened NIC accepts IBV_ACCESS_RELAXED_ORDERING (AND of
  // nics_[n].relaxedOrderingCapable, computed once in openNics). The MR cache
  // keys on a single effective-ordering bool per allocation, so Relaxed
  // Ordering must be uniform across NICs; gating on this aggregate keeps it so.
  bool relaxedOrderingCapable_{false};

  struct RegistrationState {
    // Maps allocation base address -> cached MR covering the full allocation.
    // Ordered map enables O(log n) containment lookup via upper_bound.
    std::map<uintptr_t, CachedMr> registeredBuffers;
  };
  folly::Synchronized<RegistrationState> registrationState_;

  IbgdaLocalBuffer registerBufferLocked(
      void* ptr,
      std::size_t size,
      bool relaxedOrdering,
      RegistrationState& registrations);
  void deregisterBufferLocked(void* ptr, RegistrationState& registrations);

  // Shared send/recv staging-ring state (eager mode). Owns the bulk
  // allocations; sendRecvPeerBuffers_ slices them per peer.
  std::vector<IbSendRecvPeerBuffers> sendRecvPeerBuffers_;
  std::unique_ptr<meta::comms::DeviceBuffer> sendRecvSendStagingBulk_;
  std::unique_ptr<meta::comms::DeviceBuffer> sendRecvRecvStagingBulk_;
  // Signal + device-counter control regions packed into one granularity-aligned
  // allocation (both Data-Direct-registered; share one aligned MR).
  // Host-counter configs put only the signal region here. See
  // allocateSendRecvBuffersEager.
  std::unique_ptr<meta::comms::DeviceBuffer> sendRecvControlBulk_;
  IbgdaLocalBuffer sendRecvRecvStagingBulkReg_;
  IbgdaLocalBuffer sendRecvSignalBulkReg_;
  IbgdaLocalBuffer sendRecvCounterBulkReg_;
  IbCounterStorage sendRecvCounterStorage_{IbCounterStorage::Device};

  // Lazy materialization state machine.
  // connectPeers() holds this lock through the backend/bootstrap exchange so
  // fixed bootstrap tags cannot be reused concurrently on one communicator.
  mutable std::mutex materializationMutex_;
  std::vector<int> pendingPeers_;
  std::vector<bool> peerMaterialized_;
  bool materializationFailed_{false};

 private:
  struct DeviceSlotAllocation {
    void* ptr{nullptr};
    std::size_t bytes{0};
    bool registered{false};
    // On AMD the signal-inbox/discard buffers are host-pinned (device-memory
    // MR registration via peer-mem is unreliable); free accordingly.
    bool isHostPinned{false};

    DeviceSlotAllocation() = default;
    DeviceSlotAllocation(const DeviceSlotAllocation&) = delete;
    DeviceSlotAllocation& operator=(const DeviceSlotAllocation&) = delete;
    DeviceSlotAllocation(DeviceSlotAllocation&& other) noexcept
        : ptr(std::exchange(other.ptr, nullptr)),
          bytes(std::exchange(other.bytes, 0)),
          registered(std::exchange(other.registered, false)),
          isHostPinned(std::exchange(other.isHostPinned, false)) {}
    DeviceSlotAllocation& operator=(DeviceSlotAllocation&& other) noexcept {
      ptr = std::exchange(other.ptr, nullptr);
      bytes = std::exchange(other.bytes, 0);
      registered = std::exchange(other.registered, false);
      isHostPinned = std::exchange(other.isHostPinned, false);
      return *this;
    }
  };

  struct CounterSlotAllocation {
    void* hostPtr{nullptr};
    void* devicePtr{nullptr};
    std::size_t bytes{0};
    bool registered{false};

    CounterSlotAllocation() = default;
    CounterSlotAllocation(const CounterSlotAllocation&) = delete;
    CounterSlotAllocation& operator=(const CounterSlotAllocation&) = delete;
    CounterSlotAllocation(CounterSlotAllocation&& other) noexcept
        : hostPtr(std::exchange(other.hostPtr, nullptr)),
          devicePtr(std::exchange(other.devicePtr, nullptr)),
          bytes(std::exchange(other.bytes, 0)),
          registered(std::exchange(other.registered, false)) {}
    CounterSlotAllocation& operator=(CounterSlotAllocation&& other) noexcept {
      hostPtr = std::exchange(other.hostPtr, nullptr);
      devicePtr = std::exchange(other.devicePtr, nullptr);
      bytes = std::exchange(other.bytes, 0);
      registered = std::exchange(other.registered, false);
      return *this;
    }
  };

  void freeDeviceSlotAllocation(DeviceSlotAllocation& allocation) noexcept;
  DeviceSlotAllocation allocateDeviceSlotAllocation(
      std::size_t bytes,
      const char* label,
      meta::comms::memtrace::GpuMemoryResourceType resource);
  void freeCounterSlotAllocation(CounterSlotAllocation& allocation) noexcept;
  CounterSlotAllocation allocateCounterSlotAllocation(
      IbCounterStorage storage,
      std::size_t bytes,
      const char* label);
  IbgdaLocalBuffer registerSlotMemory(
      void* registrationPtr,
      void* devicePtr,
      std::size_t bytes,
      bool& registered);
  IbgdaBufferExchInfo registeredSlotMemoryExchInfo(void* registrationPtr) const;

  std::vector<IbgdaRemoteBuffer> slotRemoteSignalViews_;
  std::vector<IbgdaLocalBuffer> slotLocalSignalViews_;
  std::vector<IbgdaLocalBuffer> slotCounterDeviceViews_;
  std::vector<IbgdaLocalBuffer> slotCounterHostViews_;
  std::vector<IbgdaRemoteBuffer> slotDiscardSignalRemoteViews_;

  DeviceSlotAllocation slotSignalAllocation_;
  CounterSlotAllocation slotCounterAllocation_;
  DeviceSlotAllocation slotDiscardSignalAllocation_;
  // Host-mapped send/recv NIC_DONE counter (counterStorage == Host). Owns the
  // host-pinned allocation; sliced per peer into IbSendRecvPeerBuffers.counter.
  CounterSlotAllocation sendRecvHostCounterAllocation_;
  std::vector<DeviceSlotAllocation> lazySlotSignalAllocations_;
  std::vector<CounterSlotAllocation> lazySlotCounterAllocations_;
  std::vector<DeviceSlotAllocation> lazySlotDiscardSignalAllocations_;
  // Lazy per-peer send/recv allocations: one contiguous device buffer per
  // materialized peer (sendStaging|recvStaging|signal|state, plus the counter
  // when device-resident). Empty in eager mode. Shared by IBGDA (Device
  // counter) and IBRC, which additionally allocates a per-peer host-mapped
  // NIC_DONE counter below.
  std::vector<std::unique_ptr<meta::comms::DeviceBuffer>> lazyPeerBufs_;
  std::vector<CounterSlotAllocation> lazySendRecvHostCounters_;
};

/**
 * MultiPeerIbTransport<Backend> - CRTP layer over MultiPeerIbTransportBase.
 *
 * Holds ONLY the small piece of control plane that must call into the concrete
 * backend: the lazy connect loop (connectPeers) drives the backend's per-peer
 * doMaterializePeer()/cleanupPeerOnFailure() hooks via a static `backend()`
 * downcast (no vtable). Each backend derives as
 *   `class MultipeerIbgdaTransport : public
 * MultiPeerIbTransport<MultipeerIbgdaTransport>`.
 * All backend-agnostic state and methods are inherited from the non-template
 * base, so they are compiled once (in MultiPeerIbTransport.cc) and shared.
 */
template <typename Backend>
class MultiPeerIbTransport : public MultiPeerIbTransportBase {
 public:
  /** Materialize one peer (queue + connect). */
  void materializePeer(int peerRank) {
    queuePeerForMaterialization(peerRank);
    connectPeers();
  }

  /**
   * Connect all queued peers in peerMaterializationKey order (deadlock-safe for
   * >2 ranks, given the symmetric request graph materializePeers requires).
   */
  void connectPeers();

 protected:
  MultiPeerIbTransport(
      int myRank,
      int nRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      MultipeerIbTransportConfig config)
      : MultiPeerIbTransportBase(
            myRank,
            nRanks,
            std::move(bootstrap),
            std::move(config)) {}

  ~MultiPeerIbTransport() = default;

  // CRTP downcast for static dispatch into backend hooks.
  Backend& backend() {
    return static_cast<Backend&>(*this);
  }
  const Backend& backend() const {
    return static_cast<const Backend&>(*this);
  }
};

template <typename Backend>
void MultiPeerIbTransport<Backend>::connectPeers() {
  // queuePeerForMaterialization() releases this mutex before entering here;
  // backend hooks must not recursively call materializePeer()/connectPeers().
  const std::lock_guard<std::mutex> lock(materializationMutex_);
  if (materializationFailed_) {
    pendingPeers_.clear();
    throw std::runtime_error(
        "MultiPeerIbTransport: lazy peer materialization previously failed; "
        "retry is not supported");
  }
  if (pendingPeers_.empty()) {
    return;
  }
  // Deadlock-free on a symmetric request graph; see peerMaterializationKey.
  sortPendingPeers(myRank_, pendingPeers_);

  std::vector<int> peers;
  peers.swap(pendingPeers_);
  std::vector<int> touchedPeerIndexes;
  touchedPeerIndexes.reserve(peers.size());

  const auto startTime = std::chrono::steady_clock::now();
  try {
    for (int peerRank : peers) {
      if (peerMaterialized_[rankToPeerIndex(peerRank)]) {
        continue;
      }
      touchedPeerIndexes.push_back(rankToPeerIndex(peerRank));
      backend().doMaterializePeer(peerRank);
    }
  } catch (...) {
    materializationFailed_ = true;
    // Report elapsed on the way out too: a rendezvous that stalls and then
    // errors is the case where the timing matters most.
    logPeersMaterialized(
        touchedPeerIndexes.size(), elapsedMsSince(startTime), /*failed=*/true);
    for (int peerIndex : touchedPeerIndexes) {
      backend().cleanupPeerOnFailure(peerIndex);
    }
    throw;
  }
  // A rank blocks here until each peer reaches the matching rendezvous, so this
  // reports queue wait as well as local work. Elapsed far above the per-peer
  // cost means peers are arriving late, not that materialization is slow.
  logPeersMaterialized(
      touchedPeerIndexes.size(), elapsedMsSince(startTime), /*failed=*/false);
}

} // namespace comms::prims
