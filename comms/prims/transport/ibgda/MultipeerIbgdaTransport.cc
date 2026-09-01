// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"

#ifdef __HIP_PLATFORM_AMD__
// On AMD: use the HIP runtime for the cuda* API calls below (HIPify
// renames cuda* -> hip* in source before compilation), and bring in
// `meta::comms::DeviceBuffer` from the pipes-local HIP shim.
#include <hip/hip_runtime.h>

#include "comms/prims/transport/amd/HipHostCompat.h"
#else
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"
// NVIDIA-only host-side helpers. On AMD their functionality is provided
// by `comms/prims/transport/amd/DocaCompat.h` (already included via
// `MultipeerIbgdaTransport.h`) which translates `doca_*` to the
// `prims_amd_gda_*` host APIs in `amd/prims_amd_gda/PrimsAmdGdaHost.{h,cc}`.
#ifndef __HIP_PLATFORM_AMD__
#include "comms/prims/platform/CudaDriverLazy.h"
#include "comms/prims/platform/DocaHostUtils.h"
// MCCL_IBGDA_MAX_RD_ATOMIC / MCCL_IBGDA_QP_ORDERING_SEMANTIC. Generated header.
#include "comms/utils/cvars/nccl_cvars.h" // @manual
#endif
#include "comms/prims/transport/ibgda/MultipeerIbgdaDeviceTransport.cuh"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransportCuda.cuh"
#include "comms/prims/transport/rdma/NicDiscovery.h"

namespace comms::prims {

namespace {

constexpr int kHopLimit = 255;

// Host loopback QPs are only used to bring each exported companion QP to RTS.
// The device-visible companion QP is created by create_qp_group_hl() with
// mainAttr and therefore uses config_.qpDepth.
constexpr uint32_t kLoopbackCompanionQpDepth = 32;
} // namespace

namespace {

// Convert ibverbx::ibv_mtu enum to doca_verbs_mtu_size enum.
doca_verbs_mtu_size ibv_mtu_to_doca_mtu(ibverbx::ibv_mtu ibvMtu) {
  switch (ibvMtu) {
    case ibverbx::IBV_MTU_256:
      return DOCA_VERBS_MTU_SIZE_256_BYTES;
    case ibverbx::IBV_MTU_512:
      return DOCA_VERBS_MTU_SIZE_512_BYTES;
    case ibverbx::IBV_MTU_1024:
      return DOCA_VERBS_MTU_SIZE_1K_BYTES;
    case ibverbx::IBV_MTU_2048:
      return DOCA_VERBS_MTU_SIZE_2K_BYTES;
    case ibverbx::IBV_MTU_4096:
      return DOCA_VERBS_MTU_SIZE_4K_BYTES;
    default:
      throw std::runtime_error(
          "Invalid ibv_mtu value: " + std::to_string(ibvMtu));
  }
}

// Convert DOCA error to string using lookup table
// Values match the doca_error_t enum (0 = DOCA_SUCCESS through 31)
const char* docaErrorToString(doca_error_t err) {
  static constexpr const char* kDocaErrorNames[] = {
      "DOCA_SUCCESS",
      "DOCA_ERROR_UNKNOWN",
      "DOCA_ERROR_NOT_PERMITTED",
      "DOCA_ERROR_IN_USE",
      "DOCA_ERROR_NOT_SUPPORTED",
      "DOCA_ERROR_AGAIN",
      "DOCA_ERROR_INVALID_VALUE",
      "DOCA_ERROR_NO_MEMORY",
      "DOCA_ERROR_INITIALIZATION",
      "DOCA_ERROR_TIME_OUT",
      "DOCA_ERROR_SHUTDOWN",
      "DOCA_ERROR_CONNECTION_RESET",
      "DOCA_ERROR_CONNECTION_ABORTED",
      "DOCA_ERROR_CONNECTION_INPROGRESS",
      "DOCA_ERROR_NOT_CONNECTED",
      "DOCA_ERROR_NO_LOCK",
      "DOCA_ERROR_NOT_FOUND",
      "DOCA_ERROR_IO_FAILED",
      "DOCA_ERROR_BAD_STATE",
      "DOCA_ERROR_UNSUPPORTED_VERSION",
      "DOCA_ERROR_OPERATING_SYSTEM",
      "DOCA_ERROR_DRIVER",
      "DOCA_ERROR_UNEXPECTED",
      "DOCA_ERROR_ALREADY_EXIST",
      "DOCA_ERROR_FULL",
      "DOCA_ERROR_EMPTY",
      "DOCA_ERROR_IN_PROGRESS",
      "DOCA_ERROR_TOO_BIG",
      "DOCA_ERROR_AUTHENTICATION",
      "DOCA_ERROR_BAD_CONFIG",
      "DOCA_ERROR_SKIPPED",
      "DOCA_ERROR_DEVICE_FATAL_ERROR",
  };
  auto idx = static_cast<int>(err);
  if (idx >= 0 && idx < static_cast<int>(std::size(kDocaErrorNames))) {
    return kDocaErrorNames[idx];
  }
  return "DOCA_ERROR_UNKNOWN_CODE";
}

#ifndef __HIP_PLATFORM_AMD__
const char* reliableDoorbellModeName(const std::optional<bool>& enabled) {
  if (!enabled.has_value()) {
    return "auto";
  }
  return *enabled ? "enable" : "disable";
}

// ---------------------------------------------------------------------------
// RDMA-Read/Atomic depth (QPC log_sra_max / log_rra_max)
// ---------------------------------------------------------------------------

// MCCL_IBGDA_MAX_RD_ATOMIC wins over MultipeerIbTransportConfig::maxRdAtomic
// only when it holds something other than its registered default. Comparing
// against the generated _DEFAULTCVARVALUE rather than a literal also covers the
// case where ncclCvarInit() was never called: prims is driven from benchmarks
// and unit tests that do not run it, and both globals are then still
// zero-initialized and therefore equal.
uint8_t resolveMaxRdAtomic(const MultipeerIbgdaTransportConfig& config) {
  unsigned value = config.maxRdAtomic;
  const int cvarValue = MCCL_IBGDA_MAX_RD_ATOMIC;
  if (cvarValue != MCCL_IBGDA_MAX_RD_ATOMIC_DEFAULTCVARVALUE) {
    if (cvarValue < 0 ||
        !isIbMaxRdAtomicValid(static_cast<unsigned>(cvarValue))) {
      throw std::invalid_argument(
          fmt::format(
              "MCCL_IBGDA_MAX_RD_ATOMIC={} is not a power of two in [1, 128]",
              cvarValue));
    }
    value = static_cast<unsigned>(cvarValue);
  }
  if (!isIbMaxRdAtomicValid(value)) {
    throw std::invalid_argument(
        fmt::format("maxRdAtomic={} is not a power of two in [1, 128]", value));
  }
  return static_cast<uint8_t>(value);
}

// Largest power of two <= limit, or 0 when limit is 0.
uint8_t floorPowerOfTwo(uint8_t limit) {
  uint8_t value = 0;
  for (unsigned candidate = 1; candidate <= limit && candidate <= 128;
       candidate *= 2) {
    value = static_cast<uint8_t>(candidate);
  }
  return value;
}

// Clamp the requested RDMA-Read/Atomic depth to what the NIC advertises.
// max_qp_rd_atom and max_qp_init_rd_atom bound the two directions, and sources
// disagree on which bounds which (DOCA's own getter docs are the reverse of the
// usual ibv_device_attr reading), so take the min of both: the single value
// here feeds both max_rd_atomic and max_dest_rd_atomic, and the min is correct
// under either mapping. Asking for more than the NIC supports would make the
// INIT2RTR/RTR2RTS DEVX command fail with an opaque syndrome.
//
// The capability query is a DEVX QUERY_HCA_CAP round trip, so it is only issued
// for a non-default depth; at the default of 1 this returns without touching
// the NIC.
uint8_t clampMaxRdAtomicToNic(
    uint8_t requested,
    ::ibv_context* ibvContext,
    const std::string& deviceName) {
  if (requested <= 1) {
    return requested;
  }

  doca_verbs_device_attr* deviceAttr = nullptr;
  const doca_error_t queryErr =
      doca_verbs_query_device(ibvContext, &deviceAttr);
  if (queryErr != DOCA_SUCCESS) {
    LOG(WARNING) << "MultipeerIbgdaTransport: read/atomic depth capability "
                    "query failed for NIC "
                 << deviceName << ": " << docaErrorToString(queryErr)
                 << "; leaving max_rd_atomic=" << (unsigned)requested
                 << " unclamped";
    return requested;
  }
  const uint8_t maxQpRdAtom =
      doca_verbs_device_attr_get_max_qp_rd_atom(deviceAttr);
  const uint8_t maxQpInitRdAtom =
      doca_verbs_device_attr_get_max_qp_init_rd_atom(deviceAttr);
  const doca_error_t freeErr = doca_verbs_device_attr_free(deviceAttr);
  if (freeErr != DOCA_SUCCESS) {
    LOG(WARNING) << "MultipeerIbgdaTransport: failed to free DOCA device "
                    "attributes for NIC "
                 << deviceName << ": " << docaErrorToString(freeErr);
  }

  const uint8_t allowed =
      floorPowerOfTwo(std::min(maxQpRdAtom, maxQpInitRdAtom));
  if (allowed == 0) {
    LOG(WARNING) << "MultipeerIbgdaTransport: NIC " << deviceName
                 << " reports max_qp_rd_atom=" << (unsigned)maxQpRdAtom
                 << " max_qp_init_rd_atom=" << (unsigned)maxQpInitRdAtom
                 << "; keeping max_rd_atomic=1";
    return 1;
  }
  if (allowed < requested) {
    LOG(WARNING) << "MultipeerIbgdaTransport: NIC " << deviceName
                 << " caps max_rd_atomic at " << (unsigned)allowed
                 << " (requested " << (unsigned)requested << ")";
    return allowed;
  }
  return requested;
}

bool nicSupportsReliableDoorbell(
    ::ibv_context* ibvContext,
    const std::string& deviceName) {
  doca_verbs_device_attr* deviceAttr = nullptr;
  const doca_error_t queryErr =
      doca_verbs_query_device(ibvContext, &deviceAttr);
  if (queryErr != DOCA_SUCCESS) {
    LOG(WARNING)
        << "MultipeerIbgdaTransport: reliable doorbell capability query "
           "failed for NIC "
        << deviceName << ": " << docaErrorToString(queryErr)
        << "; treating the NIC as unsupported";
    return false;
  }

  const bool supported =
      doca_verbs_device_attr_get_send_dbr_mode_no_dbr_ext(deviceAttr) != 0;
  const doca_error_t freeErr = doca_verbs_device_attr_free(deviceAttr);
  if (freeErr != DOCA_SUCCESS) {
    LOG(WARNING) << "MultipeerIbgdaTransport: failed to free DOCA device "
                    "attributes for NIC "
                 << deviceName << ": " << docaErrorToString(freeErr);
  }
  return supported;
}

// ---------------------------------------------------------------------------
// dp_ordering (QPC dp_ordering_0 / dp_ordering_1 / dp_ordering_force)
// ---------------------------------------------------------------------------

// MCCL_IBGDA_QP_ORDERING_SEMANTIC wins over
// MultipeerIbTransportConfig::qpOrderingPolicy only when it holds something
// other than its registered default. Comparing against the generated
// _DEFAULTCVARVALUE rather than a literal also covers the case where
// ncclCvarInit() was never called: prims is driven from benchmarks and unit
// tests that do not run it, and both globals are then still zero-initialized
// and therefore equal.
//
// This is why "auto" is the FIRST choice in the cvar's yaml, and so the one
// that gets enum value 0. A binary that never calls ncclCvarInit() reads the
// cvar as zero; with auto at 0 that lands on the same policy as the registered
// default, by construction rather than by coincidence. Were the default any
// other choice, every benchmark would silently run a different policy from
// production -- which is exactly the failure this A/B hit once already.
IbQpOrderingPolicy resolveQpOrderingPolicy(
    const MultipeerIbgdaTransportConfig& config) {
  if (MCCL_IBGDA_QP_ORDERING_SEMANTIC ==
      MCCL_IBGDA_QP_ORDERING_SEMANTIC_DEFAULTCVARVALUE) {
    return config.qpOrderingPolicy;
  }
  using OrderingCvar = decltype(MCCL_IBGDA_QP_ORDERING_SEMANTIC);
  switch (MCCL_IBGDA_QP_ORDERING_SEMANTIC) {
    case OrderingCvar::ibta:
      return IbQpOrderingPolicy::Ibta;
    case OrderingCvar::ibta_forced:
      return IbQpOrderingPolicy::IbtaForced;
    case OrderingCvar::ooo_rw:
      return IbQpOrderingPolicy::OooRw;
    case OrderingCvar::ooo_all:
      return IbQpOrderingPolicy::OooAll;
    case OrderingCvar::auto_:
      break;
  }
  return IbQpOrderingPolicy::Auto;
}

doca_verbs_qp_ordering_semantic toDocaOrderingSemantic(
    IbQpOrderingSemantic mode) {
  switch (ibQpOrderingTier(mode)) {
    case 1:
      return DOCA_VERBS_QP_ORDERING_SEMANTIC_OOO_RW;
    case 2:
      return DOCA_VERBS_QP_ORDERING_SEMANTIC_OOO_ALL;
    default:
      return DOCA_VERBS_QP_ORDERING_SEMANTIC_IBTA;
  }
}

// What one NIC will let us do with dp_ordering.
struct NicDpOrderingCap {
  int capTier{0};
  bool forceSupported{false};
};

// A DEVX QUERY_HCA_CAP round trip. Returns nullopt when the NIC cannot be
// asked at all -- the query is mlx5-specific, so a non-mlx5 NIC fails here
// rather than reporting "not capable". `failureReason` receives a
// human-readable explanation in that case.
std::optional<NicDpOrderingCap> queryNicDpOrderingCap(
    ::ibv_context* ibvContext,
    const std::string& deviceName,
    std::string& failureReason) {
  doca_verbs_device_attr* deviceAttr = nullptr;
  const doca_error_t queryErr =
      doca_verbs_query_device(ibvContext, &deviceAttr);
  if (queryErr != DOCA_SUCCESS) {
    failureReason = fmt::format(
        "the DOCA capability query for NIC {} failed: {}",
        deviceName,
        docaErrorToString(queryErr));
    return std::nullopt;
  }
  const NicDpOrderingCap cap{
      static_cast<int>(
          doca_verbs_device_attr_get_dp_ordering_cap_rc(deviceAttr)),
      doca_verbs_device_attr_get_dp_ordering_force_supported(deviceAttr) != 0};
  const doca_error_t freeErr = doca_verbs_device_attr_free(deviceAttr);
  if (freeErr != DOCA_SUCCESS) {
    LOG(WARNING) << "MultipeerIbgdaTransport: failed to free DOCA device "
                    "attributes for NIC "
                 << deviceName << ": " << docaErrorToString(freeErr);
  }
  return cap;
}

// Resolve a requested policy against one NIC's capabilities.
//
// The two request kinds fail in opposite directions on purpose:
//
//   Auto     - never throws. It is the fleet default, so it must not be the
//              reason a job refuses to start, and the capability query itself
//              is mlx5-only. Anything it cannot get, it demotes to Ibta and
//              logs why.
//   explicit - fails closed, exactly as before. Somebody named a tier because
//              they are measuring it; a silent downgrade would turn their A/B
//              into a no-op that reads as "OOO does not help". NVIDIA's GDAKI
//              refuses here for the same reason.
IbQpOrderingSemantic resolveQpOrderingSemanticForNic(
    IbQpOrderingPolicy policy,
    ::ibv_context* ibvContext,
    const std::string& deviceName) {
  // Ibta writes nothing to the QPC, so there is nothing to check and no reason
  // to spend a DEVX round trip finding that out.
  if (policy == IbQpOrderingPolicy::Ibta) {
    return IbQpOrderingSemantic::Ibta;
  }

  std::string queryFailure;
  const std::optional<NicDpOrderingCap> cap =
      queryNicDpOrderingCap(ibvContext, deviceName, queryFailure);

  if (ibQpOrderingPolicyIsAuto(policy)) {
    if (!cap.has_value()) {
      LOG(INFO) << "MultipeerIbgdaTransport: qp_ordering_semantic=auto falling "
                   "back to ibta on NIC "
                << deviceName << " because " << queryFailure;
      return IbQpOrderingSemantic::Ibta;
    }
    if (!cap->forceSupported) {
      LOG(INFO) << "MultipeerIbgdaTransport: qp_ordering_semantic=auto falling "
                   "back to ibta on NIC "
                << deviceName
                << ": the NIC does not report cmd_hca_cap_2.dp_ordering_force, "
                   "without which the QPC tier is ignored";
      return IbQpOrderingSemantic::Ibta;
    }
    // Ladder: take the strongest tier this NIC reports, ooo_all first.
    //
    //   ooo_all -> ooo_rw -> ibta
    //
    // ooo_all is the rung worth having: it is the only tier that lifts the
    // ordering fence in front of an atomic, and every prims signal is an
    // ATOMIC_FETCH_AND_ADD. ooo_rw relaxes Read/Write placement only and leaves
    // that fence in place.
    //
    // ConnectX-8 rails report ooo_all; ConnectX-7 reports ooo_rw but not
    // ooo_all (measured: cmd_hca_cap.dp_ordering_ooo_all_rc = 0), which is why
    // the ladder exists rather than a flat "ask for ooo_all". Note the caller
    // then narrows across a rank's NICs, so a rank holding both generations
    // settles on ooo_rw for all of them rather than letting its own NICs
    // disagree.
    //
    // A job spanning ranks with different NIC generations will resolve
    // different tiers and be rejected at peer connect; those jobs must pin
    // MCCL_IBGDA_QP_ORDERING_SEMANTIC=ooo_rw explicitly. That is deliberate --
    // see the mismatch check in doMaterializePeer().
    for (const auto candidate :
         {IbQpOrderingSemantic::OooAll, IbQpOrderingSemantic::OooRw}) {
      if (ibQpOrderingTier(candidate) <= cap->capTier) {
        return candidate;
      }
    }
    LOG(INFO) << "MultipeerIbgdaTransport: qp_ordering_semantic=auto falling "
                 "back to ibta on NIC "
              << deviceName
              << ": the NIC reports no out-of-order placement tier "
                 "(cmd_hca_cap.dp_ordering_ooo_{rw,all}_rc both clear)";
    return IbQpOrderingSemantic::Ibta;
  }

  const IbQpOrderingSemantic mode = ibQpOrderingPolicyToSemantic(policy);
  if (!cap.has_value()) {
    throw std::runtime_error(
        fmt::format(
            "qp_ordering_semantic={} requested but {}",
            ibQpOrderingSemanticName(mode),
            queryFailure));
  }
  if (ibQpOrderingTier(mode) > cap->capTier) {
    throw std::runtime_error(
        fmt::format(
            "qp_ordering_semantic={} needs dp_ordering tier {} but NIC {} "
            "supports at most tier {} (cmd_hca_cap.dp_ordering_ooo_rw_rc / "
            "dp_ordering_ooo_all_rc)",
            ibQpOrderingSemanticName(mode),
            ibQpOrderingTier(mode),
            deviceName,
            cap->capTier));
  }
  if (ibQpOrderingForce(mode) && !cap->forceSupported) {
    throw std::runtime_error(
        fmt::format(
            "qp_ordering_semantic={} needs the QPC dp_ordering_force bit but "
            "NIC {} does not report cmd_hca_cap_2.dp_ordering_force",
            ibQpOrderingSemanticName(mode),
            deviceName));
  }
  return mode;
}
#endif

// Check DOCA error and throw on failure
void checkDocaError(doca_error_t err, const char* msg) {
  if (err != DOCA_SUCCESS) {
    throw std::runtime_error(std::string(msg) + ": " + docaErrorToString(err));
  }
}

} // namespace

// Helper method implementations

void MultipeerIbgdaTransport::initDocaGpu() {
  // CRITICAL: Set CUDA device before any DOCA GPU operations
  cudaError_t cudaErr = cudaSetDevice(config_.cudaDevice);
  if (cudaErr != cudaSuccess) {
    throw std::runtime_error(
        "Failed to set CUDA device: " +
        std::string(cudaGetErrorString(cudaErr)));
  }

  gpuPciBusId_ = GpuNicDiscovery::getCudaPciBusId(config_.cudaDevice);

  VLOG(1) << "MultipeerIbgdaTransport: GPU " << config_.cudaDevice << " PCIe "
          << gpuPciBusId_;

  doca_error_t err = doca_gpu_create(gpuPciBusId_.c_str(), &docaGpu_);
  checkDocaError(err, "Failed to create DOCA GPU context");

  VLOG(1) << "MultipeerIbgdaTransport: DOCA GPU context created: "
          << (void*)docaGpu_;
}

void MultipeerIbgdaTransport::openIbDevice() {
  // Generic NIC bring-up (name resolution, open device + PD, query GID + port,
  // MTU, link layer) is owned by the base; it fills nics_ and localMtu_ using
  // the base-owned gidIndex_.
  openNics();

  // Backend-specific tail: per-NIC DOCA address-handle attributes. addrType is
  // derived from NIC 0's link layer + the configured address family (same for
  // all NICs — same fabric/HCA generation assumed), matching the prior inline
  // behavior.
  nicDoca_.resize(numNics_);
  const doca_verbs_addr_type addrType =
      (nics_[0].linkLayer == ibverbx::IBV_LINK_LAYER_INFINIBAND)
      ? DOCA_VERBS_ADDR_TYPE_IB_NO_GRH
      : ((config_.addressFamily == AddressFamily::IPV4)
             ? DOCA_VERBS_ADDR_TYPE_IPv4
             : DOCA_VERBS_ADDR_TYPE_IPv6);
  for (int n = 0; n < numNics_; ++n) {
#ifndef __HIP_PLATFORM_AMD__
    // Narrow the read/atomic depth to the most restrictive NIC before any QP
    // exists. At the default depth of 1 this returns immediately and issues no
    // capability query.
    maxRdAtomic_ = clampMaxRdAtomicToNic(
        maxRdAtomic_,
        reinterpret_cast<::ibv_context*>(nics_[n].ibvCtx),
        nics_[n].deviceName);
    LOG(INFO) << "MultipeerIbgdaTransport: NIC " << nics_[n].deviceName
              << " max_rd_atomic=" << static_cast<unsigned>(maxRdAtomic_);

    const bool nicReliableDoorbellCapable =
        reliableDoorbellNeedsCapabilityQuery(config_)
        ? nicSupportsReliableDoorbell(
              reinterpret_cast<::ibv_context*>(nics_[n].ibvCtx),
              nics_[n].deviceName)
        : false;
    nicDoca_[n].useReliableDoorbell =
        reliableDoorbellActiveForNic(config_, nicReliableDoorbellCapable);
    LOG(INFO) << "MultipeerIbgdaTransport: NIC " << nics_[n].deviceName
              << " reliable_doorbell_mode="
              << reliableDoorbellModeName(config_.enableReliableDoorbell)
              << " send_dbr_mode="
              << (nicDoca_[n].useReliableDoorbell ? "NO_DBR_HW" : "VALID_DBR");

    // Resolve the dp_ordering tier against this NIC before any QP exists. An
    // explicit policy throws here; auto demotes to Ibta and logs why.
    //
    // Narrow to the least capable NIC: a single qpOrderingSemantic_ drives
    // every NIC on the rank and is what goes on the wire to peers, so under
    // auto one NIC that cannot do OooRw demotes the whole rank rather than
    // leaving the rank's NICs disagreeing with each other. Same in-loop clamp
    // as maxRdAtomic_ above, and monotone for the same reason: auto only ever
    // yields Ibta or OooRw, so taking the weaker of the two is well defined.
    const IbQpOrderingSemantic nicOrdering = resolveQpOrderingSemanticForNic(
        qpOrderingPolicy_,
        reinterpret_cast<::ibv_context*>(nics_[n].ibvCtx),
        nics_[n].deviceName);
    // The first NIC seeds; later NICs can only demote. Seeding rather than
    // clamping from the member's initial value matters because that value is
    // Ibta (tier 0), which nothing can clamp below.
    if (n == 0 ||
        ibQpOrderingTier(nicOrdering) < ibQpOrderingTier(qpOrderingSemantic_)) {
      qpOrderingSemantic_ = nicOrdering;
    }
    // Say "written", not "effective". When we write nothing the firmware
    // supplies its own tier -- on a ConnectX-8 rail with adaptive routing
    // enabled that is OOO_RW, not IBTA. Logging "tier=0" as though it were the
    // QP's state would be actively misleading: it reads as strict ordering when
    // the QP is in fact relaxed for reads and writes.
    LOG(INFO) << "MultipeerIbgdaTransport: NIC " << nics_[n].deviceName
              << " qp_ordering_policy="
              << ibQpOrderingPolicyName(qpOrderingPolicy_)
              << " qp_ordering_semantic="
              << ibQpOrderingSemanticName(qpOrderingSemantic_)
              << " (dp_ordering written tier="
              << ibQpOrderingTier(qpOrderingSemantic_)
              << " force=" << ibQpOrderingForce(qpOrderingSemantic_)
              << (ibQpOrderingIsWireNoOp(qpOrderingSemantic_)
                      ? "; nothing written, QP keeps the firmware default"
                      : "")
              << ")";
#endif

    doca_error_t err = doca_verbs_ah_attr_create(
        reinterpret_cast<::ibv_context*>(nics_[n].ibvCtx), &nicDoca_[n].ahAttr);
    checkDocaError(err, "Failed to create AH attributes");
    err = doca_verbs_ah_attr_set_addr_type(nicDoca_[n].ahAttr, addrType);
    checkDocaError(err, "Failed to set address type");
    err = doca_verbs_ah_attr_set_sgid_index(nicDoca_[n].ahAttr, gidIndex_);
    checkDocaError(err, "Failed to set SGID index");
    err = doca_verbs_ah_attr_set_hop_limit(nicDoca_[n].ahAttr, kHopLimit);
    checkDocaError(err, "Failed to set hop limit");
    // ionic RoCE fabric: the mlx5/NCCL default TC=224 (DSCP 56) lands on a
    // PFC-paused/rate-limited priority that caps a single QP at ~3.9 GB/s; TC=0
    // is unthrottled and restores line rate. Only override the default (224) so
    // an explicit traffic class is still honored.
    uint8_t effectiveTc = config_.trafficClass;
#if defined(NIC_IONIC)
    if (effectiveTc == 224) {
      effectiveTc = 0;
    }
#endif
    err = doca_verbs_ah_attr_set_traffic_class(nicDoca_[n].ahAttr, effectiveTc);
    checkDocaError(err, "Failed to set traffic class");
    err = doca_verbs_ah_attr_set_sl(nicDoca_[n].ahAttr, config_.serviceLevel);
    checkDocaError(err, "Failed to set service level");
  }
}

void MultipeerIbgdaTransport::allocateResources() {
  // Allocate sink buffer for RDMA atomic return values (discarded).
  // DOCA's OPCODE_ATOMIC_FA requires a local address for the fetch-add
  // result. We don't need it, so we use a small "sink" buffer.
  sinkBufferSize_ = sizeof(uint64_t);

#ifdef __HIP_PLATFORM_AMD__
  // On AMD: use host-pinned memory for the sink. AMD doesn't have a
  // direct equivalent of CUDA's `gpuDirectRDMACapable=1` flag (the AMD
  // path uses HSA + DMA-buf for GPU memory RDMA registration). Host-
  // pinned memory works fine for the discarded atomic-FA result and
  // matches what `comms/prims/transport/amd/MultipeerIbgdaTransportAmd.cu` does
  // for the same purpose.
  sinkBufferAllocSize_ = sinkBufferSize_;
  sinkBufferHandle_ = 0;
  hipError_t hipErr =
      hipHostMalloc(&sinkBuffer_, sinkBufferSize_, hipHostMallocDefault);
  if (hipErr != hipSuccess) {
    throw std::runtime_error(
        "Failed to allocate AMD sink buffer: " +
        std::string(hipGetErrorString(hipErr)));
  }
  std::memset(sinkBuffer_, 0, sinkBufferSize_);
#else
  // Uses cuMemCreate with gpuDirectRDMACapable=1 (instead of cudaMalloc /
  // doca_gpu_mem_alloc) so the memory can be registered as an IB MR on
  // aarch64/SMMU platforms (GB200). This matches GIN's ncclCuMemAlloc
  // pattern in gin_host_gdaki.cc.
  if (cuda_driver_lazy_init() != 0) {
    throw std::runtime_error(
        "CUDA driver API not available for sink buffer allocation");
  }

  CUdevice cuDevice;
  CUresult cuErr = pfn_cuDeviceGet(&cuDevice, config_.cudaDevice);
  if (cuErr != CUDA_SUCCESS) {
    throw std::runtime_error(
        "Failed to get CUdevice for device " +
        std::to_string(config_.cudaDevice));
  }

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = cuDevice;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

  int rdmaFlag = 0;
  cuErr = pfn_cuDeviceGetAttribute(
      &rdmaFlag,
      CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
      cuDevice);
  if (cuErr != CUDA_SUCCESS) {
    LOG(WARNING) << "Failed to query GPU Direct RDMA support: " << cuErr;
    rdmaFlag = 0;
  }
  if (rdmaFlag) {
    prop.allocFlags.gpuDirectRDMACapable = 1;
  }

  size_t granularity = 0;
  cuErr = pfn_cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (cuErr != CUDA_SUCCESS) {
    throw std::runtime_error("Failed to get allocation granularity");
  }

  sinkBufferAllocSize_ =
      ((sinkBufferSize_ + granularity - 1) / granularity) * granularity;

  CUmemGenericAllocationHandle handle;
  cuErr = pfn_cuMemCreate(&handle, sinkBufferAllocSize_, &prop, 0);
  if (cuErr != CUDA_SUCCESS) {
    throw std::runtime_error("Failed to create sink buffer allocation");
  }
  sinkBufferHandle_ = static_cast<uint64_t>(handle);

  CUdeviceptr devPtr = 0;
  cuErr =
      pfn_cuMemAddressReserve(&devPtr, sinkBufferAllocSize_, granularity, 0, 0);
  if (cuErr != CUDA_SUCCESS) {
    pfn_cuMemRelease(handle);
    throw std::runtime_error("Failed to reserve address for sink buffer");
  }

  cuErr = pfn_cuMemMap(devPtr, sinkBufferAllocSize_, 0, handle, 0);
  if (cuErr != CUDA_SUCCESS) {
    pfn_cuMemAddressFree(devPtr, sinkBufferAllocSize_);
    pfn_cuMemRelease(handle);
    throw std::runtime_error("Failed to map sink buffer");
  }

  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cuDevice;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuErr = pfn_cuMemSetAccess(devPtr, sinkBufferAllocSize_, &accessDesc, 1);
  if (cuErr != CUDA_SUCCESS) {
    pfn_cuMemUnmap(devPtr, sinkBufferAllocSize_);
    pfn_cuMemAddressFree(devPtr, sinkBufferAllocSize_);
    pfn_cuMemRelease(handle);
    throw std::runtime_error("Failed to set access for sink buffer");
  }

  sinkBuffer_ = reinterpret_cast<void*>(devPtr);

  cudaError_t cudaErr = cudaMemset(sinkBuffer_, 0, sinkBufferSize_);
  if (cudaErr != cudaSuccess) {
    throw std::runtime_error("Failed to zero sink buffer");
  }
#endif
}

void MultipeerIbgdaTransport::registerMemory() {
  auto& symbols = ibverbx::ibvSymbols;
  int accessFlags = ibverbx::IBV_ACCESS_LOCAL_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_WRITE | ibverbx::IBV_ACCESS_REMOTE_READ |
      ibverbx::IBV_ACCESS_REMOTE_ATOMIC;

  // Register the sink buffer (which receives discarded RDMA atomic fetch-add
  // return values) as a zero-based MR (iova=0) on each NIC's PD. Device code
  // addresses it as sinkAddr.addr=0, so the MR must be zero-based: a standard
  // ibv_reg_mr() has IOVA==virtual address, so addr=0 would be out of range →
  // NIC local protection error → QP error → hang. ibv_reg_mr_iova2(pd, addr,
  // length, iova=0, access) maps IOVA [0, length) onto [addr, addr+length).
  // One MR per NIC.
  for (int n = 0; n < numNics_; ++n) {
#if defined(__HIP_PLATFORM_AMD__) && defined(NIC_MLX5)
    // AMD+mlx5: the sink is host-pinned, so there is no GPU dmabuf to export.
    // Register through the symbol table to keep the MR on the same libibverbs
    // that allocated nics_[n].ibvPd and that frees it in cleanup().
    if (symbols.ibv_internal_reg_mr_iova2 == nullptr) {
      throw std::runtime_error("ibv_reg_mr_iova2 is unavailable");
    }
    nicDoca_[n].sinkMr = symbols.ibv_internal_reg_mr_iova2(
        nics_[n].ibvPd, sinkBuffer_, sinkBufferSize_, 0, accessFlags);
#else
    // NVIDIA / AMD+BNXT: DMABUF export (zero-based) with a lazy iova2 fallback.
    auto sinkDmabuf = export_gpu_dmabuf_aligned(sinkBuffer_, sinkBufferSize_);
    if (sinkDmabuf) {
      if (symbols.ibv_internal_reg_dmabuf_mr != nullptr) {
        nicDoca_[n].sinkMr = symbols.ibv_internal_reg_dmabuf_mr(
            nics_[n].ibvPd,
            sinkDmabuf->alignment.dmabufOffset,
            sinkBufferSize_,
            0, // iova=0: zero-based MR
            sinkDmabuf->fd,
            accessFlags);
      }
      close(sinkDmabuf->fd);
    }
    if (!nicDoca_[n].sinkMr) {
      if (symbols.ibv_internal_reg_mr_iova2 == nullptr) {
        throw std::runtime_error("ibv_reg_mr_iova2 is unavailable");
      }
      nicDoca_[n].sinkMr = symbols.ibv_internal_reg_mr_iova2(
          nics_[n].ibvPd, sinkBuffer_, sinkBufferSize_, 0, accessFlags);
    }
#endif
    if (!nicDoca_[n].sinkMr) {
      throw std::runtime_error(
          "Failed to register sink memory region on NIC " + std::to_string(n));
    }

    VLOG(1) << "MultipeerIbgdaTransport: NIC " << n
            << " sink lkey=" << nicDoca_[n].sinkMr->lkey
            << " (zero-based MR, iova=0)";
  }
}
void MultipeerIbgdaTransport::createQpGroups() {
  const int numPeers = nRanks_ - 1;
  const int directionCount = config_.fixedChannelDirectionCount();
  const int mainQpsPerPeerPerNic = config_.fixedChannelMainQpsPerPeerPerNic();
  const int totalMainQpsPerPeer = numNics_ * mainQpsPerPeerPerNic;
  const int companionQpsPerPeerPerNic =
      config_.fixedChannelCompanionQpsPerPeerPerNic();
  const int totalCompanionQpsPerPeer = numNics_ * companionQpsPerPeerPerNic;
  for (auto& nic : nicDoca_) {
    nic.blockQpGroups.resize(
        static_cast<size_t>(numPeers) * companionQpsPerPeerPerNic);
    nic.extraMainQps.clear();
    nic.loopbackCompanionQps.resize(
        static_cast<size_t>(numPeers) * companionQpsPerPeerPerNic);
  }

  // Verify CUDA device is still set correctly
  int currentDevice = -1;
  cudaError_t cudaErr = cudaGetDevice(&currentDevice);
  if (cudaErr != cudaSuccess) {
    throw std::runtime_error(
        "Failed to get CUDA device: " +
        std::string(cudaGetErrorString(cudaErr)));
  }
  VLOG(1) << "MultipeerIbgdaTransport::createQpGroups: current CUDA device="
          << currentDevice << " expected=" << config_.cudaDevice;

  // Query IB device capabilities for debugging (NIC 0 is representative).
  ibverbx::ibv_device_attr devAttr{};
  auto& symbols = ibverbx::ibvSymbols;
  if (symbols.ibv_internal_query_device(nics_[0].ibvCtx, &devAttr) == 0) {
    VLOG(1) << "MultipeerIbgdaTransport: IB device - max_qp=" << devAttr.max_qp
            << " max_cq=" << devAttr.max_cq << " max_mr=" << devAttr.max_mr
            << " max_qp_wr=" << devAttr.max_qp_wr;
  }

  VLOG(1) << "MultipeerIbgdaTransport: creating " << totalMainQpsPerPeer
          << " main QPs/peer and " << totalCompanionQpsPerPeer
          << " companion QPs/peer (" << numNics_
          << " NICs × max_num_channels=" << config_.max_num_channels
          << " × direction_count=" << directionCount
          << " × qpsPerConnection=" << config_.qpsPerConnection
          << ", peers=" << numPeers << ") gpu_dev=" << (void*)docaGpu_
          << " sq_nwqe=" << config_.qpDepth
          << " nic_handler=AUTO mreg_type=DEFAULT";

  for (int peer = 0; peer < numPeers; peer++) {
    createPeerQps(peer);
  }
}

void MultipeerIbgdaTransport::connectQp(
    doca_gpu_verbs_qp_hl* qpHl,
    const IbgdaTransportExchInfo& peerInfo,
    int nic) {
  // Set remote GID in AH attributes (per-NIC: each local NIC has its own
  // AH attr, modified in-place per connection target).
  doca_verbs_gid remoteGid{};
  memcpy(remoteGid.raw, peerInfo.gid, sizeof(remoteGid.raw));
  doca_error_t err =
      doca_verbs_ah_attr_set_gid(nicDoca_[nic].ahAttr, remoteGid);
  checkDocaError(err, "Failed to set remote GID");

  // Query port for IB-specific parameters
  ibverbx::ibv_port_attr portAttr{};
  auto& symbols = ibverbx::ibvSymbols;
  if (symbols.ibv_internal_query_port(nics_[nic].ibvCtx, 1, &portAttr) != 0) {
    LOG(WARNING) << "Failed to query port for IB-specific parameters";
  } else if (portAttr.link_layer == ibverbx::IBV_LINK_LAYER_INFINIBAND) {
    err = doca_verbs_ah_attr_set_dlid(nicDoca_[nic].ahAttr, peerInfo.lid);
    checkDocaError(err, "Failed to set DLID");
  }

  // Create QP attributes for modification
  doca_verbs_qp_attr* qpAttr = nullptr;
  err = doca_verbs_qp_attr_create(&qpAttr);
  checkDocaError(err, "Failed to create QP attributes");
  if (qpAttr == nullptr) {
    throw std::runtime_error("Failed to create QP attributes: qpAttr is null");
  }

  try {
    // Transition to INIT state
    err = doca_verbs_qp_attr_set_next_state(qpAttr, DOCA_VERBS_QP_STATE_INIT);
    checkDocaError(err, "Failed to set next state INIT");
    err = doca_verbs_qp_attr_set_allow_remote_write(qpAttr, 1);
    checkDocaError(err, "Failed to set allow remote write");
    err = doca_verbs_qp_attr_set_allow_remote_read(qpAttr, 1);
    checkDocaError(err, "Failed to set allow remote read");
    err = doca_verbs_qp_attr_set_allow_remote_atomic(
        qpAttr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC);
    checkDocaError(err, "Failed to set allow remote atomic");
    err = doca_verbs_qp_attr_set_port_num(qpAttr, 1);
    checkDocaError(err, "Failed to set port number");

    err = doca_verbs_qp_modify(
        qpHl->qp,
        qpAttr,
        DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
            DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ |
            DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM);
    checkDocaError(err, "Failed to modify QP to INIT");

    // Transition to RTR state
    err = doca_verbs_qp_attr_set_next_state(qpAttr, DOCA_VERBS_QP_STATE_RTR);
    checkDocaError(err, "Failed to set next state RTR");
    // Negotiate path MTU: use the minimum of local and remote active MTU
    auto negotiatedMtu = ibv_mtu_to_doca_mtu(std::min(localMtu_, peerInfo.mtu));
    err = doca_verbs_qp_attr_set_path_mtu(qpAttr, negotiatedMtu);
    checkDocaError(err, "Failed to set MTU");
    err = doca_verbs_qp_attr_set_rq_psn(qpAttr, 0);
    checkDocaError(err, "Failed to set RQ PSN");
    err = doca_verbs_qp_attr_set_dest_qp_num(qpAttr, peerInfo.qpn);
    checkDocaError(err, "Failed to set dest QP number");
    err = doca_verbs_qp_attr_set_ah_attr(qpAttr, nicDoca_[nic].ahAttr);
    checkDocaError(err, "Failed to set AH attributes");
    err = doca_verbs_qp_attr_set_min_rnr_timer(qpAttr, config_.minRnrTimer);
    checkDocaError(err, "Failed to set min RNR timer");
#ifndef __HIP_PLATFORM_AMD__
    // Responder depth (QPC log_rra_max). Without MAX_DEST_RD_ATOMIC in the mask
    // DOCA never writes the field and it stays 0 == one outstanding inbound
    // read/atomic. The default is now 16, so this writes log2(16) == 4; a
    // NIC reporting a smaller max_qp_rd_atom has already clamped maxRdAtomic_
    // down in openIbDevice().
    err = doca_verbs_qp_attr_set_max_dest_rd_atomic(qpAttr, maxRdAtomic_);
    checkDocaError(err, "Failed to set max_dest_rd_atomic");
#endif

    err = doca_verbs_qp_modify(
        qpHl->qp,
        qpAttr,
        DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
            DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
            DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER
#ifndef __HIP_PLATFORM_AMD__
            | DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC
#endif
    );
    checkDocaError(err, "Failed to modify QP to RTR");

    // Transition to RTS state
    err = doca_verbs_qp_attr_set_next_state(qpAttr, DOCA_VERBS_QP_STATE_RTS);
    checkDocaError(err, "Failed to set next state RTS");
    err = doca_verbs_qp_attr_set_sq_psn(qpAttr, 0);
    checkDocaError(err, "Failed to set SQ PSN");
    err = doca_verbs_qp_attr_set_ack_timeout(qpAttr, config_.timeout);
    checkDocaError(err, "Failed to set ACK timeout");
    err = doca_verbs_qp_attr_set_retry_cnt(qpAttr, config_.retryCount);
    checkDocaError(err, "Failed to set retry count");
    err = doca_verbs_qp_attr_set_rnr_retry(qpAttr, config_.rnrRetry);
    checkDocaError(err, "Failed to set RNR retry");
#ifndef __HIP_PLATFORM_AMD__
    // Initiator depth (QPC log_sra_max). This is the one that bounds how many
    // ATOMIC_FA signals prims can have in flight on a QP at once.
    err = doca_verbs_qp_attr_set_max_rd_atomic(qpAttr, maxRdAtomic_);
    checkDocaError(err, "Failed to set max_rd_atomic");
#endif

    err = doca_verbs_qp_modify(
        qpHl->qp,
        qpAttr,
        DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
            DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
            DOCA_VERBS_QP_ATTR_RNR_RETRY
#ifndef __HIP_PLATFORM_AMD__
            | DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC
#endif
    );
    checkDocaError(err, "Failed to modify QP to RTS");
  } catch (const std::runtime_error&) {
    doca_verbs_qp_attr_destroy(qpAttr);
    throw;
  }
  doca_verbs_qp_attr_destroy(qpAttr);

  VLOG(1) << "MultipeerIbgdaTransport: connected QP to remote qpn="
          << peerInfo.qpn;
}

void MultipeerIbgdaTransport::createPeerQps(int peerIndex) {
  for (int nic = 0; nic < numNics_; nic++) {
    doca_gpu_verbs_qp_init_attr_hl mainAttr{};
    mainAttr.gpu_dev = docaGpu_;
    mainAttr.ibpd = reinterpret_cast<::ibv_pd*>(nics_[nic].ibvPd);
    mainAttr.sq_nwqe = config_.qpDepth;
    mainAttr.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
    mainAttr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;
#ifndef __HIP_PLATFORM_AMD__
    mainAttr.send_dbr_mode_ext = nicDoca_[nic].useReliableDoorbell
        ? DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_HW
        : DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;
    // dp_ordering tier is carried on the init attr and applied to the QPC when
    // DOCA moves the QP INIT->RTR. At the Ibta default both fields stay zero
    // (the struct is value-initialized above) and DOCA executes no DEVX_SET
    // for them, so the emitted command is unchanged.
    mainAttr.ordering_semantic = toDocaOrderingSemantic(qpOrderingSemantic_);
    mainAttr.ordering_semantic_force =
        ibQpOrderingForce(qpOrderingSemantic_) ? 1 : 0;
#endif

    doca_gpu_verbs_qp_init_attr_hl loopbackAttr = mainAttr;
    loopbackAttr.sq_nwqe = kLoopbackCompanionQpDepth;
#ifndef __HIP_PLATFORM_AMD__
    loopbackAttr.send_dbr_mode_ext =
        DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;
    // Host side of a pure loopback pair: never traverses the fabric, so it
    // cannot see the reordering that OOO exists to absorb. Note the WAIT and
    // counter ATOMIC_FA ride the *device-visible companion* built inside
    // create_qp_group_hl(), not this QP; that companion is pinned to IBTA in
    // DOCA. This is the responder half of the same pair, so it must match.
    loopbackAttr.ordering_semantic = DOCA_VERBS_QP_ORDERING_SEMANTIC_IBTA;
    loopbackAttr.ordering_semantic_force = 0;
#endif

    auto& nicQps = nicDoca_[nic].blockQpGroups;
    auto& nicLoopback = nicDoca_[nic].loopbackCompanionQps;
    const int companionSlots = config_.fixedChannelCompanionQpsPerPeerPerNic();
    for (int slot = 0; slot < companionSlots; slot++) {
      const int slotIdx = peerIndex * companionSlots + slot;
      doca_error_t err =
          doca_gpu_verbs_create_qp_group_hl(&mainAttr, &nicQps[slotIdx]);
      checkDocaError(err, "Failed to create QP group");
      err = doca_gpu_verbs_create_qp_hl(&loopbackAttr, &nicLoopback[slotIdx]);
      checkDocaError(err, "Failed to create loopback companion QP");
    }
  }
}

void MultipeerIbgdaTransport::connectPeerLoopback(int peerIndex) {
  for (int nic = 0; nic < numNics_; nic++) {
    auto& nicQps = nicDoca_[nic].blockQpGroups;
    auto& nicLoopback = nicDoca_[nic].loopbackCompanionQps;

    IbgdaTransportExchInfo selfInfo;
    memcpy(selfInfo.gid, nics_[nic].localGid.raw, sizeof(selfInfo.gid));
    selfInfo.gidIndex = gidIndex_;
    selfInfo.mtu = localMtu_;
    ibverbx::ibv_port_attr portAttr{};
    auto& symbols = ibverbx::ibvSymbols;
    if (symbols.ibv_internal_query_port(nics_[nic].ibvCtx, 1, &portAttr) == 0) {
      selfInfo.lid = portAttr.lid;
    }

    const int companionSlots = config_.fixedChannelCompanionQpsPerPeerPerNic();
    for (int slot = 0; slot < companionSlots; slot++) {
      const int slotIdx = peerIndex * companionSlots + slot;
      selfInfo.qpn = doca_verbs_qp_get_qpn(nicLoopback[slotIdx]->qp);
      connectQp(&nicQps[slotIdx]->qp_companion, selfInfo, nic);
      selfInfo.qpn = doca_verbs_qp_get_qpn(nicQps[slotIdx]->qp_companion.qp);
      connectQp(nicLoopback[slotIdx], selfInfo, nic);
    }
  }
}

P2pIbgdaTransportBuildParams MultipeerIbgdaTransport::buildPeerTransportParams(
    int peerIndex) const {
  const int mainQpsPerPeerPerNic = config_.fixedChannelMainQpsPerPeerPerNic();
  const int companionSlots = config_.fixedChannelCompanionQpsPerPeerPerNic();
  // Build the device-side send/recv layout from the shared base.
  P2pIbgdaTransportBuildParams params(channelLayoutForPeer(peerIndex));
  params.maxChannels = config_.max_num_channels;
  params.qpDirectionCount = config_.fixedChannelDirectionCount();
  params.qpsPerConnection = config_.qpsPerConnection;
  params.h_nicDeviceIbgdaResources.resize(numNics_);
  for (int n = 0; n < numNics_; ++n) {
    auto& nicSpec = params.h_nicDeviceIbgdaResources[n];
    nicSpec.qps.resize(mainQpsPerPeerPerNic);
    nicSpec.companionQps.resize(companionSlots);
    nicSpec.sinkLkey = NetworkLKey(HostLKey(nicDoca_[n].sinkMr->lkey));
    nicSpec.deviceId = n;
  }

  for (int nic = 0; nic < numNics_; nic++) {
    auto& nicQps = nicDoca_[nic].blockQpGroups;
    auto& nicSpec = params.h_nicDeviceIbgdaResources[nic];
    for (int slot = 0; slot < companionSlots; slot++) {
      const int slotIdx = peerIndex * companionSlots + slot;
      doca_error_t err = doca_gpu_verbs_get_qp_dev(
          nicQps[slotIdx]->qp_main.qp_gverbs, &nicSpec.qps[slot]);
      checkDocaError(err, "Failed to get GPU QP handle");

      err = doca_gpu_verbs_get_qp_dev(
          nicQps[slotIdx]->qp_companion.qp_gverbs, &nicSpec.companionQps[slot]);
      checkDocaError(err, "Failed to get companion GPU QP handle");
    }
  }

  if (config_.numSignalSlots > 0) {
    params.remoteSignalBuf = slotRemoteSignalView(peerIndex);
    params.localSignalBuf = slotLocalSignalView(peerIndex);
    params.numSignalSlots = config_.numSignalSlots;
  }
  if (config_.numCounterSlots > 0) {
    params.counterBuf = slotCounterDeviceView(peerIndex);
    params.discardSignalSlot = slotDiscardSignalRemoteView(peerIndex);
    params.numCounterSlots = config_.numCounterSlots;
  }
  return params;
}

// Main class implementation

MultipeerIbgdaTransport::MultipeerIbgdaTransport(
    int myRank,
    int nRanks,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    const MultipeerIbgdaTransportConfig& config)
    : MultiPeerIbTransport<MultipeerIbgdaTransport>(
          myRank,
          nRanks,
          std::move(bootstrap),
          config) {
  if (config_.max_num_channels < 1) {
    throw std::invalid_argument("max_num_channels must be >= 1");
  }
  if (config_.qpsPerConnection < 1) {
    throw std::invalid_argument("qpsPerConnection must be >= 1");
  }
  if (config_.max_num_channels > kMaxIbGroups) {
    throw std::invalid_argument(
        fmt::format(
            "max_num_channels must be <= {}, got {}",
            kMaxIbGroups,
            config_.max_num_channels));
  }
  if (config_.qpsPerConnection > kMaxIbQpsPerBlockPerNic) {
    throw std::invalid_argument(
        fmt::format(
            "qpsPerConnection must be <= {}, got {}",
            kMaxIbQpsPerBlockPerNic,
            config_.qpsPerConnection));
  }
  const int mainQpsPerPeerPerNic = config_.fixedChannelMainQpsPerPeerPerNic();
  const int directionCount = config_.fixedChannelDirectionCount();
  if (mainQpsPerPeerPerNic > kMaxIbQpsPerPeerPerNic) {
    throw std::invalid_argument(
        fmt::format(
            "max_num_channels * direction_count * qpsPerConnection must be <= "
            "{}, got {} * {} * {} = {}",
            kMaxIbQpsPerPeerPerNic,
            config_.max_num_channels,
            directionCount,
            config_.qpsPerConnection,
            mainQpsPerPeerPerNic));
  }
  if (numNics_ * config_.qpsPerConnection > kIbMaxQpLanesPerChannelDirection) {
    throw std::invalid_argument(
        fmt::format(
            "numNics ({}) * qpsPerConnection ({}) must be <= {}",
            numNics_,
            config_.qpsPerConnection,
            kIbMaxQpLanesPerChannelDirection));
  }
  if (mainQpsPerPeerPerNic * (nRanks_ - 1) * 3 > 1000) {
    LOG(WARNING) << "MultipeerIbgdaTransport: high QP count: "
                 << mainQpsPerPeerPerNic << " main QPs/(peer,NIC) * "
                 << (nRanks_ - 1)
                 << " peers * 3 ~= " << mainQpsPerPeerPerNic * (nRanks_ - 1) * 3
                 << " total QPs (per NIC)";
  }
  try {
#ifndef __HIP_PLATFORM_AMD__
    // Resolve the read/atomic depth (config value, optionally overridden by
    // MCCL_IBGDA_MAX_RD_ATOMIC) before any NIC or QP exists, so a bad value
    // fails immediately.
    maxRdAtomic_ = resolveMaxRdAtomic(config_);

    // Resolve CUDA driver function pointers (NVIDIA-only; AMD doesn't
    // use the CUDA driver API for GPU memory allocation).
    if (cuda_driver_lazy_init() != 0) {
      throw std::runtime_error("CUDA driver not available");
    }
#endif

    // Initialize DOCA GPU context
    initDocaGpu();

#ifndef __HIP_PLATFORM_AMD__
    // Take the requested dp_ordering policy (config value, optionally
    // overridden by MCCL_IBGDA_QP_ORDERING_SEMANTIC). openIbDevice() resolves
    // it against each NIC's capability into qpOrderingSemantic_.
    qpOrderingPolicy_ = resolveQpOrderingPolicy(config_);
#endif

    // Open IB device and create PD
    openIbDevice();

    const int numPeers = nRanks - 1;
    const int companionSlots = config_.fixedChannelCompanionQpsPerPeerPerNic();
    for (auto& nic : nicDoca_) {
      nic.blockQpGroups.resize(static_cast<size_t>(numPeers) * companionSlots);
      nic.extraMainQps.clear();
      nic.loopbackCompanionQps.resize(
          static_cast<size_t>(numPeers) * companionSlots);
    }
    peerMaterialized_.resize(numPeers, false);

    // Allocate and register sink buffer for atomic return values
    allocateResources();
    registerMemory();

    // Allocate send/recv staging buffers when fixed channels are configured.
    // The shared base fills each entry when that peer is materialized.
    if (sendRecvBuffersEnabled()) {
      sendRecvPeerBuffers_.resize(nRanks_ - 1);
    }
  } catch (const std::exception&) {
    // Destructor won't run for a partially-constructed object, so clean up
    // all resources allocated by the init methods above.
    cleanup();
    throw;
  }

  VLOG(1) << "MultipeerIbgdaTransport: rank " << myRank_ << "/" << nRanks_
          << " initialized on GPU " << gpuPciBusId_;
}

MultipeerIbgdaTransport::~MultipeerIbgdaTransport() {
  cleanup();
}

void MultipeerIbgdaTransport::cleanup() {
  auto& symbols = ibverbx::ibvSymbols;

  // Free all GPU memory (transport objects + QP pointer arrays)
  for (auto* ptr : gpuAllocations_) {
    if (ptr != nullptr) {
      cudaError_t err = cudaFree(ptr);
      if (err != cudaSuccess) {
        LOG(WARNING) << "Failed to free GPU memory: "
                     << cudaGetErrorString(err);
      }
    }
  }
  gpuAllocations_.clear();
  peerTransportsGpu_ = nullptr;

  // Free send/recv staging buffers (eager bulks + any lazy per-peer
  // allocations) via the shared base cleanup.
  cleanupSendRecvBuffers();

  // Destroy per-NIC QPs and loopback responders.
  for (auto& nic : nicDoca_) {
    for (auto* qpGroup : nic.blockQpGroups) {
      if (qpGroup != nullptr) {
        doca_gpu_verbs_destroy_qp_group_hl(qpGroup);
      }
    }
    nic.blockQpGroups.clear();
    for (auto* qpHl : nic.extraMainQps) {
      if (qpHl != nullptr) {
        doca_gpu_verbs_destroy_qp_hl(qpHl);
      }
    }
    nic.extraMainQps.clear();
    for (auto* qpHl : nic.loopbackCompanionQps) {
      if (qpHl != nullptr) {
        doca_gpu_verbs_destroy_qp_hl(qpHl);
      }
    }
    nic.loopbackCompanionQps.clear();
  }

  cleanupSignalCounterResources();

  // Destroy user buffer MRs
  for (auto& [_, cached] : registeredBuffers_) {
    // numNics_=1 today; loop is the multi-NIC-ready shape (P2.x fills the
    // rest of mrs[]).
    for (int n = 0; n < numNics_; ++n) {
      if (cached.mrs[n] != nullptr &&
          symbols.ibv_internal_dereg_mr != nullptr) {
        symbols.ibv_internal_dereg_mr(cached.mrs[n]);
      }
    }
  }
  registeredBuffers_.clear();

  // Destroy per-NIC sink MRs. Iterate over actual nicDoca_ entries
  // (vector is empty if cleanup runs before openIbDevice; partial init leaves
  // unset fields as nullptr).
  for (int n = 0; n < static_cast<int>(nicDoca_.size()); ++n) {
    if (nicDoca_[n].sinkMr != nullptr) {
      if (symbols.ibv_internal_dereg_mr != nullptr) {
        symbols.ibv_internal_dereg_mr(nicDoca_[n].sinkMr);
      }
      nicDoca_[n].sinkMr = nullptr;
    }
  }

  // Free sink buffer. NVIDIA: cuMem-allocated with gpuDirectRDMACapable.
  // AMD: hipHostMalloc'd. Shared across NICs — only one allocation,
  // freed after all per-NIC MRs.
  if (sinkBuffer_ != nullptr) {
#ifdef __HIP_PLATFORM_AMD__
    (void)hipHostFree(sinkBuffer_);
#else
    auto devPtr = reinterpret_cast<CUdeviceptr>(sinkBuffer_);
    pfn_cuMemUnmap(devPtr, sinkBufferAllocSize_);
    pfn_cuMemAddressFree(devPtr, sinkBufferAllocSize_);
    pfn_cuMemRelease(
        static_cast<CUmemGenericAllocationHandle>(sinkBufferHandle_));
#endif
    sinkBuffer_ = nullptr;
  }

  // Destroy per-NIC AH attributes
  for (int n = 0; n < static_cast<int>(nicDoca_.size()); ++n) {
    if (nicDoca_[n].ahAttr != nullptr) {
      doca_verbs_ah_attr_destroy(nicDoca_[n].ahAttr);
      nicDoca_[n].ahAttr = nullptr;
    }
  }

  // Destroy per-NIC PDs (bound on nics_, the vector indexed here)
  for (int n = 0; n < static_cast<int>(nics_.size()); ++n) {
    if (nics_[n].ibvPd != nullptr) {
      if (symbols.ibv_internal_dealloc_pd != nullptr) {
        symbols.ibv_internal_dealloc_pd(nics_[n].ibvPd);
      }
      nics_[n].ibvPd = nullptr;
    }
  }

  // Close per-NIC devices (bound on nics_, the vector indexed here)
  for (int n = 0; n < static_cast<int>(nics_.size()); ++n) {
    if (nics_[n].ibvCtx != nullptr) {
      if (symbols.ibv_internal_close_device != nullptr) {
        symbols.ibv_internal_close_device(nics_[n].ibvCtx);
      }
      nics_[n].ibvCtx = nullptr;
    }
  }

  // Destroy DOCA GPU context
  if (docaGpu_ != nullptr) {
    doca_gpu_destroy(docaGpu_);
    docaGpu_ = nullptr;
  }
}

void MultipeerIbgdaTransport::exchange() {
  const int numPeers = nRanks_ - 1;
  peerTransportSize_ = getP2pIbgdaTransportDeviceSize();
  const std::size_t totalBytes = numPeers * peerTransportSize_;
  cudaError_t err = cudaMalloc(&peerTransportsGpu_, totalBytes);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to allocate on-demand device transport array: " +
        std::string(cudaGetErrorString(err)));
  }
  gpuAllocations_.push_back(peerTransportsGpu_);
  err = cudaMemset(peerTransportsGpu_, 0, totalBytes);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to zero on-demand device transport array");
  }
  VLOG(1) << "MultipeerIbgdaTransport: rank " << myRank_
          << " exchange complete (per-peer state deferred to materializePeer)";
}

MultipeerIbgdaDeviceTransport MultipeerIbgdaTransport::getDeviceTransport()
    const {
  return MultipeerIbgdaDeviceTransport(
      myRank_,
      nRanks_,
      DeviceSpan<P2pIbgdaTransportDevice>(peerTransportsGpu_, nRanks_ - 1));
}

P2pIbgdaTransportDevice* MultipeerIbgdaTransport::getP2pTransportDevice(
    int peerRank) {
  if (!isPeerMaterialized(peerRank)) {
    materializePeer(peerRank);
  }
  int peerIndex = rankToPeerIndex(peerRank);
  return reinterpret_cast<P2pIbgdaTransportDevice*>(
      reinterpret_cast<char*>(peerTransportsGpu_) +
      peerIndex * peerTransportSize_);
}

P2pIbgdaTransportDevice* MultipeerIbgdaTransport::getDeviceTransportPtr()
    const {
  return peerTransportsGpu_;
}

P2pIbgdaTransportDevice* MultipeerIbgdaTransport::getP2pTransportDeviceSlot(
    int peerRank) const {
  LOG_FIRST_N(WARNING, 1)
      << "MultipeerIbgdaTransport: Transport[] array is being built with "
      << "possibly unmaterialized IBGDA slots. Call get_device_handle(peers) "
      << "before kernels access those peers.";
  int peerIndex = rankToPeerIndex(peerRank);
  return reinterpret_cast<P2pIbgdaTransportDevice*>(
      reinterpret_cast<char*>(peerTransportsGpu_) +
      peerIndex * peerTransportSize_);
}

int MultipeerIbgdaTransport::getGidIndex() const {
  return gidIndex_;
}

int MultipeerIbgdaTransport::maxGroups() const {
  return config_.max_num_channels;
}

int MultipeerIbgdaTransport::qpsPerBlockPerNic() const {
  return config_.qpsPerConnection;
}

// =============================================================================
// Send/recv buffer lifecycle
// =============================================================================

// Eager send/recv staging allocation, exchange, and cleanup are now provided by
// MultiPeerIbTransportBase (allocateSendRecvBuffersEager(Device) /
// exchangeSendRecvBuffersEager() / cleanupSendRecvBuffers()). The lazy per-peer
// path below fills the inherited sendRecvPeerBuffers_ directly.

PeerQpPayload MultipeerIbgdaTransport::buildLocalQpPayload(
    int peerIndex) const {
  const int mainQpsPerPeerPerNic = config_.fixedChannelMainQpsPerPeerPerNic();
  const int companionSlots = config_.fixedChannelCompanionQpsPerPeerPerNic();
  PeerQpPayload payload{};
  payload.gidIndex = gidIndex_;
  payload.mtu = static_cast<int>(localMtu_);
  payload.numNics = numNics_;
  payload.numQpsPerPeerPerNic = mainQpsPerPeerPerNic;
  payload.maxGroups = config_.max_num_channels;
  payload.qpsPerBlockPerNic = config_.qpsPerConnection;
  payload.qpOrderingSemantic = static_cast<int>(qpOrderingSemantic_);
  payload.maxRdAtomic = static_cast<int>(maxRdAtomic_);

  auto& symbols = ibverbx::ibvSymbols;
  for (int n = 0; n < numNics_; ++n) {
    memcpy(
        payload.nicInfo[n].gid,
        nics_[n].localGid.raw,
        sizeof(payload.nicInfo[n].gid));
    ibverbx::ibv_port_attr portAttr{};
    if (symbols.ibv_internal_query_port(nics_[n].ibvCtx, 1, &portAttr) == 0) {
      payload.nicInfo[n].lid = portAttr.lid;
    }
    auto& nicQps = nicDoca_[n].blockQpGroups;
    for (int slot = 0; slot < companionSlots; slot++) {
      const int slotIdx = peerIndex * companionSlots + slot;
      payload.nicInfo[n].qpns[slot] =
          doca_verbs_qp_get_qpn(nicQps[slotIdx]->qp_main.qp);
    }
  }
  return payload;
}

void MultipeerIbgdaTransport::connectPeerMainQps(
    int peerIndex,
    const PeerQpPayload& remotePayload) {
  for (int nic = 0; nic < numNics_; nic++) {
    auto& nicQps = nicDoca_[nic].blockQpGroups;
    const int companionSlots = config_.fixedChannelCompanionQpsPerPeerPerNic();
    for (int slot = 0; slot < companionSlots; slot++) {
      const int slotIdx = peerIndex * companionSlots + slot;
      IbgdaTransportExchInfo peerInfo;
      peerInfo.qpn = remotePayload.nicInfo[nic].qpns[slot];
      memcpy(
          peerInfo.gid, remotePayload.nicInfo[nic].gid, sizeof(peerInfo.gid));
      peerInfo.gidIndex = remotePayload.gidIndex;
      peerInfo.lid = remotePayload.nicInfo[nic].lid;
      peerInfo.mtu = static_cast<ibverbx::ibv_mtu>(remotePayload.mtu);
      connectQp(&nicQps[slotIdx]->qp_main, peerInfo, nic);
    }
  }
}

void MultipeerIbgdaTransport::cleanupPeerOnFailure(int peerIndex) {
  for (int nic = 0; nic < numNics_; nic++) {
    auto& nicQps = nicDoca_[nic].blockQpGroups;
    auto& nicLoopback = nicDoca_[nic].loopbackCompanionQps;
    const int companionSlots = config_.fixedChannelCompanionQpsPerPeerPerNic();
    for (int slot = 0; slot < companionSlots; slot++) {
      const int slotIdx = peerIndex * companionSlots + slot;
      if (nicQps[slotIdx] != nullptr) {
        doca_gpu_verbs_destroy_qp_group_hl(nicQps[slotIdx]);
        nicQps[slotIdx] = nullptr;
      }
      if (nicLoopback[slotIdx] != nullptr) {
        doca_gpu_verbs_destroy_qp_hl(nicLoopback[slotIdx]);
        nicLoopback[slotIdx] = nullptr;
      }
    }
  }
  cleanupSendRecvBufferForPeer(peerIndex);
  cleanupPeerSignalCounterResources(peerIndex);
  peerMaterialized_[peerIndex] = false;
  if (peerTransportsGpu_ != nullptr && peerTransportSize_ != 0) {
    cudaError_t err = cudaMemset(
        reinterpret_cast<char*>(peerTransportsGpu_) +
            static_cast<std::size_t>(peerIndex) * peerTransportSize_,
        0,
        peerTransportSize_);
    if (err != cudaSuccess) {
      LOG(WARNING) << "Failed to zero failed lazy peer transport slot: "
                   << cudaGetErrorString(err);
    }
  }
}

void MultipeerIbgdaTransport::doMaterializePeer(int peerRank) {
  int peerIndex = rankToPeerIndex(peerRank);

  createPeerQps(peerIndex);

  // Phase 1: exchange QP info, connect QPs.
  auto localQp = buildLocalQpPayload(peerIndex);
  auto remoteQp = exchangeWithPeer(peerRank, localQp, kIbPeerQpExchangeTag);

  if (remoteQp.numNics != numNics_) {
    throw std::runtime_error(
        fmt::format(
            "materializePeer: peer {} numNics={} vs local {}",
            peerRank,
            remoteQp.numNics,
            numNics_));
  }
  // The read/atomic depth is a per-QP-pair property: one end's responder
  // window (log_rra_max) has to cover the other end's initiator window
  // (log_sra_max), so the two ends must have resolved the same value.
  if (remoteQp.maxRdAtomic != static_cast<int>(maxRdAtomic_)) {
    throw std::runtime_error(
        fmt::format(
            "materializePeer: peer {} maxRdAtomic={} vs local maxRdAtomic={}",
            peerRank,
            remoteQp.maxRdAtomic,
            static_cast<int>(maxRdAtomic_)));
  }
  if (remoteQp.maxGroups != config_.max_num_channels ||
      remoteQp.qpsPerBlockPerNic != config_.qpsPerConnection) {
    throw std::runtime_error(
        fmt::format(
            "materializePeer: peer {} maxGroups={} qpsPerBlockPerNic={} "
            "vs local maxGroups={} qpsPerBlockPerNic={}",
            peerRank,
            remoteQp.maxGroups,
            remoteQp.qpsPerBlockPerNic,
            config_.max_num_channels,
            config_.qpsPerConnection));
  }
  // dp_ordering has to match on both ends of a connection: fail closed and name
  // both sides rather than silently let one end reassemble in order while the
  // other lets the fabric spray it.
  //
  // Measured on GB300 (ConnectX-8, mlx5_0, two hosts): a matched pair passes --
  // tier1<->tier1 and tier2<->tier2 both verified clean over 2100 iterations of
  // 64 KiB x 8 writes. BOTH mismatched orientations failed outright with
  // "transport retry counter exceeded". That measurement used
  // MLX5DV_QP_CREATE_OOO_DP, which also enables out-of-order receive-WR
  // handling, so it is not proof that a QPC-bit-only mismatch fails -- but it
  // is the only direct evidence, and it says mismatch is not safe.
  //
  // Hence: fail fast here even under the auto policy, where a mismatch is
  // reachable without anyone misconfiguring anything (a rank whose NIC lacks
  // the capability falls back to ibta while its peers resolve ooo_rw). If
  // mismatch is in fact broken, this error beats an opaque RDMA retry failure
  // later; if it is in fact safe, this is merely conservative and the operator
  // has a documented way out. Both are better than an asymmetry we cannot
  // vouch for.
  if (remoteQp.qpOrderingSemantic != static_cast<int>(qpOrderingSemantic_)) {
    throw std::runtime_error(
        fmt::format(
            "materializePeer: peer {} qpOrderingSemantic={} ({}) vs local "
            "qpOrderingSemantic={} ({}). Ranks resolved different dp_ordering "
            "tiers, which happens when they do not all have the same NIC "
            "capability. Set MCCL_IBGDA_QP_ORDERING_SEMANTIC=ibta on every "
            "rank to pin them all to the firmware default.",
            peerRank,
            remoteQp.qpOrderingSemantic,
            ibQpOrderingSemanticNameFromWire(remoteQp.qpOrderingSemantic),
            static_cast<int>(qpOrderingSemantic_),
            ibQpOrderingSemanticName(qpOrderingSemantic_)));
  }

  connectPeerMainQps(peerIndex, remoteQp);
  connectPeerLoopback(peerIndex);

  // Phase 2: exchange buffer info (acts as QP-ready barrier).
  PeerBufferPayload localBuf{};
  allocateSendRecvBufferForPeer(peerIndex, localBuf, IbCounterStorage::Device);
  allocatePeerSignalCounterResources(
      peerIndex,
      localBuf,
      IbCounterStorage::Device,
      /*allocateDiscardSignal=*/true);
  auto remoteBuf =
      exchangeWithPeer(peerRank, localBuf, kIbPeerBufferExchangeTag);
  applyRemoteSendRecvBuffer(peerIndex, remoteBuf);
  applyRemoteSignalCounterResources(
      peerIndex, remoteBuf, /*hasDiscardSignal=*/true);

  auto params = buildPeerTransportParams(peerIndex);
  writeDeviceTransportSlot(
      peerTransportsGpu_, peerIndex, params, gpuAllocations_);
  peerMaterialized_[peerIndex] = true;

  VLOG(1) << "MultipeerIbgdaTransport: rank " << myRank_
          << " materialized peer " << peerRank;
}

} // namespace comms::prims
