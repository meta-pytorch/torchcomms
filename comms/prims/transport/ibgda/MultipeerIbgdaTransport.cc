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
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"
// NVIDIA-only host-side helpers. On AMD their functionality is provided
// by `comms/prims/transport/amd/DocaCompat.h` (already included via
// `MultipeerIbgdaTransport.h`) which translates `doca_*` to the
// `pipes_gda_*` host APIs in `amd/pipes_gda/PipesGdaHost.{h,cc}`.
#ifndef __HIP_PLATFORM_AMD__
#include "comms/prims/platform/CudaDriverLazy.h"
#include "comms/prims/platform/DocaHostUtils.h"
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
constexpr int kMaxQpLanesPerBlock = 64;

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
    doca_error_t err = doca_verbs_ah_attr_create(
        reinterpret_cast<::ibv_context*>(nics_[n].ibvCtx), &nicDoca_[n].ahAttr);
    checkDocaError(err, "Failed to create AH attributes");
    err = doca_verbs_ah_attr_set_addr_type(nicDoca_[n].ahAttr, addrType);
    checkDocaError(err, "Failed to set address type");
    err = doca_verbs_ah_attr_set_sgid_index(nicDoca_[n].ahAttr, gidIndex_);
    checkDocaError(err, "Failed to set SGID index");
    err = doca_verbs_ah_attr_set_hop_limit(nicDoca_[n].ahAttr, kHopLimit);
    checkDocaError(err, "Failed to set hop limit");
    err = doca_verbs_ah_attr_set_traffic_class(
        nicDoca_[n].ahAttr, config_.trafficClass);
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
    // AMD+mlx5: register directly against the PD's libibverbs, the same policy
    // registerBuffer() uses to avoid the lazy/dlopen'd libibverbs on AMD+mlx5
    // (a separate instance from the PD's). The sink is host-pinned, so unlike
    // the data path there is no GPU dmabuf to export — hence direct
    // ibv_reg_mr_iova2 here vs direct ibv_reg_dmabuf_mr in registerBuffer.
    nicDevices_[n].sinkMr = ibv_reg_mr_iova2(
        nicDevices_[n].ibvPd, sinkBuffer_, sinkBufferSize_, 0, accessFlags);
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
  const int mainQpsPerPeerPerNic =
      config_.maxGroups * config_.qpsPerBlockPerNic;
  const int totalMainQpsPerPeer = numNics_ * mainQpsPerPeerPerNic;
  const int totalCompanionQpsPerPeer = numNics_ * config_.maxGroups;
  for (auto& nic : nicDoca_) {
    nic.blockQpGroups.resize(static_cast<size_t>(numPeers) * config_.maxGroups);
    nic.extraMainQps.resize(
        static_cast<size_t>(numPeers) * config_.maxGroups *
        static_cast<size_t>(config_.qpsPerBlockPerNic - 1));
    nic.loopbackCompanionQps.resize(
        static_cast<size_t>(numPeers) * config_.maxGroups);
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
          << " NICs × maxGroups=" << config_.maxGroups
          << " × qpsPerBlockPerNic=" << config_.qpsPerBlockPerNic
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

    err = doca_verbs_qp_modify(
        qpHl->qp,
        qpAttr,
        DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
            DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
            DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER);
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

    err = doca_verbs_qp_modify(
        qpHl->qp,
        qpAttr,
        DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
            DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
            DOCA_VERBS_QP_ATTR_RNR_RETRY);
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

    doca_gpu_verbs_qp_init_attr_hl loopbackAttr = mainAttr;
    loopbackAttr.sq_nwqe = kLoopbackCompanionQpDepth;

    auto& nicQps = nicDoca_[nic].blockQpGroups;
    auto& nicExtraMainQps = nicDoca_[nic].extraMainQps;
    auto& nicLoopback = nicDoca_[nic].loopbackCompanionQps;
    for (int block = 0; block < config_.maxGroups; block++) {
      const int blockIdx = peerIndex * config_.maxGroups + block;
      doca_error_t err =
          doca_gpu_verbs_create_qp_group_hl(&mainAttr, &nicQps[blockIdx]);
      checkDocaError(err, "Failed to create QP group");
      err = doca_gpu_verbs_create_qp_hl(&loopbackAttr, &nicLoopback[blockIdx]);
      checkDocaError(err, "Failed to create loopback companion QP");
      for (int lane = 1; lane < config_.qpsPerBlockPerNic; ++lane) {
        const int extraIdx = (peerIndex * config_.maxGroups + block) *
                (config_.qpsPerBlockPerNic - 1) +
            (lane - 1);
        err =
            doca_gpu_verbs_create_qp_hl(&mainAttr, &nicExtraMainQps[extraIdx]);
        checkDocaError(err, "Failed to create extra main QP");
      }
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

    for (int block = 0; block < config_.maxGroups; block++) {
      const int blockIdx = peerIndex * config_.maxGroups + block;
      selfInfo.qpn = doca_verbs_qp_get_qpn(nicLoopback[blockIdx]->qp);
      connectQp(&nicQps[blockIdx]->qp_companion, selfInfo, nic);
      selfInfo.qpn = doca_verbs_qp_get_qpn(nicQps[blockIdx]->qp_companion.qp);
      connectQp(nicLoopback[blockIdx], selfInfo, nic);
    }
  }
}

P2pIbgdaTransportBuildParams MultipeerIbgdaTransport::buildPeerTransportParams(
    int peerIndex) const {
  const int mainQpsPerPeerPerNic =
      config_.maxGroups * config_.qpsPerBlockPerNic;
  // Build the device-side send/recv state from the shared base (delegates to
  // the inherited sendRecvPeerBuffers_).
  P2pIbgdaTransportBuildParams params(sendRecvStateForPeer(peerIndex));
  params.maxGroups = config_.maxGroups;
  params.qpsPerBlockPerNic = config_.qpsPerBlockPerNic;
  params.h_nicDeviceIbgdaResources.resize(numNics_);
  for (int n = 0; n < numNics_; ++n) {
    auto& nicSpec = params.h_nicDeviceIbgdaResources[n];
    nicSpec.qps.resize(mainQpsPerPeerPerNic);
    nicSpec.companionQps.resize(config_.maxGroups);
    nicSpec.sinkLkey = NetworkLKey(HostLKey(nicDoca_[n].sinkMr->lkey));
    nicSpec.deviceId = n;
  }

  for (int nic = 0; nic < numNics_; nic++) {
    auto& nicQps = nicDoca_[nic].blockQpGroups;
    auto& nicExtraMainQps = nicDoca_[nic].extraMainQps;
    auto& nicSpec = params.h_nicDeviceIbgdaResources[nic];
    for (int block = 0; block < config_.maxGroups; block++) {
      const int blockIdx = peerIndex * config_.maxGroups + block;
      const int lane0MainIdx = block * config_.qpsPerBlockPerNic;
      doca_error_t err = doca_gpu_verbs_get_qp_dev(
          nicQps[blockIdx]->qp_main.qp_gverbs, &nicSpec.qps[lane0MainIdx]);
      checkDocaError(err, "Failed to get GPU QP handle");

      err = doca_gpu_verbs_get_qp_dev(
          nicQps[blockIdx]->qp_companion.qp_gverbs,
          &nicSpec.companionQps[block]);
      checkDocaError(err, "Failed to get companion GPU QP handle");

      for (int lane = 1; lane < config_.qpsPerBlockPerNic; ++lane) {
        const int extraIdx = (peerIndex * config_.maxGroups + block) *
                (config_.qpsPerBlockPerNic - 1) +
            (lane - 1);
        err = doca_gpu_verbs_get_qp_dev(
            nicExtraMainQps[extraIdx]->qp_gverbs,
            &nicSpec.qps[lane0MainIdx + lane]);
        checkDocaError(err, "Failed to get extra main GPU QP handle");
      }
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
  if (config_.maxGroups < 1) {
    throw std::invalid_argument("maxGroups must be >= 1");
  }
  if (config_.qpsPerBlockPerNic < 1) {
    throw std::invalid_argument("qpsPerBlockPerNic must be >= 1");
  }
  if (config_.maxGroups > kMaxIbGroups) {
    throw std::invalid_argument(
        fmt::format(
            "maxGroups must be <= {}, got {}",
            kMaxIbGroups,
            config_.maxGroups));
  }
  if (config_.qpsPerBlockPerNic > kMaxIbQpsPerBlockPerNic) {
    throw std::invalid_argument(
        fmt::format(
            "qpsPerBlockPerNic must be <= {}, got {}",
            kMaxIbQpsPerBlockPerNic,
            config_.qpsPerBlockPerNic));
  }
  const int mainQpsPerPeerPerNic = config_.numQpsPerPeerPerNic();
  if (mainQpsPerPeerPerNic > kMaxIbQpsPerPeerPerNic) {
    throw std::invalid_argument(
        fmt::format(
            "maxGroups * qpsPerBlockPerNic must be <= {}, got {} * {} = {}",
            kMaxIbQpsPerPeerPerNic,
            config_.maxGroups,
            config_.qpsPerBlockPerNic,
            mainQpsPerPeerPerNic));
  }
  if (!config_.ibLazyConnect &&
      mainQpsPerPeerPerNic > kMaxEagerExchangeQpsPerPeerPerNic) {
    throw std::invalid_argument(
        fmt::format(
            "eager IBGDA allGather exchange supports at most {} QPs per "
            "(peer,NIC); got {}. Enable ibLazyConnect for larger "
            "maxGroups * qpsPerBlockPerNic shapes.",
            kMaxEagerExchangeQpsPerPeerPerNic,
            mainQpsPerPeerPerNic));
  }
  if (config_.sendRecv.has_value() &&
      config_.sendRecv->maxGroups > config_.maxGroups) {
    throw std::invalid_argument(
        fmt::format(
            "sendRecv.maxGroups ({}) must be <= maxGroups ({})",
            config_.sendRecv->maxGroups,
            config_.maxGroups));
  }
  if (numNics_ * config_.qpsPerBlockPerNic > kMaxQpLanesPerBlock) {
    throw std::invalid_argument(
        fmt::format(
            "numNics ({}) * qpsPerBlockPerNic ({}) must be <= {}",
            numNics_,
            config_.qpsPerBlockPerNic,
            kMaxQpLanesPerBlock));
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
    // Resolve CUDA driver function pointers (NVIDIA-only; AMD doesn't
    // use the CUDA driver API for GPU memory allocation).
    if (cuda_driver_lazy_init() != 0) {
      throw std::runtime_error("CUDA driver not available");
    }
#endif

    // Initialize DOCA GPU context
    initDocaGpu();

    // Open IB device and create PD
    openIbDevice();

    if (!config_.ibLazyConnect) {
      // Create QP groups (main + loopback) for all peers
      createQpGroups();
    } else {
      const int numPeers = nRanks - 1;
      for (auto& nic : nicDoca_) {
        nic.blockQpGroups.resize(
            static_cast<size_t>(numPeers) * config.maxGroups);
        nic.extraMainQps.resize(
            static_cast<size_t>(numPeers) * config.maxGroups *
            static_cast<size_t>(config.qpsPerBlockPerNic - 1));
        nic.loopbackCompanionQps.resize(
            static_cast<size_t>(numPeers) * config.maxGroups);
      }
      peerMaterialized_.resize(numPeers, false);
    }

    // Allocate and register sink buffer for atomic return values
    allocateResources();
    registerMemory();

    // Allocate send/recv staging buffers (if configured). Eager mode delegates
    // to the shared base (Device counter: NIC loopback atomic); lazy mode only
    // sizes the inherited per-peer view vector — the shared base
    // allocateSendRecvBufferForPeer() fills it per peer at materialization.
    if (config_.sendRecv.has_value()) {
      if (!config_.ibLazyConnect) {
        allocateSendRecvBuffersEager(IbCounterStorage::Device);
      } else {
        sendRecvPeerBuffers_.resize(nRanks_ - 1);
      }
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
    hipHostFree(sinkBuffer_);
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
  const int mainQpsPerPeerPerNic =
      config_.maxGroups * config_.qpsPerBlockPerNic;
  auto& symbols = ibverbx::ibvSymbols;

  if (config_.ibLazyConnect) {
    peerTransportSize_ = getP2pIbgdaTransportDeviceSize();
    std::size_t totalBytes = numPeers * peerTransportSize_;
    cudaError_t err = cudaMalloc(&peerTransportsGpu_, totalBytes);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          "Failed to allocate lazy device transport array: " +
          std::string(cudaGetErrorString(err)));
    }
    gpuAllocations_.push_back(peerTransportsGpu_);
    err = cudaMemset(peerTransportsGpu_, 0, totalBytes);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to zero lazy device transport array");
    }
    VLOG(1)
        << "MultipeerIbgdaTransport: rank " << myRank_
        << " lazy exchange complete (per-peer state deferred to materializePeer)";
    return;
  }

  // Build this rank's exchange info. Per-NIC GID/LID land in nicInfo[n];
  // gidIndex + MTU are common across NICs (same fabric/HCA generation in
  // multi-NIC platforms). The rank-count guard, allGather, and per-peer
  // topology validation are owned by the base (MultiPeerIbTransport).
  IbgdaTransportExchInfoAll myInfo{};
  myInfo.gidIndex = gidIndex_;
  myInfo.mtu = localMtu_;
  myInfo.numNics = numNics_;
  myInfo.numQpsPerPeerPerNic = mainQpsPerPeerPerNic;
  myInfo.maxGroups = config_.maxGroups;
  myInfo.qpsPerBlockPerNic = config_.qpsPerBlockPerNic;
  for (int n = 0; n < numNics_; ++n) {
    memcpy(
        myInfo.nicInfo[n].gid,
        nics_[n].localGid.raw,
        sizeof(myInfo.nicInfo[n].gid));
    // Query NIC n's port for LID (IB only — RoCE leaves LID as 0).
    ibverbx::ibv_port_attr exchPortAttr{};
    if (symbols.ibv_internal_query_port(nics_[n].ibvCtx, 1, &exchPortAttr) !=
        0) {
      LOG(WARNING) << "Failed to query port for LID on NIC " << n;
    } else {
      myInfo.nicInfo[n].lid = exchPortAttr.lid;
    }
  }

  const int totalQpsPerPeer = numNics_ * mainQpsPerPeerPerNic;
  for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
    const int peerRank = peerIndexToRank(peerIndex);
    for (int nic = 0; nic < numNics_; nic++) {
      const auto& nicQps = nicDoca_[nic].blockQpGroups;
      const auto& nicExtraMainQps = nicDoca_[nic].extraMainQps;
      for (int block = 0; block < config_.maxGroups; block++) {
        const int blockIdx = peerIndex * config_.maxGroups + block;
        const int lane0MainIdx = block * config_.qpsPerBlockPerNic;
        myInfo.nicInfo[nic].qpnForRank[peerRank][lane0MainIdx] =
            doca_verbs_qp_get_qpn(nicQps[blockIdx]->qp_main.qp);
        for (int lane = 1; lane < config_.qpsPerBlockPerNic; ++lane) {
          const int extraIdx = (peerIndex * config_.maxGroups + block) *
                  (config_.qpsPerBlockPerNic - 1) +
              (lane - 1);
          myInfo.nicInfo[nic].qpnForRank[peerRank][lane0MainIdx + lane] =
              doca_verbs_qp_get_qpn(nicExtraMainQps[extraIdx]->qp);
        }
      }
    }
  }

  VLOG(1) << "MultipeerIbgdaTransport: rank " << myRank_
          << " performing allGather exchange (" << totalQpsPerPeer
          << " main QPs/peer = " << numNics_ << " NICs × "
          << mainQpsPerPeerPerNic << " main QPs)";

  // allGather (rank-count guarded) + per-peer topology validation (numNics +
  // numQpsPerPeerPerNic) are owned by the base.
  std::vector<IbgdaTransportExchInfoAll> allInfo = allGatherExchInfo(myInfo);
  validatePeerTopology(allInfo);

  // Stash per-peer summary info (slot 0 / NIC 0) for retrospect/debug.
  // Per-slot connection info is computed inline in the connect loop below.
  peerExchInfo_.resize(numPeers);
  for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
    int peerRank = peerIndexToRank(peerIndex);
    const IbgdaTransportExchInfoAll& peerInfo = allInfo[peerRank];

    CHECK_EQ(peerInfo.maxGroups, config_.maxGroups)
        << "Rank " << peerRank << " has maxGroups=" << peerInfo.maxGroups
        << " but local rank " << myRank_ << " has " << config_.maxGroups
        << ". All ranks must use the same maxGroups.";
    CHECK_EQ(peerInfo.qpsPerBlockPerNic, config_.qpsPerBlockPerNic)
        << "Rank " << peerRank
        << " has qpsPerBlockPerNic=" << peerInfo.qpsPerBlockPerNic
        << " but local rank " << myRank_ << " has " << config_.qpsPerBlockPerNic
        << ". All ranks must use the same qpsPerBlockPerNic.";

    // Store common connection info (from QP 0 — same GID/LID for all QPs)
    peerExchInfo_[peerIndex].qpn = peerInfo.nicInfo[0].qpnForRank[myRank_][0];
    memcpy(
        peerExchInfo_[peerIndex].gid,
        peerInfo.nicInfo[0].gid,
        sizeof(peerInfo.nicInfo[0].gid));
    peerExchInfo_[peerIndex].gidIndex = peerInfo.gidIndex;
    peerExchInfo_[peerIndex].lid = peerInfo.nicInfo[0].lid;
    peerExchInfo_[peerIndex].mtu = peerInfo.mtu;

    VLOG(1) << "MultipeerIbgdaTransport: received from peer " << peerRank
            << " numNics=" << peerInfo.numNics
            << " maxGroups=" << peerInfo.maxGroups
            << " qpsPerBlockPerNic=" << peerInfo.qpsPerBlockPerNic
            << " slot0_qpn=" << peerExchInfo_[peerIndex].qpn;
  }

  // Connect main QPs + loopback companions for all peers
  for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
    const IbgdaTransportExchInfoAll& peerInfo =
        allInfo[peerIndexToRank(peerIndex)];

    for (int nic = 0; nic < numNics_; nic++) {
      auto& nicQps = nicDoca_[nic].blockQpGroups;
      auto& nicExtraMainQps = nicDoca_[nic].extraMainQps;
      for (int block = 0; block < config_.maxGroups; block++) {
        const int blockIdx = peerIndex * config_.maxGroups + block;
        const int lane0MainIdx = block * config_.qpsPerBlockPerNic;
        IbgdaTransportExchInfo qpPeerInfo;
        qpPeerInfo.qpn =
            peerInfo.nicInfo[nic].qpnForRank[myRank_][lane0MainIdx];
        memcpy(
            qpPeerInfo.gid, peerInfo.nicInfo[nic].gid, sizeof(qpPeerInfo.gid));
        qpPeerInfo.gidIndex = peerInfo.gidIndex;
        qpPeerInfo.lid = peerInfo.nicInfo[nic].lid;
        qpPeerInfo.mtu = peerInfo.mtu;
        connectQp(&nicQps[blockIdx]->qp_main, qpPeerInfo, nic);

        for (int lane = 1; lane < config_.qpsPerBlockPerNic; ++lane) {
          const int extraIdx = (peerIndex * config_.maxGroups + block) *
                  (config_.qpsPerBlockPerNic - 1) +
              (lane - 1);
          qpPeerInfo.qpn =
              peerInfo.nicInfo[nic].qpnForRank[myRank_][lane0MainIdx + lane];
          connectQp(nicExtraMainQps[extraIdx], qpPeerInfo, nic);
        }
      }
    }
    connectPeerLoopback(peerIndex);
  }
  allocateSignalCounterResources(
      IbCounterStorage::Device, /*allocateDiscardSignal=*/true);

  exchangeSendRecvBuffersEager();

  // Build device transports on GPU
  std::vector<P2pIbgdaTransportBuildParams> buildParams;
  buildParams.reserve(numPeers);
  for (int peer = 0; peer < numPeers; peer++) {
    buildParams.emplace_back(buildPeerTransportParams(peer));
  }

  peerTransportsGpu_ =
      buildDeviceTransportsOnGpu(buildParams, numPeers, gpuAllocations_);
  peerTransportSize_ = getP2pIbgdaTransportDeviceSize();

  VLOG(1) << "MultipeerIbgdaTransport: rank " << myRank_
          << " exchange complete, connected to " << numPeers << " peers"
          << " (" << mainQpsPerPeerPerNic << " main QPs/(peer,NIC) × "
          << numNics_ << " NICs)";
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
  if (config_.ibLazyConnect && !isPeerMaterialized(peerRank)) {
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
  if (config_.ibLazyConnect) {
    LOG_FIRST_N(WARNING, 1)
        << "MultipeerIbgdaTransport: lazy mode is enabled but "
        << "Transport[] array is being built with unmaterialized IBGDA "
        << "slots. Algorithms using Transport[] directly (DeviceWindow, "
        << "AllToAllv) should disable lazy mode (ibLazyConnect=false). "
        << "Algorithms using getP2pTransportDevice() per peer (Ring, "
        << "SendRecv) are unaffected.";
  }
  int peerIndex = rankToPeerIndex(peerRank);
  return reinterpret_cast<P2pIbgdaTransportDevice*>(
      reinterpret_cast<char*>(peerTransportsGpu_) +
      peerIndex * peerTransportSize_);
}

int MultipeerIbgdaTransport::getGidIndex() const {
  return gidIndex_;
}

int MultipeerIbgdaTransport::maxGroups() const {
  return config_.maxGroups;
}

int MultipeerIbgdaTransport::qpsPerBlockPerNic() const {
  return config_.qpsPerBlockPerNic;
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
  const int mainQpsPerPeerPerNic =
      config_.maxGroups * config_.qpsPerBlockPerNic;
  PeerQpPayload payload{};
  payload.gidIndex = gidIndex_;
  payload.mtu = static_cast<int>(localMtu_);
  payload.numNics = numNics_;
  payload.numQpsPerPeerPerNic = mainQpsPerPeerPerNic;
  payload.maxGroups = config_.maxGroups;
  payload.qpsPerBlockPerNic = config_.qpsPerBlockPerNic;

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
    auto& nicExtraMainQps = nicDoca_[n].extraMainQps;
    for (int block = 0; block < config_.maxGroups; block++) {
      const int blockIdx = peerIndex * config_.maxGroups + block;
      const int lane0MainIdx = block * config_.qpsPerBlockPerNic;
      payload.nicInfo[n].qpns[lane0MainIdx] =
          doca_verbs_qp_get_qpn(nicQps[blockIdx]->qp_main.qp);
      for (int lane = 1; lane < config_.qpsPerBlockPerNic; ++lane) {
        const int extraIdx = (peerIndex * config_.maxGroups + block) *
                (config_.qpsPerBlockPerNic - 1) +
            (lane - 1);
        payload.nicInfo[n].qpns[lane0MainIdx + lane] =
            doca_verbs_qp_get_qpn(nicExtraMainQps[extraIdx]->qp);
      }
    }
  }
  return payload;
}

void MultipeerIbgdaTransport::connectPeerMainQps(
    int peerIndex,
    const PeerQpPayload& remotePayload) {
  for (int nic = 0; nic < numNics_; nic++) {
    auto& nicQps = nicDoca_[nic].blockQpGroups;
    auto& nicExtraMainQps = nicDoca_[nic].extraMainQps;
    for (int block = 0; block < config_.maxGroups; block++) {
      const int blockIdx = peerIndex * config_.maxGroups + block;
      const int lane0MainIdx = block * config_.qpsPerBlockPerNic;
      IbgdaTransportExchInfo peerInfo;
      peerInfo.qpn = remotePayload.nicInfo[nic].qpns[lane0MainIdx];
      memcpy(
          peerInfo.gid, remotePayload.nicInfo[nic].gid, sizeof(peerInfo.gid));
      peerInfo.gidIndex = remotePayload.gidIndex;
      peerInfo.lid = remotePayload.nicInfo[nic].lid;
      peerInfo.mtu = static_cast<ibverbx::ibv_mtu>(remotePayload.mtu);
      connectQp(&nicQps[blockIdx]->qp_main, peerInfo, nic);

      for (int lane = 1; lane < config_.qpsPerBlockPerNic; ++lane) {
        const int extraIdx = (peerIndex * config_.maxGroups + block) *
                (config_.qpsPerBlockPerNic - 1) +
            (lane - 1);
        peerInfo.qpn = remotePayload.nicInfo[nic].qpns[lane0MainIdx + lane];
        connectQp(nicExtraMainQps[extraIdx], peerInfo, nic);
      }
    }
  }
}

void MultipeerIbgdaTransport::cleanupPeerOnFailure(int peerIndex) {
  for (int nic = 0; nic < numNics_; nic++) {
    auto& nicQps = nicDoca_[nic].blockQpGroups;
    auto& nicExtraMainQps = nicDoca_[nic].extraMainQps;
    auto& nicLoopback = nicDoca_[nic].loopbackCompanionQps;
    for (int block = 0; block < config_.maxGroups; block++) {
      const int blockIdx = peerIndex * config_.maxGroups + block;
      if (nicQps[blockIdx] != nullptr) {
        doca_gpu_verbs_destroy_qp_group_hl(nicQps[blockIdx]);
        nicQps[blockIdx] = nullptr;
      }
      if (nicLoopback[blockIdx] != nullptr) {
        doca_gpu_verbs_destroy_qp_hl(nicLoopback[blockIdx]);
        nicLoopback[blockIdx] = nullptr;
      }
      for (int lane = 1; lane < config_.qpsPerBlockPerNic; ++lane) {
        const int extraIdx = (peerIndex * config_.maxGroups + block) *
                (config_.qpsPerBlockPerNic - 1) +
            (lane - 1);
        if (nicExtraMainQps[extraIdx] != nullptr) {
          doca_gpu_verbs_destroy_qp_hl(nicExtraMainQps[extraIdx]);
          nicExtraMainQps[extraIdx] = nullptr;
        }
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
  if (remoteQp.maxGroups != config_.maxGroups ||
      remoteQp.qpsPerBlockPerNic != config_.qpsPerBlockPerNic) {
    throw std::runtime_error(
        fmt::format(
            "materializePeer: peer {} maxGroups={} qpsPerBlockPerNic={} "
            "vs local maxGroups={} qpsPerBlockPerNic={}",
            peerRank,
            remoteQp.maxGroups,
            remoteQp.qpsPerBlockPerNic,
            config_.maxGroups,
            config_.qpsPerBlockPerNic));
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
