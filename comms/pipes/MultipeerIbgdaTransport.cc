// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultipeerIbgdaTransport.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>

#include "comms/pipes/CudaDriverLazy.h"
#include "comms/pipes/DocaHostUtils.h"
#include "comms/pipes/IbverbsLazy.h"
#include "comms/pipes/MultipeerIbgdaDeviceTransport.cuh"
#include "comms/pipes/MultipeerIbgdaTransportCuda.cuh"
#include "comms/pipes/rdma/NicDiscovery.h"

namespace comms::pipes {

namespace {

constexpr int kDefaultGidIndex = 3; // Default GID index
constexpr int kHopLimit = 255;

// Companion QPs use the same init attributes as main QPs but with smaller
// depth since they only carry WAIT + atomic operations (2 WQEs per round).
constexpr uint32_t kCompanionQpDepth = 32;

// Convert ibv_mtu enum to doca_verbs_mtu_size enum.
doca_verbs_mtu_size ibv_mtu_to_doca_mtu(enum ibv_mtu ibvMtu) {
  switch (ibvMtu) {
    case IBV_MTU_256:
      return DOCA_VERBS_MTU_SIZE_256_BYTES;
    case IBV_MTU_512:
      return DOCA_VERBS_MTU_SIZE_512_BYTES;
    case IBV_MTU_1024:
      return DOCA_VERBS_MTU_SIZE_1K_BYTES;
    case IBV_MTU_2048:
      return DOCA_VERBS_MTU_SIZE_2K_BYTES;
    case IBV_MTU_4096:
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

  gidIndex_ = config_.gidIndex.value_or(kDefaultGidIndex);
}

void MultipeerIbgdaTransport::openIbDevice() {
  nicDevices_.resize(numNics_);

  // Get all IB devices via DOCA's dlopen wrapper
  int numDevices = 0;
  ibv_device** deviceList = nullptr;
  doca_error_t docaRet =
      doca_verbs_wrapper_ibv_get_device_list(&numDevices, &deviceList);
  if (docaRet != DOCA_SUCCESS || !deviceList || numDevices == 0) {
    throw std::runtime_error("No IB devices found");
  }

  // Resolve nicDevices_[0..numNics_).deviceName — config override first,
  // then topology-aware auto-discovery.
  //
  // Priority 1: Explicit GPU-to-NIC mapping from config (vector per GPU,
  // entries [0..numNics_) used in order — first is preferred).
  auto it = config_.gpuNicMap.find(config_.cudaDevice);
  if (it != config_.gpuNicMap.end() && !it->second.empty()) {
    const auto& names = it->second;
    if (static_cast<int>(names.size()) < numNics_) {
      throw std::runtime_error(
          fmt::format(
              "config.gpuNicMap[{}] supplies {} NIC(s) but numNics_={}; "
              "provide at least numNics_ NIC names",
              config_.cudaDevice,
              names.size(),
              numNics_));
    }
    for (int n = 0; n < numNics_; ++n) {
      nicDevices_[n].deviceName = names[n];
    }
    VLOG(1) << "MultipeerIbgdaTransport: using config.gpuNicMap for GPU "
            << config_.cudaDevice << " -> " << nicDevices_[0].deviceName
            << (numNics_ > 1 ? " (+ " + std::to_string(numNics_ - 1) +
                        " more for multi-NIC)"
                             : "");
  }

  // Priority 2: Auto-discovery (top-numNics_ candidates by NUMA affinity).
  if (nicDevices_[0].deviceName.empty()) {
    auto discovery = GpuNicDiscovery(config_.cudaDevice, config_.ibHca);
    const auto& candidates = discovery.getCandidates();
    if (static_cast<int>(candidates.size()) < numNics_) {
      throw std::runtime_error(
          fmt::format(
              "NIC auto-discovery found {} candidate(s) for GPU {} but "
              "numNics_={}; set config.gpuNicMap or config.ibHca to expose "
              "additional NICs",
              candidates.size(),
              config_.cudaDevice,
              numNics_));
    }
    for (int n = 0; n < numNics_; ++n) {
      nicDevices_[n].deviceName = candidates[n].name;
    }
    VLOG(1) << "MultipeerIbgdaTransport: auto-discovered NIC "
            << nicDevices_[0].deviceName << " for GPU device "
            << config_.cudaDevice;
  }

  // Open + setup each NIC: find by name, open ctx, alloc PD, query GID +
  // port, create AH attributes.
  doca_verbs_addr_type addrType = DOCA_VERBS_ADDR_TYPE_IB_NO_GRH;
  for (int n = 0; n < numNics_; ++n) {
    int nicIdx = -1;
    for (int i = 0; i < numDevices; i++) {
      const char* devName = nullptr;
      doca_verbs_wrapper_ibv_get_device_name(deviceList[i], &devName);
      if (devName && nicDevices_[n].deviceName == devName) {
        nicIdx = i;
        break;
      }
    }
    if (nicIdx < 0) {
      doca_verbs_wrapper_ibv_free_device_list(deviceList);
      throw std::runtime_error(
          "Specified NIC not found: " + nicDevices_[n].deviceName);
    }
    VLOG(1) << "MultipeerIbgdaTransport: NIC " << n << " = "
            << nicDevices_[n].deviceName << " at device-list index " << nicIdx;

    docaRet = doca_verbs_wrapper_ibv_open_device(
        deviceList[nicIdx], &nicDevices_[n].ibvCtx);
    if (docaRet != DOCA_SUCCESS || !nicDevices_[n].ibvCtx) {
      doca_verbs_wrapper_ibv_free_device_list(deviceList);
      throw std::runtime_error(
          "Failed to open IB device: " + nicDevices_[n].deviceName);
    }

    docaRet = doca_verbs_wrapper_ibv_alloc_pd(
        nicDevices_[n].ibvCtx, &nicDevices_[n].ibvPd);
    if (docaRet != DOCA_SUCCESS || !nicDevices_[n].ibvPd) {
      doca_verbs_wrapper_ibv_free_device_list(deviceList);
      throw std::runtime_error(
          "Failed to allocate protection domain on NIC " +
          nicDevices_[n].deviceName);
    }

    docaRet = doca_verbs_wrapper_ibv_query_gid(
        nicDevices_[n].ibvCtx, 1, gidIndex_, &nicDevices_[n].localGid);
    if (docaRet != DOCA_SUCCESS) {
      doca_verbs_wrapper_ibv_free_device_list(deviceList);
      throw std::runtime_error(
          "Failed to query GID at index " + std::to_string(gidIndex_) +
          " on NIC " + nicDevices_[n].deviceName);
    }

    auto gidStr = fmt::format(
        "{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:"
        "{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}",
        nicDevices_[n].localGid.raw[0],
        nicDevices_[n].localGid.raw[1],
        nicDevices_[n].localGid.raw[2],
        nicDevices_[n].localGid.raw[3],
        nicDevices_[n].localGid.raw[4],
        nicDevices_[n].localGid.raw[5],
        nicDevices_[n].localGid.raw[6],
        nicDevices_[n].localGid.raw[7],
        nicDevices_[n].localGid.raw[8],
        nicDevices_[n].localGid.raw[9],
        nicDevices_[n].localGid.raw[10],
        nicDevices_[n].localGid.raw[11],
        nicDevices_[n].localGid.raw[12],
        nicDevices_[n].localGid.raw[13],
        nicDevices_[n].localGid.raw[14],
        nicDevices_[n].localGid.raw[15]);
    VLOG(1) << "MultipeerIbgdaTransport: NIC " << n << " GID[" << gidIndex_
            << "] = " << gidStr;

    ibv_port_attr portAttr{};
    docaRet =
        doca_verbs_wrapper_ibv_query_port(nicDevices_[n].ibvCtx, 1, &portAttr);
    if (docaRet != DOCA_SUCCESS) {
      doca_verbs_wrapper_ibv_free_device_list(deviceList);
      throw std::runtime_error(
          "Failed to query port attributes on NIC " +
          nicDevices_[n].deviceName);
    }

    VLOG(1) << "MultipeerIbgdaTransport: NIC " << n
            << " port 1 state=" << portAttr.state
            << " link_layer=" << (int)portAttr.link_layer
            << " (1=IB, 2=Ethernet) active_mtu=" << portAttr.active_mtu;

    if (portAttr.state != IBV_PORT_ACTIVE) {
      doca_verbs_wrapper_ibv_free_device_list(deviceList);
      throw std::runtime_error(
          "NIC " + nicDevices_[n].deviceName + " port 1 is not active (state=" +
          std::to_string(portAttr.state) + ")");
    }

    // MTU + addr type are common across NICs (same fabric/HCA generation
    // assumed). Capture from NIC 0; cross-check the rest match.
    if (n == 0) {
      localMtu_ = portAttr.active_mtu;
      if (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
        addrType = DOCA_VERBS_ADDR_TYPE_IB_NO_GRH;
      } else {
        addrType = (config_.addressFamily == AddressFamily::IPV4)
            ? DOCA_VERBS_ADDR_TYPE_IPv4
            : DOCA_VERBS_ADDR_TYPE_IPv6;
      }
    } else if (portAttr.active_mtu != localMtu_) {
      LOG(WARNING) << "MultipeerIbgdaTransport: NIC " << n << " ("
                   << nicDevices_[n].deviceName
                   << ") active_mtu=" << portAttr.active_mtu
                   << " differs from NIC 0 active_mtu=" << localMtu_
                   << "; using NIC 0's MTU for negotiation";
    }

    doca_error_t err = doca_verbs_ah_attr_create(
        nicDevices_[n].ibvCtx, &nicDevices_[n].ahAttr);
    checkDocaError(err, "Failed to create AH attributes");
    err = doca_verbs_ah_attr_set_addr_type(nicDevices_[n].ahAttr, addrType);
    checkDocaError(err, "Failed to set address type");
    err = doca_verbs_ah_attr_set_sgid_index(nicDevices_[n].ahAttr, gidIndex_);
    checkDocaError(err, "Failed to set SGID index");
    err = doca_verbs_ah_attr_set_hop_limit(nicDevices_[n].ahAttr, kHopLimit);
    checkDocaError(err, "Failed to set hop limit");
    err = doca_verbs_ah_attr_set_traffic_class(
        nicDevices_[n].ahAttr, config_.trafficClass);
    checkDocaError(err, "Failed to set traffic class");
    err =
        doca_verbs_ah_attr_set_sl(nicDevices_[n].ahAttr, config_.serviceLevel);
    checkDocaError(err, "Failed to set service level");
  }
  doca_verbs_wrapper_ibv_free_device_list(deviceList);
}

void MultipeerIbgdaTransport::allocateResources() {
  // Allocate sink buffer for RDMA atomic return values (discarded).
  // DOCA's OPCODE_ATOMIC_FA requires a local address for the fetch-add
  // result. We don't need it, so we use a small "sink" buffer.
  //
  // Uses cuMemCreate with gpuDirectRDMACapable=1 (instead of cudaMalloc /
  // doca_gpu_mem_alloc) so the memory can be registered as an IB MR on
  // aarch64/SMMU platforms (GB200). This matches GIN's ncclCuMemAlloc
  // pattern in gin_host_gdaki.cc.
  sinkBufferSize_ = sizeof(uint64_t);

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
}

void MultipeerIbgdaTransport::registerMemory() {
  int accessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

  // Register sink buffer as a zero-based MR (iova=0) on each NIC's PD.
  //
  // The sink buffer receives the discarded return value from RDMA atomic
  // fetch-add operations. Device code uses sinkAddr.addr=0 with the sink
  // lkey, so the MR must be zero-based: addr=0 maps to offset 0 within the
  // MR (i.e., the actual sinkBuffer_ GPU address).
  //
  // With a standard ibv_reg_mr(), the IOVA equals the virtual address, so
  // addr=0 would be outside the MR's valid range → NIC local protection
  // error → QP error state → hang.
  //
  // ibv_reg_mr_iova2(pd, addr, length, iova=0, access) creates a zero-based
  // MR where IOVA range [0, length) maps to [addr, addr+length). Matches
  // GIN's gdakiRegMr() pattern (gin_host_gdaki.cc).
  //
  // Multi-NIC: the same physical sink buffer is registered once per PD
  // (one MR per NIC). DMABUF export is shared across NICs (same fd
  // re-imported per PD); on first dmabuf failure we fall back to plain
  // ibv_reg_mr_iova2 across the remaining NICs to keep behavior consistent.
  for (int n = 0; n < numNics_; ++n) {
    auto sinkDmabuf =
        export_gpu_dmabuf_aligned(docaGpu_, sinkBuffer_, sinkBufferSize_);
    if (sinkDmabuf) {
      nicDevices_[n].sinkMr = lazy_ibv_reg_dmabuf_mr(
          nicDevices_[n].ibvPd,
          sinkDmabuf->alignment.dmabufOffset,
          sinkBufferSize_,
          0, // iova=0: zero-based MR
          sinkDmabuf->fd,
          accessFlags);
      close(sinkDmabuf->fd);
    }
    if (!nicDevices_[n].sinkMr) {
      nicDevices_[n].sinkMr = lazy_ibv_reg_mr_iova2(
          nicDevices_[n].ibvPd, sinkBuffer_, sinkBufferSize_, 0, accessFlags);
      if (!nicDevices_[n].sinkMr) {
        throw std::runtime_error(
            "Failed to register sink memory region on NIC " +
            std::to_string(n));
      }
    }

    VLOG(1) << "MultipeerIbgdaTransport: NIC " << n
            << " sink lkey=" << nicDevices_[n].sinkMr->lkey
            << " (zero-based MR, iova=0)";
  }
}
void MultipeerIbgdaTransport::createQpGroups() {
  const int numPeers = nRanks_ - 1;
  const int numQps = config_.numQpsPerPeer;
  const int totalQpsPerPeer = numNics_ * numQps;
  const int totalQpGroups = numPeers * totalQpsPerPeer;
  for (auto& nic : nicDevices_) {
    nic.qpGroups.assign(numPeers * numQps, nullptr);
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
  ibv_device_attr devAttr{};
  if (doca_verbs_wrapper_ibv_query_device(nicDevices_[0].ibvCtx, &devAttr) ==
      DOCA_SUCCESS) {
    VLOG(1) << "MultipeerIbgdaTransport: IB device - max_qp=" << devAttr.max_qp
            << " max_cq=" << devAttr.max_cq << " max_mr=" << devAttr.max_mr
            << " max_qp_wr=" << devAttr.max_qp_wr;
  }

  VLOG(1) << "MultipeerIbgdaTransport: creating " << totalQpGroups
          << " QP groups (" << totalQpsPerPeer << " slots/peer = " << numNics_
          << " NICs × " << numQps << " QPs, " << numPeers
          << " peers) gpu_dev=" << (void*)docaGpu_
          << " sq_nwqe=" << config_.qpDepth
          << " nic_handler=AUTO mreg_type=DEFAULT";

  for (int nic = 0; nic < numNics_; nic++) {
    doca_gpu_verbs_qp_init_attr_hl initAttr{};
    initAttr.gpu_dev = docaGpu_;
    initAttr.ibpd = nicDevices_[nic].ibvPd;
    initAttr.sq_nwqe = config_.qpDepth;
    initAttr.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
    initAttr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;

    auto& nicQps = nicDevices_[nic].qpGroups;
    for (int peer = 0; peer < numPeers; peer++) {
      for (int q = 0; q < numQps; q++) {
        const int qpIdx = peer * numQps + q;
        doca_error_t err =
            doca_gpu_verbs_create_qp_group_hl(&initAttr, &nicQps[qpIdx]);
        if (err != DOCA_SUCCESS) {
          LOG(ERROR) << "MultipeerIbgdaTransport: QP group (nic=" << nic
                     << " peer=" << peer << " q=" << q
                     << ") creation failed: " << docaErrorToString(err)
                     << " (code " << (int)err << ")";
          checkDocaError(err, "Failed to create QP group");
        }

        VLOG(1) << "MultipeerIbgdaTransport: created QP group (nic=" << nic
                << " peer=" << peer << " q=" << q << ") main_qpn="
                << doca_verbs_qp_get_qpn(nicQps[qpIdx]->qp_main.qp)
                << " companion_qpn="
                << doca_verbs_qp_get_qpn(nicQps[qpIdx]->qp_companion.qp);
      }
    }
  }
}

void MultipeerIbgdaTransport::createLoopbackCompanionQps() {
  const int numPeers = nRanks_ - 1;
  const int numQps = config_.numQpsPerPeer;
  // One self-loop responder companion QP per QP group, on the SAME NIC's
  // PD as the active companion (loopback only works within a single NIC).
  for (auto& nic : nicDevices_) {
    nic.loopbackCompanionQps.assign(numPeers * numQps, nullptr);
  }

  VLOG(1) << "MultipeerIbgdaTransport: creating "
          << numNics_ * numPeers * numQps
          << " loopback companion QPs with depth=" << kCompanionQpDepth;

  for (int nic = 0; nic < numNics_; nic++) {
    doca_gpu_verbs_qp_init_attr_hl initAttr{};
    initAttr.gpu_dev = docaGpu_;
    initAttr.ibpd = nicDevices_[nic].ibvPd;
    initAttr.sq_nwqe = kCompanionQpDepth;
    initAttr.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
    initAttr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;

    auto& nicLoopback = nicDevices_[nic].loopbackCompanionQps;
    for (int peer = 0; peer < numPeers; peer++) {
      for (int q = 0; q < numQps; q++) {
        const int qpIdx = peer * numQps + q;
        doca_error_t err =
            doca_gpu_verbs_create_qp_hl(&initAttr, &nicLoopback[qpIdx]);
        if (err != DOCA_SUCCESS) {
          LOG(ERROR) << "MultipeerIbgdaTransport: loopback companion QP (nic="
                     << nic << " peer=" << peer << " q=" << q
                     << ") creation failed: " << docaErrorToString(err)
                     << " (code " << (int)err << ")";
          checkDocaError(err, "Failed to create loopback companion QP");
        }

        VLOG(1) << "MultipeerIbgdaTransport: created loopback companion QP "
                   "(nic="
                << nic << " peer=" << peer << " q=" << q
                << ") qpn=" << doca_verbs_qp_get_qpn(nicLoopback[qpIdx]->qp);
      }
    }
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
      doca_verbs_ah_attr_set_gid(nicDevices_[nic].ahAttr, remoteGid);
  checkDocaError(err, "Failed to set remote GID");

  // Query port for IB-specific parameters
  ibv_port_attr portAttr{};
  if (doca_verbs_wrapper_ibv_query_port(
          nicDevices_[nic].ibvCtx, 1, &portAttr) != DOCA_SUCCESS) {
    LOG(WARNING) << "Failed to query port for IB-specific parameters";
  } else if (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    err = doca_verbs_ah_attr_set_dlid(nicDevices_[nic].ahAttr, peerInfo.lid);
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
    err = doca_verbs_qp_attr_set_ah_attr(qpAttr, nicDevices_[nic].ahAttr);
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

int MultipeerIbgdaTransport::rankToPeerIndex(int rank) const {
  return (rank < myRank_) ? rank : (rank - 1);
}

int MultipeerIbgdaTransport::peerIndexToRank(int peerIndex) const {
  return (peerIndex < myRank_) ? peerIndex : (peerIndex + 1);
}

// Main class implementation

MultipeerIbgdaTransport::MultipeerIbgdaTransport(
    int myRank,
    int nRanks,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    const MultipeerIbgdaTransportConfig& config)
    : myRank_(myRank),
      nRanks_(nRanks),
      bootstrap_(std::move(bootstrap)),
      config_(config) {
  if (myRank < 0 || myRank >= nRanks) {
    throw std::invalid_argument("Invalid rank");
  }
  if (nRanks < 2) {
    throw std::invalid_argument("Need at least 2 ranks");
  }
  if (config.numQpsPerPeer < 1 || config.numQpsPerPeer > kMaxQpsPerPeer) {
    throw std::invalid_argument(
        fmt::format(
            "numQpsPerPeer must be in [1, {}], got {}",
            kMaxQpsPerPeer,
            config.numQpsPerPeer));
  }
  if (config.numQpsPerPeer * (nRanks - 1) * 3 > 1000) {
    LOG(WARNING) << "MultipeerIbgdaTransport: high QP count: "
                 << config.numQpsPerPeer << " QPs/peer * " << (nRanks - 1)
                 << " peers * 3 = " << config.numQpsPerPeer * (nRanks - 1) * 3
                 << " total QPs";
  }
  try {
    // Resolve CUDA driver function pointers
    if (cuda_driver_lazy_init() != 0) {
      throw std::runtime_error("CUDA driver not available");
    }

    // Initialize DOCA GPU context
    initDocaGpu();

    // Open IB device and create PD
    openIbDevice();

    // Create QP groups (main + companion with shared UAR and core_direct)
    createQpGroups();

    // Create self-loop responder companion QPs for counter loopback
    createLoopbackCompanionQps();

    // Allocate and register sink buffer for atomic return values
    allocateResources();
    registerMemory();

    // Allocate tile sendrecv buffers (if configured)
    allocate_send_recv_buffers();
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

  // Free tile sendrecv buffers
  cleanup_send_recv_buffers();

  // Destroy per-NIC QP groups (main + companion) and loopback responders.
  for (auto& nic : nicDevices_) {
    for (auto* qpGroup : nic.qpGroups) {
      if (qpGroup != nullptr) {
        doca_gpu_verbs_destroy_qp_group_hl(qpGroup);
      }
    }
    nic.qpGroups.clear();
    for (auto* qpHl : nic.loopbackCompanionQps) {
      if (qpHl != nullptr) {
        doca_gpu_verbs_destroy_qp_hl(qpHl);
      }
    }
    nic.loopbackCompanionQps.clear();
  }

  // Deregister and free transport-owned signal/counter buffers.
  // MRs must be deregistered BEFORE cudaFree (correct RDMA teardown order).
  if (signalInboxGpu_ != nullptr) {
    deregisterBuffer(signalInboxGpu_);
    cudaFree(signalInboxGpu_);
    signalInboxGpu_ = nullptr;
  }
  signalRemoteViews_.clear();
  signalLocalViews_.clear();

  if (counterGpu_ != nullptr) {
    deregisterBuffer(counterGpu_);
    cudaFree(counterGpu_);
    counterGpu_ = nullptr;
  }
  counterViews_.clear();

  if (discardSignalGpu_ != nullptr) {
    deregisterBuffer(discardSignalGpu_);
    cudaFree(discardSignalGpu_);
    discardSignalGpu_ = nullptr;
  }
  discardSignalRemoteViews_.clear();

  // Destroy user buffer MRs
  for (auto& [_, cached] : registeredBuffers_) {
    // numNics_=1 today; loop is the multi-NIC-ready shape (P2.x fills the
    // rest of mrs[]).
    for (int n = 0; n < numNics_; ++n) {
      doca_verbs_wrapper_ibv_dereg_mr(cached.mrs[n]);
    }
  }
  registeredBuffers_.clear();

  // Destroy per-NIC sink MRs. Iterate over actual nicDevices_ entries
  // (vector is empty if cleanup runs before openIbDevice; partial init leaves
  // unset fields as nullptr).
  for (int n = 0; n < static_cast<int>(nicDevices_.size()); ++n) {
    if (nicDevices_[n].sinkMr != nullptr) {
      doca_verbs_wrapper_ibv_dereg_mr(nicDevices_[n].sinkMr);
      nicDevices_[n].sinkMr = nullptr;
    }
  }

  // Free sink buffer (cuMem-allocated with gpuDirectRDMACapable). Shared
  // across NICs — only one allocation, freed after all per-NIC MRs.
  if (sinkBuffer_ != nullptr) {
    auto devPtr = reinterpret_cast<CUdeviceptr>(sinkBuffer_);
    pfn_cuMemUnmap(devPtr, sinkBufferAllocSize_);
    pfn_cuMemAddressFree(devPtr, sinkBufferAllocSize_);
    pfn_cuMemRelease(
        static_cast<CUmemGenericAllocationHandle>(sinkBufferHandle_));
    sinkBuffer_ = nullptr;
  }

  // Destroy per-NIC AH attributes
  for (int n = 0; n < static_cast<int>(nicDevices_.size()); ++n) {
    if (nicDevices_[n].ahAttr != nullptr) {
      doca_verbs_ah_attr_destroy(nicDevices_[n].ahAttr);
      nicDevices_[n].ahAttr = nullptr;
    }
  }

  // Destroy per-NIC PDs
  for (int n = 0; n < static_cast<int>(nicDevices_.size()); ++n) {
    if (nicDevices_[n].ibvPd != nullptr) {
      doca_verbs_wrapper_ibv_dealloc_pd(nicDevices_[n].ibvPd);
      nicDevices_[n].ibvPd = nullptr;
    }
  }

  // Close per-NIC devices
  for (int n = 0; n < static_cast<int>(nicDevices_.size()); ++n) {
    if (nicDevices_[n].ibvCtx != nullptr) {
      doca_verbs_wrapper_ibv_close_device(nicDevices_[n].ibvCtx);
      nicDevices_[n].ibvCtx = nullptr;
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
  const int numQps = config_.numQpsPerPeer;

  // Validate rank count for allGather-based exchange
  if (nRanks_ > kMaxRanksForAllGather) {
    throw std::runtime_error(
        fmt::format(
            "Too many ranks ({}) for allGather-based exchange, max is {}",
            nRanks_,
            kMaxRanksForAllGather));
  }

  // Build local exchange info for allGather
  std::vector<IbgdaTransportExchInfoAll> allInfo(nRanks_);

  // Fill in my info at my rank's slot. Per-NIC GID/LID land in nicInfo[n];
  // gidIndex + MTU are common across NICs (same fabric/HCA generation in
  // multi-NIC platforms).
  IbgdaTransportExchInfoAll& myInfo = allInfo[myRank_];
  myInfo.gidIndex = gidIndex_;
  myInfo.mtu = localMtu_;
  myInfo.numNics = numNics_;
  myInfo.numQpsPerNic = numQps;
  for (int n = 0; n < numNics_; ++n) {
    memcpy(
        myInfo.nicInfo[n].gid,
        nicDevices_[n].localGid.raw,
        sizeof(myInfo.nicInfo[n].gid));
    // Query NIC n's port for LID (IB only — RoCE leaves LID as 0).
    ibv_port_attr exchPortAttr{};
    if (doca_verbs_wrapper_ibv_query_port(
            nicDevices_[n].ibvCtx, 1, &exchPortAttr) != DOCA_SUCCESS) {
      LOG(WARNING) << "Failed to query port for LID on NIC " << n;
    } else {
      myInfo.nicInfo[n].lid = exchPortAttr.lid;
    }
  }

  // Fill in per-target QPNs for every (nic, q). NIC-fast interleaving:
  // slot s = q * numNics_ + nic.
  const int totalQpsPerPeer = numNics_ * numQps;
  for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
    const int peerRank = peerIndexToRank(peerIndex);
    for (int nic = 0; nic < numNics_; nic++) {
      const auto& nicQps = nicDevices_[nic].qpGroups;
      for (int q = 0; q < numQps; q++) {
        const int qpIdx = peerIndex * numQps + q;
        myInfo.nicInfo[nic].qpnForRank[peerRank][q] =
            doca_verbs_qp_get_qpn(nicQps[qpIdx]->qp_main.qp);
      }
    }
  }

  VLOG(1) << "MultipeerIbgdaTransport: rank " << myRank_
          << " performing allGather exchange (" << totalQpsPerPeer
          << " slots/peer = " << numNics_ << " NICs × " << numQps << " QPs)";

  // Use allGather to exchange transport info with all ranks
  auto result = bootstrap_
                    ->allGather(
                        allInfo.data(),
                        sizeof(IbgdaTransportExchInfoAll),
                        myRank_,
                        nRanks_)
                    .get();
  if (result != 0) {
    throw std::runtime_error(
        "MultipeerIbgdaTransport::exchange allGather failed");
  }

  // Validate every peer's numNics matches mine — same-rail pairing relies
  // on the symmetric (myRank+peerRank) % numNics offset, which only makes
  // sense when both sides agree on numNics.
  for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
    const int peerRank = peerIndexToRank(peerIndex);
    const auto& peerInfo = allInfo[peerRank];
    if (peerInfo.numNics != numNics_) {
      throw std::runtime_error(
          fmt::format(
              "Peer rank {} reports numNics={} but my numNics={}; all ranks "
              "must agree on numNics for same-rail pairing",
              peerRank,
              peerInfo.numNics,
              numNics_));
    }
  }

  // Stash per-peer summary info (slot 0 / NIC 0) for retrospect/debug.
  // Per-slot connection info is computed inline in the connect loop below.
  peerExchInfo_.resize(numPeers);
  for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
    int peerRank = peerIndexToRank(peerIndex);
    const IbgdaTransportExchInfoAll& peerInfo = allInfo[peerRank];

    CHECK_EQ(peerInfo.numQpsPerNic, numQps)
        << "Rank " << peerRank << " has numQpsPerNic=" << peerInfo.numQpsPerNic
        << " but local rank " << myRank_ << " has " << numQps
        << ". All ranks must use the same numQpsPerNic.";

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
            << " numQps=" << peerInfo.numQpsPerNic
            << " slot0_qpn=" << peerExchInfo_[peerIndex].qpn;
  }

  // Connect main QPs to peers — same-rail pairing: my (nic, q) for peer P
  // talks to P's (nic, q) for me. NIC-fast interleaving: slot s = q * numNics_
  // + nic.
  for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
    const IbgdaTransportExchInfoAll& peerInfo =
        allInfo[peerIndexToRank(peerIndex)];

    for (int nic = 0; nic < numNics_; nic++) {
      auto& nicQps = nicDevices_[nic].qpGroups;
      for (int q = 0; q < numQps; q++) {
        const int qpIdx = peerIndex * numQps + q;
        IbgdaTransportExchInfo qpPeerInfo;
        qpPeerInfo.qpn = peerInfo.nicInfo[nic].qpnForRank[myRank_][q];
        memcpy(
            qpPeerInfo.gid, peerInfo.nicInfo[nic].gid, sizeof(qpPeerInfo.gid));
        qpPeerInfo.gidIndex = peerInfo.gidIndex;
        qpPeerInfo.lid = peerInfo.nicInfo[nic].lid;
        qpPeerInfo.mtu = peerInfo.mtu;
        connectQp(&nicQps[qpIdx]->qp_main, qpPeerInfo, nic);
      }
    }
  }

  // Connect companion QPs as loopback pairs — each slot's companion QP
  // must loopback within the same NIC's PD (loopback can't cross PDs).
  {
    std::vector<IbgdaTransportExchInfo> selfInfoPerNic(numNics_);
    for (int n = 0; n < numNics_; ++n) {
      memcpy(
          selfInfoPerNic[n].gid,
          nicDevices_[n].localGid.raw,
          sizeof(selfInfoPerNic[n].gid));
      selfInfoPerNic[n].gidIndex = gidIndex_;
      selfInfoPerNic[n].mtu = localMtu_;
      ibv_port_attr loopbackPortAttr{};
      if (doca_verbs_wrapper_ibv_query_port(
              nicDevices_[n].ibvCtx, 1, &loopbackPortAttr) == DOCA_SUCCESS) {
        selfInfoPerNic[n].lid = loopbackPortAttr.lid;
      }
    }

    for (int nic = 0; nic < numNics_; nic++) {
      auto& nicQps = nicDevices_[nic].qpGroups;
      auto& nicLoopback = nicDevices_[nic].loopbackCompanionQps;
      for (int peer = 0; peer < numPeers; peer++) {
        for (int q = 0; q < numQps; q++) {
          const int qpIdx = peer * numQps + q;
          IbgdaTransportExchInfo selfInfo = selfInfoPerNic[nic];

          // Connect active companion → loopback responder
          selfInfo.qpn = doca_verbs_qp_get_qpn(nicLoopback[qpIdx]->qp);
          connectQp(&nicQps[qpIdx]->qp_companion, selfInfo, nic);

          // Connect loopback responder → active companion
          selfInfo.qpn = doca_verbs_qp_get_qpn(nicQps[qpIdx]->qp_companion.qp);
          connectQp(nicLoopback[qpIdx], selfInfo, nic);

          VLOG(1) << "MultipeerIbgdaTransport: connected companion QP loopback "
                     "pair (nic="
                  << nic << " peer=" << peer << " q=" << q << ")";
        }
      }
    }
  }

  // ---- Allocate transport-owned signal buffers (if configured) ----
  if (config_.numSignalSlots > 0) {
    // Signal inbox: one contiguous buffer with numSignalSlots per peer.
    // Total size = numPeers * numSignalSlots * sizeof(uint64_t).
    // Each peer writes to its own region via RDMA atomic fetch-add.
    const std::size_t slotsPerPeer =
        static_cast<std::size_t>(config_.numSignalSlots);
    const std::size_t totalSignalBytes =
        static_cast<std::size_t>(numPeers) * slotsPerPeer * sizeof(uint64_t);

    cudaError_t cudaErr = cudaMalloc(&signalInboxGpu_, totalSignalBytes);
    if (cudaErr != cudaSuccess) {
      throw std::runtime_error(
          "Failed to allocate signal inbox: " +
          std::string(cudaGetErrorString(cudaErr)));
    }
    cudaErr = cudaMemset(signalInboxGpu_, 0, totalSignalBytes);
    if (cudaErr != cudaSuccess) {
      throw std::runtime_error("Failed to zero signal inbox");
    }

    // Register and exchange signal inbox
    auto localSignalBuf = registerBuffer(signalInboxGpu_, totalSignalBytes);
    auto remoteSignalBufs = exchangeBuffer(localSignalBuf);

    // Build per-peer views:
    // - remoteSignalViews_[peerIndex] = remote view into peer's inbox at the
    //   region reserved for us (offset = myPeerIndexOnPeer * slotsPerPeer)
    // - signalLocalViews_[peerIndex] = local view into our inbox at the
    //   region where this peer writes (offset = peerIndex * slotsPerPeer)
    signalRemoteViews_.resize(numPeers);
    signalLocalViews_.resize(numPeers);
    for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
      int peerRank = peerIndexToRank(peerIndex);
      int myPeerIndexOnPeer = (myRank_ < peerRank) ? myRank_ : (myRank_ - 1);
      signalRemoteViews_[peerIndex] = remoteSignalBufs[peerIndex].subBuffer(
          static_cast<std::size_t>(myPeerIndexOnPeer) * slotsPerPeer *
          sizeof(uint64_t));
      signalLocalViews_[peerIndex] = localSignalBuf.subBuffer(
          static_cast<std::size_t>(peerIndex) * slotsPerPeer *
          sizeof(uint64_t));
    }

    VLOG(1) << "MultipeerIbgdaTransport: allocated signal inbox "
            << totalSignalBytes << " bytes (" << config_.numSignalSlots
            << " slots/peer, " << numPeers << " peers)";
  }

  // ---- Allocate transport-owned counter buffers (if configured) ----
  if (config_.numCounterSlots > 0) {
    // Counter buffer: local only, no exchange needed.
    // Each peer's companion QP writes to its own counter region.
    const std::size_t slotsPerPeer =
        static_cast<std::size_t>(config_.numCounterSlots);
    const std::size_t totalCounterBytes =
        static_cast<std::size_t>(numPeers) * slotsPerPeer * sizeof(uint64_t);

    cudaError_t cudaErr = cudaMalloc(&counterGpu_, totalCounterBytes);
    if (cudaErr != cudaSuccess) {
      throw std::runtime_error(
          "Failed to allocate counter buffer: " +
          std::string(cudaGetErrorString(cudaErr)));
    }
    cudaErr = cudaMemset(counterGpu_, 0, totalCounterBytes);
    if (cudaErr != cudaSuccess) {
      throw std::runtime_error("Failed to zero counter buffer");
    }

    auto localCounterBuf = registerBuffer(counterGpu_, totalCounterBytes);

    // Build per-peer views (local only)
    counterViews_.resize(numPeers);
    for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
      counterViews_[peerIndex] = localCounterBuf.subBuffer(
          static_cast<std::size_t>(peerIndex) * slotsPerPeer *
          sizeof(uint64_t));
    }

    VLOG(1) << "MultipeerIbgdaTransport: allocated counter buffer "
            << totalCounterBytes << " bytes (" << config_.numCounterSlots
            << " slots/peer, " << numPeers << " peers)";
  }

  // ---- Allocate transport-owned discard-signal buffer (if counter used) ----
  //
  // The discard-signal buffer exists solely so that counter-only puts can be
  // routed through the async signal_counter compound (see
  // P2pIbgdaTransportDevice::put_impl). DOCA verbs has no "counter-only"
  // primitive: signal_counter posts the counter atomic on the companion QP
  // ordered against a FENCEd signal on the primary QP. To use it without a
  // real signal recipient, we need a remote-addressable uint64_t to act as
  // the signal target — peers never read these slots, so the value is
  // garbage by design.
  //
  // Layout: numPeers slots, one per peer that may write to us. Each rank
  // exchanges the buffer addr/rkey; per-peer remote view points to *our*
  // slot in the peer's discard buffer (offset = myPeerIndexOnPeer).
  if (config_.numCounterSlots > 0) {
    const std::size_t totalDiscardBytes =
        static_cast<std::size_t>(numPeers) * sizeof(uint64_t);

    cudaError_t cudaErr = cudaMalloc(&discardSignalGpu_, totalDiscardBytes);
    if (cudaErr != cudaSuccess) {
      throw std::runtime_error(
          "Failed to allocate discard-signal buffer: " +
          std::string(cudaGetErrorString(cudaErr)));
    }
    cudaErr = cudaMemset(discardSignalGpu_, 0, totalDiscardBytes);
    if (cudaErr != cudaSuccess) {
      throw std::runtime_error("Failed to zero discard-signal buffer");
    }

    auto localDiscardBuf = registerBuffer(discardSignalGpu_, totalDiscardBytes);
    auto remoteDiscardBufs = exchangeBuffer(localDiscardBuf);

    discardSignalRemoteViews_.resize(numPeers);
    for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
      int peerRank = peerIndexToRank(peerIndex);
      int myPeerIndexOnPeer = (myRank_ < peerRank) ? myRank_ : (myRank_ - 1);
      discardSignalRemoteViews_[peerIndex] =
          remoteDiscardBufs[peerIndex].subBuffer(
              static_cast<std::size_t>(myPeerIndexOnPeer) * sizeof(uint64_t));
    }

    VLOG(1) << "MultipeerIbgdaTransport: allocated discard-signal buffer "
            << totalDiscardBytes << " bytes (" << numPeers << " peers)";
  }

  exchange_send_recv_buffers();

  // Build device transports on GPU. One NicDeviceIbgdaResourcesBuildSpec
  // per (peer, NIC) carries that NIC's QP pointers and sink lkey.
  std::vector<P2pIbgdaTransportBuildParams> buildParams(numPeers);

  for (int peer = 0; peer < numPeers; peer++) {
    // One NicDeviceIbgdaResourcesBuildSpec per physical NIC. NIC-fast
    // interleaving: slot s = q * numNics_ + nic.
    auto& bp = buildParams[peer];
    bp.h_nicDeviceIbgdaResources.resize(numNics_);
    for (int n = 0; n < numNics_; ++n) {
      auto& nicSpec = bp.h_nicDeviceIbgdaResources[n];
      nicSpec.qps.resize(numQps);
      nicSpec.companionQps.resize(numQps);
      nicSpec.sinkLkey = NetworkLKey(HostLKey(nicDevices_[n].sinkMr->lkey));
      nicSpec.deviceId = n;
    }

    for (int nic = 0; nic < numNics_; nic++) {
      auto& nicQps = nicDevices_[nic].qpGroups;
      auto& nicSpec = bp.h_nicDeviceIbgdaResources[nic];
      for (int q = 0; q < numQps; q++) {
        const int qpIdx = peer * numQps + q;
        doca_error_t err = doca_gpu_verbs_get_qp_dev(
            nicQps[qpIdx]->qp_main.qp_gverbs, &nicSpec.qps[q]);
        checkDocaError(err, "Failed to get GPU QP handle");

        err = doca_gpu_verbs_get_qp_dev(
            nicQps[qpIdx]->qp_companion.qp_gverbs, &nicSpec.companionQps[q]);
        checkDocaError(err, "Failed to get companion GPU QP handle");
      }
    }

    if (config_.numSignalSlots > 0) {
      buildParams[peer].remoteSignalBuf = signalRemoteViews_[peer];
      buildParams[peer].localSignalBuf = signalLocalViews_[peer];
      buildParams[peer].numSignalSlots = config_.numSignalSlots;
    }
    if (config_.numCounterSlots > 0) {
      buildParams[peer].counterBuf = counterViews_[peer];
      buildParams[peer].discardSignalSlot = discardSignalRemoteViews_[peer];
      buildParams[peer].numCounterSlots = config_.numCounterSlots;
    }
    if (!sendRecvPeerBuffers_.empty()) {
      const auto& pb = sendRecvPeerBuffers_[peer];
      buildParams[peer].sendRecvState = IbSendRecvState{
          .sendStagingBuf = pb.sendStaging,
          .recvStagingBuf = pb.remoteRecvStaging,
          .sendStagingPtr = static_cast<char*>(pb.sendStaging.ptr),
          .recvStagingPtr = static_cast<char*>(pb.recvStaging.ptr),
          .localSignalBuf = pb.signal,
          .remoteSignalBuf = pb.remoteSignal,
          .localCounterBuf = pb.counter,
          .stepState = pb.stepState,
          .maxGroups = config_.sendRecv->maxGroups,
          .pipelineDepth = config_.sendRecv->pipelineDepth,
          .dataBufferSize = config_.dataBufferSize,
      };
    }
  }

  peerTransportsGpu_ =
      buildDeviceTransportsOnGpu(buildParams, numPeers, gpuAllocations_);
  peerTransportSize_ = getP2pIbgdaTransportDeviceSize();

  VLOG(1) << "MultipeerIbgdaTransport: rank " << myRank_
          << " exchange complete, connected to " << numPeers << " peers"
          << " (" << numQps << " QPs/peer)";
}

MultipeerIbgdaDeviceTransport MultipeerIbgdaTransport::getDeviceTransport()
    const {
  return MultipeerIbgdaDeviceTransport(
      myRank_,
      nRanks_,
      DeviceSpan<P2pIbgdaTransportDevice>(peerTransportsGpu_, nRanks_ - 1));
}

P2pIbgdaTransportDevice* MultipeerIbgdaTransport::getP2pTransportDevice(
    int peerRank) const {
  int peerIndex = rankToPeerIndex(peerRank);
  // Use byte-level arithmetic since P2pIbgdaTransportDevice is incomplete
  // here
  return reinterpret_cast<P2pIbgdaTransportDevice*>(
      reinterpret_cast<char*>(peerTransportsGpu_) +
      peerIndex * peerTransportSize_);
}

P2pIbgdaTransportDevice* MultipeerIbgdaTransport::getDeviceTransportPtr()
    const {
  return peerTransportsGpu_;
}

int MultipeerIbgdaTransport::numPeers() const {
  return nRanks_ - 1;
}

int MultipeerIbgdaTransport::myRank() const {
  return myRank_;
}

int MultipeerIbgdaTransport::getGidIndex() const {
  return gidIndex_;
}

int MultipeerIbgdaTransport::numQpsPerPeer() const {
  return config_.numQpsPerPeer;
}

IbgdaLocalBuffer MultipeerIbgdaTransport::registerBuffer(
    void* ptr,
    std::size_t size) {
  if (ptr == nullptr || size == 0) {
    throw std::invalid_argument("Invalid buffer pointer or size");
  }

  // Fast path: containment lookup — if [ptr, ptr+size) falls entirely
  // within an existing registration, return the cached per-NIC lkeys
  // without any CUDA driver call.
  auto addr = reinterpret_cast<uintptr_t>(ptr);
  auto it = registeredBuffers_.upper_bound(addr);
  if (it != registeredBuffers_.begin()) {
    --it;
    if (addr + size <= it->first + it->second.allocSize) {
      it->second.refs++;
      VLOG(1) << "MultipeerIbgdaTransport: cache hit for ptr=" << ptr
              << " allocBase=0x" << std::hex << it->first << std::dec
              << " refs=" << it->second.refs;
      NetworkLKeys keys(numNics_);
      for (int n = 0; n < numNics_; ++n) {
        keys[n] = NetworkLKey(HostLKey(it->second.mrs[n]->lkey));
      }
      return IbgdaLocalBuffer(ptr, keys);
    }
  }

  // Cache miss — find the CUDA allocation base and register it on every
  // NIC's PD. Each NIC gets an independent MR over the same physical
  // memory; lkey/rkey differ per NIC.
  CUdeviceptr allocBase = 0;
  size_t allocSize = 0;
  CUresult cuRes =
      pfn_cuMemGetAddressRange(&allocBase, &allocSize, (CUdeviceptr)ptr);
  if (cuRes != CUDA_SUCCESS || allocBase == 0) {
    throw std::runtime_error(
        "registerBuffer: cuMemGetAddressRange failed for ptr");
  }
  int accessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

  CachedMr cached;
  cached.allocSize = allocSize;
  cached.refs = 1;

  // Try DMABUF first per NIC, fall back to plain reg_mr per NIC. If any
  // NIC's registration fails, deregister everything we already registered
  // and propagate the error.
  for (int n = 0; n < numNics_; ++n) {
    ibv_mr* mr = nullptr;
    auto dmabuf = export_gpu_dmabuf_aligned(
        docaGpu_, reinterpret_cast<void*>(allocBase), allocSize);
    if (dmabuf) {
      mr = lazy_ibv_reg_dmabuf_mr(
          nicDevices_[n].ibvPd,
          dmabuf->alignment.dmabufOffset,
          allocSize,
          static_cast<uint64_t>(allocBase),
          dmabuf->fd,
          accessFlags);
      close(dmabuf->fd);
    }
    if (!mr) {
      doca_error_t regErr = doca_verbs_wrapper_ibv_reg_mr(
          nicDevices_[n].ibvPd,
          reinterpret_cast<void*>(allocBase),
          allocSize,
          accessFlags,
          &mr);
      if (regErr != DOCA_SUCCESS || !mr) {
        // Roll back partial registration before throwing.
        for (int j = 0; j < n; ++j) {
          doca_verbs_wrapper_ibv_dereg_mr(cached.mrs[j]);
        }
        throw std::runtime_error(
            "Failed to register buffer with RDMA on NIC " + std::to_string(n));
      }
    }
    cached.mrs[n] = mr;
  }

  VLOG(1) << "MultipeerIbgdaTransport: registered allocation allocBase=0x"
          << std::hex << allocBase << std::dec << " allocSize=" << allocSize
          << " across " << numNics_
          << " NIC(s) (NIC0 lkey=" << cached.mrs[0]->lkey
          << " rkey=" << cached.mrs[0]->rkey << ", requested ptr=" << ptr
          << " size=" << size << ")";

  registeredBuffers_.emplace(static_cast<uintptr_t>(allocBase), cached);

  NetworkLKeys keys(numNics_);
  for (int n = 0; n < numNics_; ++n) {
    keys[n] = NetworkLKey(HostLKey(cached.mrs[n]->lkey));
  }
  return IbgdaLocalBuffer(ptr, keys);
}

void MultipeerIbgdaTransport::deregisterBuffer(void* ptr) {
  // Containment lookup on the ordered map: find the allocation whose base
  // address is <= ptr and whose range covers ptr.  This avoids calling
  // cuMemGetAddressRange, which fails when CUDA has already freed the
  // underlying memory (e.g. PyTorch caching allocator teardown).
  auto addr = reinterpret_cast<uintptr_t>(ptr);
  auto it = registeredBuffers_.upper_bound(addr);
  if (it != registeredBuffers_.begin()) {
    --it;
    if (addr < it->first + it->second.allocSize) {
      it->second.refs--;
      VLOG(1) << "MultipeerIbgdaTransport: deregister ptr=" << ptr
              << " allocBase=0x" << std::hex << it->first << std::dec
              << " refs=" << it->second.refs;
      if (it->second.refs <= 0) {
        for (int n = 0; n < numNics_; ++n) {
          doca_verbs_wrapper_ibv_dereg_mr(it->second.mrs[n]);
        }
        registeredBuffers_.erase(it);
      }
      return;
    }
  }
  LOG(WARNING) << "MultipeerIbgdaTransport: buffer not registered: " << ptr;
}

std::vector<IbgdaRemoteBuffer> MultipeerIbgdaTransport::exchangeBuffer(
    const IbgdaLocalBuffer& localBuf) {
  const int numPeers = nRanks_ - 1;

  // Find the MR for this buffer via its CUDA allocation base.
  CUdeviceptr allocBase = 0;
  size_t allocSize = 0;
  CUresult cuRes = pfn_cuMemGetAddressRange(
      &allocBase, &allocSize, (CUdeviceptr)localBuf.ptr);
  if (cuRes != CUDA_SUCCESS || allocBase == 0) {
    throw std::runtime_error(
        "exchangeBuffer: cuMemGetAddressRange failed for ptr");
  }
  auto it = registeredBuffers_.find(static_cast<uintptr_t>(allocBase));
  if (it == registeredBuffers_.end()) {
    throw std::runtime_error(
        "Buffer not registered - call registerBuffer() first");
  }

  // Allocate buffer for allGather: one entry per rank.
  std::vector<IbgdaBufferExchInfo> allInfo(nRanks_);

  // Write my info at my rank's slot — populate per-NIC rkeys (each PD
  // gave us its own MR for the same physical buffer).
  allInfo[myRank_].addr = reinterpret_cast<uint64_t>(localBuf.ptr);
  allInfo[myRank_].numNics = numNics_;
  for (int n = 0; n < numNics_; ++n) {
    allInfo[myRank_].rkey_per_device[n] = HostRKey(it->second.mrs[n]->rkey);
  }

  // Use allGather to exchange buffer info with all ranks
  auto result =
      bootstrap_
          ->allGather(
              allInfo.data(), sizeof(IbgdaBufferExchInfo), myRank_, nRanks_)
          .get();
  if (result != 0) {
    throw std::runtime_error(
        "MultipeerIbgdaTransport::exchangeBuffer allGather failed");
  }

  // Convert to IbgdaRemoteBuffer vector, extracting peer entries
  // peerIndex maps to ranks: 0..myRank_-1 -> ranks 0..myRank_-1
  //                          myRank_..numPeers-1 -> ranks
  //                          myRank_+1..nRanks_-1
  std::vector<IbgdaRemoteBuffer> peerBuffers(numPeers);
  for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
    int peerRank = peerIndexToRank(peerIndex);
    peerBuffers[peerIndex] = allInfo[peerRank].toRemoteBuffer();
  }

  VLOG(1) << "MultipeerIbgdaTransport: exchanged buffer info with " << numPeers
          << " peers";

  return peerBuffers;
}
int MultipeerIbgdaTransport::nRanks() const {
  return nRanks_;
}

// =============================================================================
// Send/recv buffer lifecycle
// =============================================================================

void MultipeerIbgdaTransport::allocate_send_recv_buffers() {
  if (!config_.sendRecv.has_value()) {
    return;
  }
  const auto& sr = *config_.sendRecv;
  if (sr.pipelineDepth < 1) {
    throw std::invalid_argument("sendRecv.pipelineDepth must be >= 1");
  }
  if (sr.maxGroups < 1) {
    throw std::invalid_argument("sendRecv.maxGroups must be >= 1");
  }
  if (config_.dataBufferSize == 0) {
    throw std::invalid_argument(
        "dataBufferSize must be > 0 when sendRecv is enabled");
  }
  if ((config_.dataBufferSize / sr.maxGroups) < 16) {
    throw std::invalid_argument(
        fmt::format(
            "dataBufferSize / maxGroups must be >= 16, got {} / {} = {}",
            config_.dataBufferSize,
            sr.maxGroups,
            config_.dataBufferSize / sr.maxGroups));
  }

  const int numPeers = nRanks_ - 1;
  const std::size_t stagingPerPeer = sr.pipelineDepth * config_.dataBufferSize;
  const std::size_t signalPerPeer = 2 * sr.maxGroups * sizeof(uint64_t);
  const std::size_t counterPerPeer = sr.maxGroups * sizeof(uint64_t);
  const std::size_t stepStatePerPeer = 2 * sr.maxGroups * sizeof(int64_t);

  auto allocateBulk = [&](std::size_t perPeer) {
    auto buf = std::make_unique<meta::comms::DeviceBuffer>(perPeer * numPeers);
    auto err = cudaMemset(buf->get(), 0, perPeer * numPeers);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          fmt::format(
              "Failed to zero send/recv buffer: {}", cudaGetErrorString(err)));
    }
    return buf;
  };

  sendStagingBulk_ = allocateBulk(stagingPerPeer);
  recvStagingBulk_ = allocateBulk(stagingPerPeer);
  signalBulk_ = allocateBulk(signalPerPeer);
  counterBulk_ = allocateBulk(counterPerPeer);
  stepStateBulk_ = allocateBulk(stepStatePerPeer);

  auto sendStagingBulkReg =
      registerBuffer(sendStagingBulk_->get(), stagingPerPeer * numPeers);
  recvStagingBulkReg_ =
      registerBuffer(recvStagingBulk_->get(), stagingPerPeer * numPeers);
  signalBulkReg_ = registerBuffer(signalBulk_->get(), signalPerPeer * numPeers);
  auto counterBulkReg =
      registerBuffer(counterBulk_->get(), counterPerPeer * numPeers);

  sendRecvPeerBuffers_.resize(numPeers);
  for (int i = 0; i < numPeers; ++i) {
    auto& pb = sendRecvPeerBuffers_[i];
    pb.sendStaging = IbgdaLocalBuffer{
        static_cast<char*>(sendStagingBulk_->get()) + i * stagingPerPeer,
        sendStagingBulkReg.lkey_per_device};
    pb.recvStaging = IbgdaLocalBuffer{
        static_cast<char*>(recvStagingBulk_->get()) + i * stagingPerPeer,
        recvStagingBulkReg_.lkey_per_device};
    pb.signal = IbgdaLocalBuffer{
        static_cast<char*>(signalBulk_->get()) + i * signalPerPeer,
        signalBulkReg_.lkey_per_device};
    pb.counter = IbgdaLocalBuffer{
        static_cast<char*>(counterBulk_->get()) + i * counterPerPeer,
        counterBulkReg.lkey_per_device};
    pb.stepState = reinterpret_cast<int64_t*>(
        static_cast<char*>(stepStateBulk_->get()) + i * stepStatePerPeer);
  }

  VLOG(1) << "MultipeerIbgdaTransport: allocated tile buffers for " << numPeers
          << " peers (staging=" << stagingPerPeer << "B per peer)";
}

void MultipeerIbgdaTransport::exchange_send_recv_buffers() {
  if (!config_.sendRecv.has_value() || sendRecvPeerBuffers_.empty()) {
    return;
  }

  const int numPeers = nRanks_ - 1;
  const std::size_t stagingPerPeer =
      config_.sendRecv->pipelineDepth * config_.dataBufferSize;

  const std::size_t signalPerPeer =
      2 * config_.sendRecv->maxGroups * sizeof(uint64_t);

  auto recvStagingRemotes = exchangeBuffer(recvStagingBulkReg_);
  auto signalRemotes = exchangeBuffer(signalBulkReg_);

  for (int i = 0; i < numPeers; ++i) {
    int peerRank = peerIndexToRank(i);
    int remotePeerIndex = (myRank_ < peerRank) ? myRank_ : (myRank_ - 1);

    sendRecvPeerBuffers_[i].remoteRecvStaging =
        recvStagingRemotes[i].subBuffer(remotePeerIndex * stagingPerPeer);
    sendRecvPeerBuffers_[i].remoteSignal =
        signalRemotes[i].subBuffer(remotePeerIndex * signalPerPeer);
  }

  VLOG(1) << "MultipeerIbgdaTransport: exchanged tile buffers with " << numPeers
          << " peers";
}

void MultipeerIbgdaTransport::cleanup_send_recv_buffers() {
  sendRecvPeerBuffers_.clear();

  if (sendStagingBulk_) {
    deregisterBuffer(sendStagingBulk_->get());
  }
  if (recvStagingBulk_) {
    deregisterBuffer(recvStagingBulk_->get());
  }
  if (signalBulk_) {
    deregisterBuffer(signalBulk_->get());
  }
  if (counterBulk_) {
    deregisterBuffer(counterBulk_->get());
  }

  sendStagingBulk_.reset();
  recvStagingBulk_.reset();
  signalBulk_.reset();
  counterBulk_.reset();
  stepStateBulk_.reset();
}

} // namespace comms::pipes
