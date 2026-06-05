// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultipeerIbgdaTransport.h"

#ifdef __HIP_PLATFORM_AMD__
// On AMD: use the HIP runtime for the cuda* API calls below (HIPify
// renames cuda* -> hip* in source before compilation), and bring in
// `meta::comms::DeviceBuffer` from the pipes-local HIP shim.
#include <hip/hip_runtime.h>

#include "comms/pipes/amd/HipHostCompat.h"
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

// NVIDIA-only host-side helpers. On AMD their functionality is provided
// by `comms/pipes/amd/DocaCompat.h` (already included via
// `MultipeerIbgdaTransport.h`) which translates `doca_*` to the
// `pipes_gda_*` host APIs in `amd/pipes_gda/PipesGdaHost.{h,cc}`.
#ifndef __HIP_PLATFORM_AMD__
#include "comms/pipes/CudaDriverLazy.h"
#include "comms/pipes/DocaHostUtils.h"
#endif
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

// Bootstrap tags for the two-phase bilateral exchange in materializePeer.
constexpr int kTagQpExchange = 0;
constexpr int kTagBufferExchange = 1;
constexpr const char* kMaterializationFailedError =
    "MultipeerIbgdaTransport: lazy peer materialization previously failed; "
    "retry is not supported";

} // namespace

// Wire formats for bilateral exchange in materializePeer.
// Split into two phases: QP info first (to connect), then buffer info
// (acts as QP-ready barrier — mirrors the eager path's two-allGather pattern).
struct PeerQpPayload {
  struct NicQpInfo {
    uint8_t gid[16]{};
    uint16_t lid{0};
    uint32_t qpns[kMaxQpsPerPeerPerNic]{};
  };
  NicQpInfo nicInfo[kMaxNicsPerGpu]{};
  int gidIndex{0};
  int mtu{0};
  int numNics{0};
  int numQpsPerPeerPerNic{0};
};

struct PeerBufferPayload {
  IbgdaBufferExchInfo recvStaging;
  IbgdaBufferExchInfo srSignal;
  IbgdaBufferExchInfo slotSignal;
  IbgdaBufferExchInfo slotDiscard;
};

struct PeerBufferSizes {
  std::size_t staging{0};
  std::size_t srSignal{0};
  std::size_t srCounter{0};
  std::size_t srStepState{0};
  std::size_t srProgressState{0};
  std::size_t slotSignal{0};
  std::size_t slotCounter{0};
  std::size_t slotDiscard{0};

  // Pointers into a contiguous allocation (set by layout())
  void* sendStagingPtr{nullptr};
  void* recvStagingPtr{nullptr};
  void* srSignalPtr{nullptr};
  void* srCounterPtr{nullptr};
  void* srStepStatePtr{nullptr};
  void* srProgressStatePtr{nullptr};
  void* slotSignalPtr{nullptr};
  void* slotCounterPtr{nullptr};
  void* slotDiscardPtr{nullptr};

  std::size_t total() const {
    return staging * 2 + srSignal + srCounter + srStepState + srProgressState +
        slotSignal + slotCounter + slotDiscard;
  }

  void layout(void* base) {
    char* p = static_cast<char*>(base);
    sendStagingPtr = p;
    p += staging;
    recvStagingPtr = p;
    p += staging;
    srSignalPtr = p;
    p += srSignal;
    srCounterPtr = p;
    p += srCounter;
    srStepStatePtr = p;
    p += srStepState;
    srProgressStatePtr = p;
    p += srProgressState;
    slotSignalPtr = p;
    p += slotSignal;
    slotCounterPtr = p;
    p += slotCounter;
    slotDiscardPtr = p;
  }
};

namespace {

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
    // On AMD, the `DataDirectMode::Only` default triggers
    // `ibv_reg_dmabuf_mr` inside `augmentWithDataDirect()`, which is not
    // exercised on AMD's libibverbs path here. Force `Disabled` to skip the
    // DataDirect probe; the prior AMD-specific transport ran with the same
    // configuration.
#ifdef __HIP_PLATFORM_AMD__
    auto discovery = GpuNicDiscovery(
        config_.cudaDevice, config_.ibHca, DataDirectMode::Disabled);
#else
    auto discovery = GpuNicDiscovery(config_.cudaDevice, config_.ibHca);
#endif
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
  sinkBufferSize_ = sizeof(uint64_t);

#ifdef __HIP_PLATFORM_AMD__
  // On AMD: use host-pinned memory for the sink. AMD doesn't have a
  // direct equivalent of CUDA's `gpuDirectRDMACapable=1` flag (the AMD
  // path uses HSA + DMA-buf for GPU memory RDMA registration). Host-
  // pinned memory works fine for the discarded atomic-FA result and
  // matches what `comms/pipes/amd/MultipeerIbgdaTransportAmd.cu` does
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
  const int numQps = config_.numQpsPerPeerPerNic;
  const int totalQpsPerPeer = numNics_ * numQps;
  const int totalQpGroups = numPeers * totalQpsPerPeer;
  for (auto& nic : nicDevices_) {
    nic.qpGroups.resize(static_cast<size_t>(numPeers) * numQps);
    nic.loopbackCompanionQps.resize(static_cast<size_t>(numPeers) * numQps);
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

void MultipeerIbgdaTransport::createPeerQps(int peerIndex) {
  const int numQps = config_.numQpsPerPeerPerNic;
  for (int nic = 0; nic < numNics_; nic++) {
    doca_gpu_verbs_qp_init_attr_hl mainAttr{};
    mainAttr.gpu_dev = docaGpu_;
    mainAttr.ibpd = nicDevices_[nic].ibvPd;
    mainAttr.sq_nwqe = config_.qpDepth;
    mainAttr.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
    mainAttr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;

    doca_gpu_verbs_qp_init_attr_hl loopbackAttr = mainAttr;
    loopbackAttr.sq_nwqe = kCompanionQpDepth;

    auto& nicQps = nicDevices_[nic].qpGroups;
    auto& nicLoopback = nicDevices_[nic].loopbackCompanionQps;
    for (int q = 0; q < numQps; q++) {
      const int qpIdx = peerIndex * numQps + q;
      doca_error_t err =
          doca_gpu_verbs_create_qp_group_hl(&mainAttr, &nicQps[qpIdx]);
      checkDocaError(err, "Failed to create QP group");
      err = doca_gpu_verbs_create_qp_hl(&loopbackAttr, &nicLoopback[qpIdx]);
      checkDocaError(err, "Failed to create loopback companion QP");
    }
  }
}

void MultipeerIbgdaTransport::connectPeerLoopback(int peerIndex) {
  const int numQps = config_.numQpsPerPeerPerNic;
  for (int nic = 0; nic < numNics_; nic++) {
    auto& nicQps = nicDevices_[nic].qpGroups;
    auto& nicLoopback = nicDevices_[nic].loopbackCompanionQps;

    IbgdaTransportExchInfo selfInfo;
    memcpy(selfInfo.gid, nicDevices_[nic].localGid.raw, sizeof(selfInfo.gid));
    selfInfo.gidIndex = gidIndex_;
    selfInfo.mtu = localMtu_;
    ibv_port_attr portAttr{};
    if (doca_verbs_wrapper_ibv_query_port(
            nicDevices_[nic].ibvCtx, 1, &portAttr) == DOCA_SUCCESS) {
      selfInfo.lid = portAttr.lid;
    }

    for (int q = 0; q < numQps; q++) {
      const int qpIdx = peerIndex * numQps + q;
      selfInfo.qpn = doca_verbs_qp_get_qpn(nicLoopback[qpIdx]->qp);
      connectQp(&nicQps[qpIdx]->qp_companion, selfInfo, nic);
      selfInfo.qpn = doca_verbs_qp_get_qpn(nicQps[qpIdx]->qp_companion.qp);
      connectQp(nicLoopback[qpIdx], selfInfo, nic);
    }
  }
}

PeerBufferSizes MultipeerIbgdaTransport::computePeerBufferSizes() const {
  PeerBufferSizes sizes;
  if (config_.sendRecv.has_value()) {
    const auto& sr = *config_.sendRecv;
    sizes.staging = sr.pipelineDepth * config_.dataBufferSize;
    sizes.srSignal = 2 * sr.maxGroups * sizeof(uint64_t);
    sizes.srCounter = sr.maxGroups * sizeof(uint64_t);
    sizes.srStepState = 2 * sr.maxGroups * sizeof(int64_t);
    sizes.srProgressState =
        2 * sr.maxGroups * sizeof(detail::IbSendRecvProgressState);
  }
  sizes.slotSignal =
      static_cast<std::size_t>(config_.numSignalSlots) * sizeof(uint64_t);
  sizes.slotCounter =
      static_cast<std::size_t>(config_.numCounterSlots) * sizeof(uint64_t);
  sizes.slotDiscard = (config_.numCounterSlots > 0) ? sizeof(uint64_t) : 0;
  return sizes;
}

P2pIbgdaTransportBuildParams MultipeerIbgdaTransport::buildPeerTransportParams(
    int peerIndex) const {
  const int numQps = config_.numQpsPerPeerPerNic;
  P2pIbgdaTransportBuildParams params;
  params.h_nicDeviceIbgdaResources.resize(numNics_);
  for (int n = 0; n < numNics_; ++n) {
    auto& nicSpec = params.h_nicDeviceIbgdaResources[n];
    nicSpec.qps.resize(numQps);
    nicSpec.companionQps.resize(numQps);
    nicSpec.sinkLkey = NetworkLKey(HostLKey(nicDevices_[n].sinkMr->lkey));
    nicSpec.deviceId = n;
  }

  for (int nic = 0; nic < numNics_; nic++) {
    auto& nicQps = nicDevices_[nic].qpGroups;
    auto& nicSpec = params.h_nicDeviceIbgdaResources[nic];
    for (int q = 0; q < numQps; q++) {
      const int qpIdx = peerIndex * numQps + q;
      doca_error_t err = doca_gpu_verbs_get_qp_dev(
          nicQps[qpIdx]->qp_main.qp_gverbs, &nicSpec.qps[q]);
      checkDocaError(err, "Failed to get GPU QP handle");

      err = doca_gpu_verbs_get_qp_dev(
          nicQps[qpIdx]->qp_companion.qp_gverbs, &nicSpec.companionQps[q]);
      checkDocaError(err, "Failed to get companion GPU QP handle");
    }
  }

  if (config_.numSignalSlots > 0) {
    params.remoteSignalBuf = signalRemoteViews_[peerIndex];
    params.localSignalBuf = signalLocalViews_[peerIndex];
    params.numSignalSlots = config_.numSignalSlots;
  }
  if (config_.numCounterSlots > 0) {
    params.counterBuf = counterViews_[peerIndex];
    params.discardSignalSlot = discardSignalRemoteViews_[peerIndex];
    params.numCounterSlots = config_.numCounterSlots;
  }
  if (!sendRecvPeerBuffers_.empty()) {
    const auto& pb = sendRecvPeerBuffers_[peerIndex];
    params.sendRecvState = IbSendRecvState{
        .sendStagingBuf = pb.sendStaging,
        .recvStagingBuf = pb.remoteRecvStaging,
        .sendStagingPtr = static_cast<char*>(pb.sendStaging.ptr),
        .recvStagingPtr = static_cast<char*>(pb.recvStaging.ptr),
        .localSignalBuf = pb.signal,
        .remoteSignalBuf = pb.remoteSignal,
        .localCounterBuf = pb.counter,
        .stepState = pb.stepState,
        .progressState = pb.progressState,
        .maxGroups = config_.sendRecv->maxGroups,
        .pipelineDepth = config_.sendRecv->pipelineDepth,
        .dataBufferSize = config_.dataBufferSize,
    };
  }
  return params;
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
  if (config.numQpsPerPeerPerNic < 1 ||
      config.numQpsPerPeerPerNic > kMaxQpsPerPeerPerNic) {
    throw std::invalid_argument(
        fmt::format(
            "numQpsPerPeerPerNic must be in [1, {}], got {}",
            kMaxQpsPerPeerPerNic,
            config.numQpsPerPeerPerNic));
  }
  if (config.numQpsPerPeerPerNic * (nRanks - 1) * 3 > 1000) {
    LOG(WARNING) << "MultipeerIbgdaTransport: high QP count: "
                 << config.numQpsPerPeerPerNic << " QPs/(peer,NIC) * "
                 << (nRanks - 1) << " peers * 3 = "
                 << config.numQpsPerPeerPerNic * (nRanks - 1) * 3
                 << " total QPs (per NIC)";
  }

  // Resolve numNics_ from the available NIC sources. No numeric knob —
  // the count is implied by what the caller / topology actually provides:
  //   1. config.gpuNicMap[cudaDevice] populated → use its NIC list.
  //   2. Otherwise auto-discover via GpuNicDiscovery — every NIC at the
  //      best-affinity tier (same pathType + bandwidth + isDataDirect as
  //      the top candidate).
  // No silent fallback to 1: if a GPU is wired to N best-affinity NICs,
  // the transport must use all N. H100 (1 NIC) and GB200/GB300 (2 NICs)
  // both get the right count automatically; an unexpected count throws
  // with a clear hint.
  {
    auto it = config.gpuNicMap.find(config.cudaDevice);
    int n = 0;
    const char* source = nullptr;
    if (it != config.gpuNicMap.end() && !it->second.empty()) {
      n = static_cast<int>(it->second.size());
      source = "config.gpuNicMap";
    } else {
      // See comment on the auto-discovery branch in openIbDevice for why we
      // force DataDirectMode::Disabled on AMD (skip the DataDirect probe).
#ifdef __HIP_PLATFORM_AMD__
      GpuNicDiscovery discovery(
          config.cudaDevice, config.ibHca, DataDirectMode::Disabled);
#else
      GpuNicDiscovery discovery(config.cudaDevice, config.ibHca);
#endif
      auto bestNics = discovery.getBestAffinityNics();
      if (bestNics.empty()) {
        throw std::runtime_error(
            fmt::format(
                "MultipeerIbgdaTransport: NIC auto-discovery returned no "
                "candidates for GPU {}; set config.gpuNicMap or config.ibHca "
                "to expose at least one NIC",
                config.cudaDevice));
      }
      n = static_cast<int>(bestNics.size());
      source = "auto-discovery (best-affinity tier)";
    }
    if (n > kMaxNicsPerGpu) {
      throw std::runtime_error(
          fmt::format(
              "MultipeerIbgdaTransport: {} found {} NIC(s) for GPU {} but "
              "kMaxNicsPerGpu={}; raise kMaxNicsPerGpu or trim the source",
              source,
              n,
              config.cudaDevice,
              kMaxNicsPerGpu));
    }
    numNics_ = n;
    VLOG(1) << "MultipeerIbgdaTransport: numNics_=" << numNics_
            << " (source=" << source << ")";
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
      for (auto& nic : nicDevices_) {
        nic.qpGroups.resize(
            static_cast<size_t>(numPeers) * config.numQpsPerPeerPerNic);
        nic.loopbackCompanionQps.resize(
            static_cast<size_t>(numPeers) * config.numQpsPerPeerPerNic);
      }
      signalRemoteViews_.resize(numPeers);
      signalLocalViews_.resize(numPeers);
      counterViews_.resize(numPeers);
      discardSignalRemoteViews_.resize(numPeers);
      lazyPeerBufs_.resize(numPeers);
      peerMaterialized_.resize(numPeers, false);
    }

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
  // MRs must be deregistered BEFORE freeing (correct RDMA teardown order).
  // On AMD: signal/discard inboxes are host-pinned (NIC-accessible);
  // counter is GPU device memory (sender-local, no NIC access).
  if (signalInboxGpu_ != nullptr) {
    deregisterBuffer(signalInboxGpu_);
#ifdef __HIP_PLATFORM_AMD__
    hipHostFree(signalInboxGpu_);
#else
    cudaFree(signalInboxGpu_);
#endif
    signalInboxGpu_ = nullptr;
  }
  signalRemoteViews_.clear();
  signalLocalViews_.clear();

  if (counterGpu_ != nullptr) {
    deregisterBuffer(counterGpu_);
#ifdef __HIP_PLATFORM_AMD__
    hipFree(counterGpu_);
#else
    cudaFree(counterGpu_);
#endif
    counterGpu_ = nullptr;
  }
  counterViews_.clear();

  if (discardSignalGpu_ != nullptr) {
    deregisterBuffer(discardSignalGpu_);
#ifdef __HIP_PLATFORM_AMD__
    hipHostFree(discardSignalGpu_);
#else
    cudaFree(discardSignalGpu_);
#endif
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
  const int numQps = config_.numQpsPerPeerPerNic;

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
  myInfo.numQpsPerPeerPerNic = numQps;
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

    CHECK_EQ(peerInfo.numQpsPerPeerPerNic, numQps)
        << "Rank " << peerRank
        << " has numQpsPerPeerPerNic=" << peerInfo.numQpsPerPeerPerNic
        << " but local rank " << myRank_ << " has " << numQps
        << ". All ranks must use the same numQpsPerPeerPerNic.";

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
            << " numQps=" << peerInfo.numQpsPerPeerPerNic
            << " slot0_qpn=" << peerExchInfo_[peerIndex].qpn;
  }

  // Connect main QPs + loopback companions for all peers
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
    connectPeerLoopback(peerIndex);
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

#ifdef __HIP_PLATFORM_AMD__
    // Host-pinned: on AMD, GPU-memory MR registration relies on amdgpu's
    // peer-mem integration which is unreliable on test hosts. Host-pinned
    // memory works with `ibv_reg_mr` (no peer_mem needed) and is GPU-
    // accessible via mapped memory — same approach as the deleted
    // `MultipeerIbgdaTransportAmd::sinkBuffer_`.
    hipError_t hipErr =
        hipHostMalloc(&signalInboxGpu_, totalSignalBytes, hipHostMallocDefault);
    if (hipErr != hipSuccess) {
      throw std::runtime_error(
          "Failed to allocate signal inbox (host-pinned): " +
          std::string(hipGetErrorString(hipErr)));
    }
    std::memset(signalInboxGpu_, 0, totalSignalBytes);
#else
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
#endif

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

#ifdef __HIP_PLATFORM_AMD__
    // Counter is GPU-only: GPU writes via atomic_fetch_add and reads in
    // wait_counter spin loop, no NIC-side access. Device memory (HBM,
    // ~100ns/access) outperforms host-pinned (PCIe, ~1us/access) and
    // dma-buf isn't needed.
    hipError_t hipErr = hipMalloc(&counterGpu_, totalCounterBytes);
    if (hipErr != hipSuccess) {
      throw std::runtime_error(
          "Failed to allocate counter buffer: " +
          std::string(hipGetErrorString(hipErr)));
    }
    hipErr = hipMemset(counterGpu_, 0, totalCounterBytes);
    if (hipErr != hipSuccess) {
      throw std::runtime_error("Failed to zero counter buffer");
    }
#else
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
#endif

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

#ifdef __HIP_PLATFORM_AMD__
    // See signal-inbox AMD branch above — same host-pinned rationale.
    hipError_t hipErr = hipHostMalloc(
        &discardSignalGpu_, totalDiscardBytes, hipHostMallocDefault);
    if (hipErr != hipSuccess) {
      throw std::runtime_error(
          "Failed to allocate discard-signal buffer (host-pinned): " +
          std::string(hipGetErrorString(hipErr)));
    }
    std::memset(discardSignalGpu_, 0, totalDiscardBytes);
#else
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
#endif

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

  // Build device transports on GPU
  std::vector<P2pIbgdaTransportBuildParams> buildParams(numPeers);
  for (int peer = 0; peer < numPeers; peer++) {
    buildParams[peer] = buildPeerTransportParams(peer);
  }

  peerTransportsGpu_ =
      buildDeviceTransportsOnGpu(buildParams, numPeers, gpuAllocations_);
  peerTransportSize_ = getP2pIbgdaTransportDeviceSize();

  VLOG(1) << "MultipeerIbgdaTransport: rank " << myRank_
          << " exchange complete, connected to " << numPeers << " peers"
          << " (" << numQps << " QPs/(peer,NIC) × " << numNics_ << " NICs)";
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

int MultipeerIbgdaTransport::numPeers() const {
  return nRanks_ - 1;
}

int MultipeerIbgdaTransport::myRank() const {
  return myRank_;
}

int MultipeerIbgdaTransport::getGidIndex() const {
  return gidIndex_;
}

int MultipeerIbgdaTransport::numQpsPerPeerPerNic() const {
  return config_.numQpsPerPeerPerNic;
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

  // Cache miss — find the GPU allocation base and register it on every
  // NIC's PD. Each NIC gets an independent MR over the same physical
  // memory; lkey/rkey differ per NIC.
#ifdef __HIP_PLATFORM_AMD__
  // On AMD: HIP doesn't expose an exact equivalent of
  // `cuMemGetAddressRange` (which returns the original allocation
  // base/size for a pointer in the middle of an allocation). For the
  // common case where the caller passes the base pointer of an
  // allocation, register exactly the requested range.
  uintptr_t allocBase = reinterpret_cast<uintptr_t>(ptr);
  size_t allocSize = size;
#else
  CUdeviceptr allocBase = 0;
  size_t allocSize = 0;
  CUresult cuRes =
      pfn_cuMemGetAddressRange(&allocBase, &allocSize, (CUdeviceptr)ptr);
  if (cuRes != CUDA_SUCCESS || allocBase == 0) {
    throw std::runtime_error(
        "registerBuffer: cuMemGetAddressRange failed for ptr");
  }
#endif
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
      errno = 0;
      doca_error_t regErr = doca_verbs_wrapper_ibv_reg_mr(
          nicDevices_[n].ibvPd,
          reinterpret_cast<void*>(allocBase),
          allocSize,
          accessFlags,
          &mr);
      if (regErr != DOCA_SUCCESS || !mr) {
        const int savedErrno = errno;
        // Roll back partial registration before throwing.
        for (int j = 0; j < n; ++j) {
          doca_verbs_wrapper_ibv_dereg_mr(cached.mrs[j]);
        }
        throw std::runtime_error(
            fmt::format(
                "Failed to register buffer with RDMA on NIC {} "
                "(allocBase=0x{:x} allocSize={} regErr={} errno={} ({}))",
                n,
                allocBase,
                allocSize,
                static_cast<int>(regErr),
                savedErrno,
                std::strerror(savedErrno)));
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

  // Find the MR for this buffer via its GPU allocation base.
#ifdef __HIP_PLATFORM_AMD__
  // Use `hipMemGetAddressRange` (direct analog of NVIDIA's
  // `cuMemGetAddressRange`) so a sub-buffer (`localBuf.ptr` pointing into
  // the middle of an allocation) resolves to the same key the MR was
  // registered under. NOTE: an earlier attempt used
  // `hipPointerGetAttributes().devicePointer`, but that returns the queried
  // pointer (or its host-mapped equivalent) — NOT the allocation base —
  // so it didn't actually fix the sub-buffer bug.
  hipDeviceptr_t basePtr = 0;
  size_t allocRange = 0;
  if (hipMemGetAddressRange(&basePtr, &allocRange, localBuf.ptr) !=
          hipSuccess ||
      basePtr == 0) {
    throw std::runtime_error(
        "exchangeBuffer: hipMemGetAddressRange failed for ptr");
  }
  uintptr_t allocBase = reinterpret_cast<uintptr_t>(basePtr);
#else
  CUdeviceptr allocBase = 0;
  size_t allocSize = 0;
  CUresult cuRes = pfn_cuMemGetAddressRange(
      &allocBase, &allocSize, (CUdeviceptr)localBuf.ptr);
  if (cuRes != CUDA_SUCCESS || allocBase == 0) {
    throw std::runtime_error(
        "exchangeBuffer: cuMemGetAddressRange failed for ptr");
  }
#endif
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
  auto sizes = computePeerBufferSizes();
  const auto stagingPerPeer = sizes.staging;
  const auto signalPerPeer = sizes.srSignal;
  const auto counterPerPeer = sizes.srCounter;
  const auto stepStatePerPeer = sizes.srStepState;
  const auto progressStatePerPeer = sizes.srProgressState;

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

  sendRecvPeerBuffers_.resize(numPeers);

  if (!config_.ibLazyConnect) {
    sendStagingBulk_ = allocateBulk(stagingPerPeer);
    recvStagingBulk_ = allocateBulk(stagingPerPeer);
    signalBulk_ = allocateBulk(signalPerPeer);
    counterBulk_ = allocateBulk(counterPerPeer);
    stepStateBulk_ = allocateBulk(stepStatePerPeer);
    progressStateBulk_ = allocateBulk(progressStatePerPeer);

    auto sendStagingBulkReg =
        registerBuffer(sendStagingBulk_->get(), stagingPerPeer * numPeers);
    recvStagingBulkReg_ =
        registerBuffer(recvStagingBulk_->get(), stagingPerPeer * numPeers);
    signalBulkReg_ =
        registerBuffer(signalBulk_->get(), signalPerPeer * numPeers);
    counterBulkReg_ =
        registerBuffer(counterBulk_->get(), counterPerPeer * numPeers);

    for (int i = 0; i < numPeers; ++i) {
      auto& pb = sendRecvPeerBuffers_[i];
      pb.sendStaging = sendStagingBulkReg.subBuffer(i * stagingPerPeer);
      pb.recvStaging = recvStagingBulkReg_.subBuffer(i * stagingPerPeer);
      pb.signal = signalBulkReg_.subBuffer(i * signalPerPeer);
      pb.counter = counterBulkReg_.subBuffer(i * counterPerPeer);
      pb.stepState = reinterpret_cast<int64_t*>(
          static_cast<char*>(stepStateBulk_->get()) + i * stepStatePerPeer);
      pb.progressState = reinterpret_cast<detail::IbSendRecvProgressState*>(
          static_cast<char*>(progressStateBulk_->get()) +
          i * progressStatePerPeer);
    }

    VLOG(1) << "MultipeerIbgdaTransport: eager mode — allocated tile buffers "
            << "for " << numPeers << " peers (staging=" << stagingPerPeer
            << "B per peer, 6 bulks)";
  } else {
    VLOG(1) << "MultipeerIbgdaTransport: lazy mode — per-peer allocation "
            << "deferred to materializePeer";
  }
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
  for (auto& buf : lazyPeerBufs_) {
    if (buf) {
      deregisterBuffer(buf->get());
      buf.reset();
    }
  }
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
  progressStateBulk_.reset();
}

bool MultipeerIbgdaTransport::isPeerMaterialized(int peerRank) const {
  if (peerRank == myRank_ || peerRank < 0 || peerRank >= nRanks_) {
    throw std::invalid_argument(
        fmt::format(
            "isPeerMaterialized: invalid peerRank={} (myRank={}, nRanks={})",
            peerRank,
            myRank_,
            nRanks_));
  }
  if (!config_.ibLazyConnect) {
    return true;
  }
  int peerIndex = rankToPeerIndex(peerRank);
  return peerMaterialized_[peerIndex];
}

PeerQpPayload MultipeerIbgdaTransport::buildLocalQpPayload(
    int peerIndex) const {
  const int numQps = config_.numQpsPerPeerPerNic;
  PeerQpPayload payload{};
  payload.gidIndex = gidIndex_;
  payload.mtu = static_cast<int>(localMtu_);
  payload.numNics = numNics_;
  payload.numQpsPerPeerPerNic = numQps;

  for (int n = 0; n < numNics_; ++n) {
    memcpy(
        payload.nicInfo[n].gid,
        nicDevices_[n].localGid.raw,
        sizeof(payload.nicInfo[n].gid));
    ibv_port_attr portAttr{};
    if (doca_verbs_wrapper_ibv_query_port(
            nicDevices_[n].ibvCtx, 1, &portAttr) == DOCA_SUCCESS) {
      payload.nicInfo[n].lid = portAttr.lid;
    }
    auto& nicQps = nicDevices_[n].qpGroups;
    for (int q = 0; q < numQps; q++) {
      const int qpIdx = peerIndex * numQps + q;
      payload.nicInfo[n].qpns[q] =
          doca_verbs_qp_get_qpn(nicQps[qpIdx]->qp_main.qp);
    }
  }
  return payload;
}

void MultipeerIbgdaTransport::allocatePeerBuffers(
    int peerIndex,
    PeerBufferPayload& payload) {
  auto sizes = computePeerBufferSizes();
  const std::size_t totalPerPeer = sizes.total();
  if (totalPerPeer == 0) {
    return;
  }

  lazyPeerBufs_[peerIndex] =
      std::make_unique<meta::comms::DeviceBuffer>(totalPerPeer);
  cudaError_t cudaErr =
      cudaMemset(lazyPeerBufs_[peerIndex]->get(), 0, totalPerPeer);
  if (cudaErr != cudaSuccess) {
    throw std::runtime_error("Failed to zero per-peer buffer");
  }

  auto& peerBuf = lazyPeerBufs_[peerIndex];
  sizes.layout(peerBuf->get());

  auto reg = registerBuffer(peerBuf->get(), totalPerPeer);

  auto addr = reinterpret_cast<uintptr_t>(peerBuf->get());
  auto mrIt = registeredBuffers_.upper_bound(addr);
  CHECK(mrIt != registeredBuffers_.begin())
      << "materializePeer: peerBuf MR not found after registerBuffer";
  --mrIt;
  auto& mr = mrIt->second;

  if (config_.sendRecv.has_value()) {
    auto& pb = sendRecvPeerBuffers_[peerIndex];
    pb.sendStaging =
        IbgdaLocalBuffer(sizes.sendStagingPtr, reg.lkey_per_device);
    pb.recvStaging =
        IbgdaLocalBuffer(sizes.recvStagingPtr, reg.lkey_per_device);
    pb.signal = IbgdaLocalBuffer(sizes.srSignalPtr, reg.lkey_per_device);
    pb.counter = IbgdaLocalBuffer(sizes.srCounterPtr, reg.lkey_per_device);
    pb.stepState = reinterpret_cast<int64_t*>(sizes.srStepStatePtr);
    pb.progressState = reinterpret_cast<detail::IbSendRecvProgressState*>(
        sizes.srProgressStatePtr);

    payload.recvStaging.addr = reinterpret_cast<uint64_t>(sizes.recvStagingPtr);
    payload.recvStaging.numNics = numNics_;
    payload.srSignal.addr = reinterpret_cast<uint64_t>(sizes.srSignalPtr);
    payload.srSignal.numNics = numNics_;
    for (int n = 0; n < numNics_; ++n) {
      auto rkey = HostRKey(mr.mrs[n]->rkey);
      payload.recvStaging.rkey_per_device[n] = rkey;
      payload.srSignal.rkey_per_device[n] = rkey;
    }
  }

  if (config_.numSignalSlots > 0) {
    signalLocalViews_[peerIndex] =
        IbgdaLocalBuffer(sizes.slotSignalPtr, reg.lkey_per_device);
    payload.slotSignal.addr = reinterpret_cast<uint64_t>(sizes.slotSignalPtr);
    payload.slotSignal.numNics = numNics_;
    for (int n = 0; n < numNics_; ++n) {
      payload.slotSignal.rkey_per_device[n] = HostRKey(mr.mrs[n]->rkey);
    }
  }
  if (config_.numCounterSlots > 0) {
    counterViews_[peerIndex] =
        IbgdaLocalBuffer(sizes.slotCounterPtr, reg.lkey_per_device);
    payload.slotDiscard.addr = reinterpret_cast<uint64_t>(sizes.slotDiscardPtr);
    payload.slotDiscard.numNics = numNics_;
    for (int n = 0; n < numNics_; ++n) {
      payload.slotDiscard.rkey_per_device[n] = HostRKey(mr.mrs[n]->rkey);
    }
  }
}

template <typename T>
T MultipeerIbgdaTransport::exchangeWithPeer(
    int peerRank,
    const T& localPayload,
    int tag) {
  T remotePayload{};

  auto timeoutUs = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::milliseconds(config_.materializePeerTimeoutMs));
  auto waitFuture = [&](auto&& future, const char* op) -> int {
    try {
      return std::move(future).get(timeoutUs);
    } catch (const std::exception&) {
      throw std::runtime_error(
          fmt::format(
              "materializePeer: rank {} {} with peer {} timed out ({}ms)",
              myRank_,
              op,
              peerRank,
              config_.materializePeerTimeoutMs));
    }
  };

  // Lower rank recvs first to avoid deadlock with blocking bootstrap
  // implementations (e.g. MpiBootstrap uses blocking MPI_Send/MPI_Recv).
  if (myRank_ < peerRank) {
    auto recvFuture =
        bootstrap_->recv(&remotePayload, sizeof(T), peerRank, /*tag=*/tag);
    int recvResult = waitFuture(std::move(recvFuture), "recv");
    if (recvResult != 0) {
      throw std::runtime_error(
          fmt::format(
              "materializePeer: rank {} recv from peer {} failed (error {})",
              myRank_,
              peerRank,
              recvResult));
    }
    auto sendFuture = bootstrap_->send(
        const_cast<T*>(&localPayload), sizeof(T), peerRank, /*tag=*/tag);
    int sendResult = waitFuture(std::move(sendFuture), "send");
    if (sendResult != 0) {
      throw std::runtime_error(
          fmt::format(
              "materializePeer: rank {} send to peer {} failed (error {})",
              myRank_,
              peerRank,
              sendResult));
    }
  } else {
    auto sendFuture = bootstrap_->send(
        const_cast<T*>(&localPayload), sizeof(T), peerRank, /*tag=*/tag);
    int sendResult = waitFuture(std::move(sendFuture), "send");
    if (sendResult != 0) {
      throw std::runtime_error(
          fmt::format(
              "materializePeer: rank {} send to peer {} failed (error {})",
              myRank_,
              peerRank,
              sendResult));
    }
    auto recvFuture =
        bootstrap_->recv(&remotePayload, sizeof(T), peerRank, /*tag=*/tag);
    int recvResult = waitFuture(std::move(recvFuture), "recv");
    if (recvResult != 0) {
      throw std::runtime_error(
          fmt::format(
              "materializePeer: rank {} recv from peer {} failed (error {})",
              myRank_,
              peerRank,
              recvResult));
    }
  }

  return remotePayload;
}

void MultipeerIbgdaTransport::connectPeerMainQps(
    int peerIndex,
    const PeerQpPayload& remotePayload) {
  const int numQps = config_.numQpsPerPeerPerNic;
  for (int nic = 0; nic < numNics_; nic++) {
    auto& nicQps = nicDevices_[nic].qpGroups;
    for (int q = 0; q < numQps; q++) {
      const int qpIdx = peerIndex * numQps + q;
      IbgdaTransportExchInfo peerInfo;
      peerInfo.qpn = remotePayload.nicInfo[nic].qpns[q];
      memcpy(
          peerInfo.gid, remotePayload.nicInfo[nic].gid, sizeof(peerInfo.gid));
      peerInfo.gidIndex = remotePayload.gidIndex;
      peerInfo.lid = remotePayload.nicInfo[nic].lid;
      peerInfo.mtu = static_cast<ibv_mtu>(remotePayload.mtu);
      connectQp(&nicQps[qpIdx]->qp_main, peerInfo, nic);
    }
  }
}

void MultipeerIbgdaTransport::applyRemoteViews(
    int peerIndex,
    const PeerBufferPayload& remotePayload) {
  if (config_.sendRecv.has_value()) {
    auto& pb = sendRecvPeerBuffers_[peerIndex];
    pb.remoteRecvStaging = remotePayload.recvStaging.toRemoteBuffer();
    pb.remoteSignal = remotePayload.srSignal.toRemoteBuffer();
  }
  if (config_.numSignalSlots > 0) {
    signalRemoteViews_[peerIndex] = remotePayload.slotSignal.toRemoteBuffer();
  }
  if (config_.numCounterSlots > 0) {
    discardSignalRemoteViews_[peerIndex] =
        remotePayload.slotDiscard.toRemoteBuffer();
  }
}

void MultipeerIbgdaTransport::cleanupPeerOnFailure(int peerIndex) {
  const int numQps = config_.numQpsPerPeerPerNic;
  for (int nic = 0; nic < numNics_; nic++) {
    auto& nicQps = nicDevices_[nic].qpGroups;
    auto& nicLoopback = nicDevices_[nic].loopbackCompanionQps;
    for (int q = 0; q < numQps; q++) {
      const int qpIdx = peerIndex * numQps + q;
      if (nicQps[qpIdx] != nullptr) {
        doca_gpu_verbs_destroy_qp_group_hl(nicQps[qpIdx]);
        nicQps[qpIdx] = nullptr;
      }
      if (nicLoopback[qpIdx] != nullptr) {
        doca_gpu_verbs_destroy_qp_hl(nicLoopback[qpIdx]);
        nicLoopback[qpIdx] = nullptr;
      }
    }
  }
  auto& buf = lazyPeerBufs_[peerIndex];
  if (buf) {
    deregisterBuffer(buf->get());
    buf.reset();
  }
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

void MultipeerIbgdaTransport::materializePeer(int peerRank) {
  queuePeerForMaterialization(peerRank);
  connectPeers();
}

void MultipeerIbgdaTransport::queuePeerForMaterialization(int peerRank) {
  if (!config_.ibLazyConnect) {
    return;
  }
  if (materializationFailed_) {
    throw std::runtime_error(kMaterializationFailedError);
  }
  if (peerRank == myRank_ || peerRank < 0 || peerRank >= nRanks_) {
    throw std::invalid_argument(
        fmt::format(
            "queuePeerForMaterialization: invalid peerRank={} (myRank={}, nRanks={})",
            peerRank,
            myRank_,
            nRanks_));
  }
  if (isPeerMaterialized(peerRank)) {
    return;
  }
  for (int p : pendingPeers_) {
    if (p == peerRank) {
      return;
    }
  }
  pendingPeers_.push_back(peerRank);
}

void MultipeerIbgdaTransport::doMaterializePeer(int peerRank) {
  int peerIndex = rankToPeerIndex(peerRank);

  createPeerQps(peerIndex);

  // Phase 1: exchange QP info, connect QPs
  auto localQp = buildLocalQpPayload(peerIndex);
  auto remoteQp = exchangeWithPeer(peerRank, localQp, kTagQpExchange);

  if (remoteQp.numNics != numNics_) {
    throw std::runtime_error(
        fmt::format(
            "materializePeer: peer {} numNics={} vs local {}",
            peerRank,
            remoteQp.numNics,
            numNics_));
  }
  if (remoteQp.numQpsPerPeerPerNic != config_.numQpsPerPeerPerNic) {
    throw std::runtime_error(
        fmt::format(
            "materializePeer: peer {} numQps={} vs local {}",
            peerRank,
            remoteQp.numQpsPerPeerPerNic,
            config_.numQpsPerPeerPerNic));
  }

  connectPeerMainQps(peerIndex, remoteQp);
  connectPeerLoopback(peerIndex);

  // Phase 2: exchange buffer info (acts as QP-ready barrier)
  PeerBufferPayload localBuf{};
  allocatePeerBuffers(peerIndex, localBuf);
  auto remoteBuf = exchangeWithPeer(peerRank, localBuf, kTagBufferExchange);
  applyRemoteViews(peerIndex, remoteBuf);

  auto params = buildPeerTransportParams(peerIndex);
  writeDeviceTransportSlot(
      peerTransportsGpu_, peerIndex, params, gpuAllocations_);
  peerMaterialized_[peerIndex] = true;

  VLOG(1) << "MultipeerIbgdaTransport: rank " << myRank_
          << " materialized peer " << peerRank;
}

void MultipeerIbgdaTransport::connectPeers() {
  if (materializationFailed_) {
    pendingPeers_.clear();
    throw std::runtime_error(kMaterializationFailedError);
  }
  if (pendingPeers_.empty()) {
    return;
  }
  std::sort(pendingPeers_.begin(), pendingPeers_.end());

  std::vector<int> peers;
  peers.swap(pendingPeers_);
  std::vector<int> touchedPeerIndexes;
  touchedPeerIndexes.reserve(peers.size());

  try {
    for (int peerRank : peers) {
      if (isPeerMaterialized(peerRank)) {
        continue;
      }
      const int peerIndex = rankToPeerIndex(peerRank);
      touchedPeerIndexes.push_back(peerIndex);
      doMaterializePeer(peerRank);
    }
  } catch (...) {
    materializationFailed_ = true;
    for (int peerIndex : touchedPeerIndexes) {
      cleanupPeerOnFailure(peerIndex);
    }
    throw;
  }
}

} // namespace comms::pipes
