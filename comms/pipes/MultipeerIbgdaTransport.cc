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

#include "comms/pipes/MultipeerIbgdaDeviceTransport.cuh"
#include "comms/pipes/MultipeerIbgdaTransportCuda.cuh"
#include "comms/pipes/rdma/NicDiscovery.h"

namespace comms::pipes {

namespace {

constexpr int kDefaultGidIndex = 3; // Default GID index
constexpr int kHopLimit = 255;

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

  gpuPciBusId_ = NicDiscovery::getCudaPciBusId(config_.cudaDevice);

  LOG(INFO) << "MultipeerIbgdaTransport: GPU " << config_.cudaDevice << " PCIe "
            << gpuPciBusId_;

  doca_error_t err = doca_gpu_create(gpuPciBusId_.c_str(), &docaGpu_);
  checkDocaError(err, "Failed to create DOCA GPU context");

  LOG(INFO) << "MultipeerIbgdaTransport: DOCA GPU context created: "
            << (void*)docaGpu_;

  gidIndex_ = config_.gidIndex.value_or(kDefaultGidIndex);
}

void MultipeerIbgdaTransport::openIbDevice() {
  // Get all IB devices
  int numDevices = 0;
  ibv_device** deviceList = ibv_get_device_list(&numDevices);
  if (!deviceList || numDevices == 0) {
    throw std::runtime_error("No IB devices found");
  }

  // Priority 1: Explicit GPU-to-NIC mapping from config
  auto it = config_.gpuNicMap.find(config_.cudaDevice);
  if (it != config_.gpuNicMap.end() && !it->second.empty()) {
    nicDeviceName_ = it->second[0]; // Use first (preferred) NIC
    LOG(INFO) << "MultipeerIbgdaTransport: using config.gpuNicMap for GPU "
              << config_.cudaDevice << " -> " << nicDeviceName_;
  }

  // Priority 2: Auto-discovery if no config override
  if (nicDeviceName_.empty()) {
    NicDiscovery discovery(config_.cudaDevice, config_.ibHca);
    const auto& candidates = discovery.getCandidates();
    nicDeviceName_ = candidates[0].name;
    LOG(INFO) << "MultipeerIbgdaTransport: using NIC " << nicDeviceName_
              << " for GPU device " << config_.cudaDevice;
  }

  // Find the NIC by name
  int nicIdx = -1;
  for (int i = 0; i < numDevices; i++) {
    if (nicDeviceName_ == deviceList[i]->name) {
      nicIdx = i;
      break;
    }
  }
  if (nicIdx < 0) {
    ibv_free_device_list(deviceList);
    throw std::runtime_error("Specified NIC not found: " + nicDeviceName_);
  }
  LOG(INFO) << "MultipeerIbgdaTransport: found NIC " << nicDeviceName_
            << " at index " << nicIdx;

  LOG(INFO) << "MultipeerIbgdaTransport: selected NIC " << nicDeviceName_
            << " for GPU " << gpuPciBusId_;

  // Open the device
  ibvCtx_ = ibv_open_device(deviceList[nicIdx]);
  ibv_free_device_list(deviceList);
  if (!ibvCtx_) {
    throw std::runtime_error("Failed to open IB device: " + nicDeviceName_);
  }

  // Allocate PD
  ibvPd_ = ibv_alloc_pd(ibvCtx_);
  if (!ibvPd_) {
    throw std::runtime_error("Failed to allocate protection domain");
  }

  // Query GID
  if (ibv_query_gid(ibvCtx_, 1, gidIndex_, &localGid_) != 0) {
    throw std::runtime_error(
        "Failed to query GID at index " + std::to_string(gidIndex_));
  }

  // Print GID value for debugging
  auto gidStr = fmt::format(
      "{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:"
      "{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}",
      localGid_.raw[0],
      localGid_.raw[1],
      localGid_.raw[2],
      localGid_.raw[3],
      localGid_.raw[4],
      localGid_.raw[5],
      localGid_.raw[6],
      localGid_.raw[7],
      localGid_.raw[8],
      localGid_.raw[9],
      localGid_.raw[10],
      localGid_.raw[11],
      localGid_.raw[12],
      localGid_.raw[13],
      localGid_.raw[14],
      localGid_.raw[15]);
  LOG(INFO) << "MultipeerIbgdaTransport: GID index " << gidIndex_ << " = "
            << gidStr;

  // Query port to determine link layer (IB vs Ethernet)
  ibv_port_attr portAttr{};
  if (ibv_query_port(ibvCtx_, 1, &portAttr) != 0) {
    throw std::runtime_error("Failed to query port attributes");
  }

  LOG(INFO) << "MultipeerIbgdaTransport: port 1 state=" << portAttr.state
            << " link_layer=" << (int)portAttr.link_layer
            << " (1=IB, 2=Ethernet)"
            << " active_mtu=" << portAttr.active_mtu;

  if (portAttr.state != IBV_PORT_ACTIVE) {
    throw std::runtime_error(
        "Port 1 is not active (state=" + std::to_string(portAttr.state) + ")");
  }

  // Store local port MTU for negotiation during connectQp
  localMtu_ = portAttr.active_mtu;

  // Determine address type based on link layer
  // For InfiniBand, always use IB_NO_GRH. For RoCE (Ethernet), use the
  // configured address family (similar to NCCL_IB_ADDR_FAMILY).
  doca_verbs_addr_type addrType;
  if (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    addrType = DOCA_VERBS_ADDR_TYPE_IB_NO_GRH;
  } else {
    addrType = (config_.addressFamily == AddressFamily::IPV4)
        ? DOCA_VERBS_ADDR_TYPE_IPv4
        : DOCA_VERBS_ADDR_TYPE_IPv6;
  }

  doca_error_t err = doca_verbs_ah_attr_create(ibvCtx_, &ahAttr_);
  checkDocaError(err, "Failed to create AH attributes");

  err = doca_verbs_ah_attr_set_addr_type(ahAttr_, addrType);
  checkDocaError(err, "Failed to set address type");

  err = doca_verbs_ah_attr_set_sgid_index(ahAttr_, gidIndex_);
  checkDocaError(err, "Failed to set SGID index");

  err = doca_verbs_ah_attr_set_hop_limit(ahAttr_, kHopLimit);
  checkDocaError(err, "Failed to set hop limit");

  err = doca_verbs_ah_attr_set_traffic_class(ahAttr_, config_.trafficClass);
  checkDocaError(err, "Failed to set traffic class");

  err = doca_verbs_ah_attr_set_sl(ahAttr_, config_.serviceLevel);
  checkDocaError(err, "Failed to set service level");
}

void MultipeerIbgdaTransport::allocateResources() {
  const int numPeers = nRanks_ - 1;

  signalBufferSize_ = config_.signalCount * sizeof(uint64_t) * numPeers;

  // Allocate GPU memory for signal buffer (transport-managed)
  void* signalBufferCpu = nullptr;
  doca_error_t err = doca_gpu_mem_alloc(
      docaGpu_,
      signalBufferSize_,
      4096,
      DOCA_GPU_MEM_TYPE_GPU,
      &signalBuffer_,
      &signalBufferCpu);
  checkDocaError(err, "Failed to allocate GPU signal buffer");

  // Zero-initialize signal buffer
  cudaError_t cudaErr = cudaMemset(signalBuffer_, 0, signalBufferSize_);
  if (cudaErr != cudaSuccess) {
    throw std::runtime_error("Failed to zero signal buffer");
  }
}

void MultipeerIbgdaTransport::registerMemory() {
  int accessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

  // Register signal buffer
  // Try DMABUF registration first, fall back to regular reg_mr
  int dmabufFd = -1;
  doca_error_t err =
      doca_gpu_dmabuf_fd(docaGpu_, signalBuffer_, signalBufferSize_, &dmabufFd);
  if (err == DOCA_SUCCESS && dmabufFd >= 0) {
    signalMr_ = ibv_reg_dmabuf_mr(
        ibvPd_,
        0,
        signalBufferSize_,
        reinterpret_cast<uint64_t>(signalBuffer_),
        dmabufFd,
        accessFlags);
  }
  if (!signalMr_) {
    signalMr_ =
        ibv_reg_mr(ibvPd_, signalBuffer_, signalBufferSize_, accessFlags);
    if (!signalMr_) {
      throw std::runtime_error("Failed to register signal memory region");
    }
  }

  LOG(INFO) << "MultipeerIbgdaTransport: registered signal buffer"
            << " lkey=" << signalMr_->lkey << " rkey=" << signalMr_->rkey;
}

void MultipeerIbgdaTransport::createQps() {
  const int numPeers = nRanks_ - 1;
  qpHlList_.resize(numPeers, nullptr);

  // Verify CUDA device is still set correctly
  int currentDevice = -1;
  cudaError_t cudaErr = cudaGetDevice(&currentDevice);
  if (cudaErr != cudaSuccess) {
    throw std::runtime_error(
        "Failed to get CUDA device: " +
        std::string(cudaGetErrorString(cudaErr)));
  }
  LOG(INFO) << "MultipeerIbgdaTransport::createQps: current CUDA device="
            << currentDevice << " expected=" << config_.cudaDevice;

  // Query IB device capabilities for debugging
  ibv_device_attr devAttr{};
  if (ibv_query_device(ibvCtx_, &devAttr) == 0) {
    LOG(INFO) << "MultipeerIbgdaTransport: IB device - max_qp="
              << devAttr.max_qp << " max_cq=" << devAttr.max_cq
              << " max_mr=" << devAttr.max_mr
              << " max_qp_wr=" << devAttr.max_qp_wr;
  }

  doca_gpu_verbs_qp_init_attr_hl initAttr{};
  initAttr.gpu_dev = docaGpu_;
  initAttr.ibpd = ibvPd_;
  initAttr.sq_nwqe = config_.qpDepth;
  initAttr.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
  initAttr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;

  LOG(INFO) << "MultipeerIbgdaTransport: creating " << numPeers << " QPs"
            << " gpu_dev=" << (void*)docaGpu_
            << " ibpd=" << (void*)initAttr.ibpd
            << " sq_nwqe=" << config_.qpDepth
            << " nic_handler=AUTO mreg_type=DEFAULT";

  for (int i = 0; i < numPeers; i++) {
    LOG(INFO) << "MultipeerIbgdaTransport: creating QP " << i;
    doca_error_t err = doca_gpu_verbs_create_qp_hl(&initAttr, &qpHlList_[i]);
    if (err != DOCA_SUCCESS) {
      LOG(ERROR) << "MultipeerIbgdaTransport: QP " << i
                 << " creation failed: " << docaErrorToString(err) << " (code "
                 << (int)err << ")";
      checkDocaError(err, "Failed to create high-level QP");
    }

    LOG(INFO) << "MultipeerIbgdaTransport: created QP " << i
              << " qpn=" << doca_verbs_qp_get_qpn(qpHlList_[i]->qp);
  }
}

void MultipeerIbgdaTransport::connectQp(
    doca_gpu_verbs_qp_hl* qpHl,
    const IbgdaTransportExchInfo& peerInfo) {
  // Set remote GID in AH attributes
  doca_verbs_gid remoteGid{};
  memcpy(remoteGid.raw, peerInfo.gid, sizeof(remoteGid.raw));
  doca_error_t err = doca_verbs_ah_attr_set_gid(ahAttr_, remoteGid);
  checkDocaError(err, "Failed to set remote GID");

  // Query port for IB-specific parameters
  ibv_port_attr portAttr{};
  if (ibv_query_port(ibvCtx_, 1, &portAttr) != 0) {
    LOG(WARNING) << "Failed to query port for IB-specific parameters";
  } else if (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    err = doca_verbs_ah_attr_set_dlid(ahAttr_, peerInfo.lid);
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
    err = doca_verbs_qp_attr_set_ah_attr(qpAttr, ahAttr_);
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

  LOG(INFO) << "MultipeerIbgdaTransport: connected QP to remote qpn="
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
    std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
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
  if (config.signalCount == 0) {
    throw std::invalid_argument("signalCount must be > 0");
  }

  try {
    // Initialize DOCA GPU context
    initDocaGpu();

    // Open IB device and create PD
    openIbDevice();

    // Allocate GPU memory
    allocateResources();

    // Register memory for RDMA
    registerMemory();

    // Create high-level QPs
    createQps();
  } catch (const std::exception&) {
    // Destructor won't run for a partially-constructed object, so clean up
    // all resources allocated by the init methods above.
    cleanup();
    throw;
  }

  LOG(INFO) << "MultipeerIbgdaTransport: rank " << myRank_ << "/" << nRanks_
            << " initialized on GPU " << gpuPciBusId_;
}

MultipeerIbgdaTransport::~MultipeerIbgdaTransport() {
  cleanup();
}

void MultipeerIbgdaTransport::cleanup() {
  // Free GPU transport memory
  if (peerTransportsGpu_ != nullptr) {
    freeDeviceTransportsOnGpu(peerTransportsGpu_);
    peerTransportsGpu_ = nullptr;
  }

  // Destroy high-level QPs
  for (auto* qpHl : qpHlList_) {
    if (qpHl != nullptr) {
      doca_gpu_verbs_destroy_qp_hl(qpHl);
    }
  }
  qpHlList_.clear();

  // Destroy user buffer MRs
  for (auto& [_, mr] : registeredBuffers_) {
    ibv_dereg_mr(mr);
  }
  registeredBuffers_.clear();

  // Destroy signal MR
  if (signalMr_) {
    ibv_dereg_mr(signalMr_);
    signalMr_ = nullptr;
  }

  // Free signal buffer (transport-managed)
  if (signalBuffer_ != nullptr) {
    doca_gpu_mem_free(docaGpu_, signalBuffer_);
    signalBuffer_ = nullptr;
  }

  // Destroy AH attributes
  if (ahAttr_ != nullptr) {
    doca_verbs_ah_attr_destroy(ahAttr_);
    ahAttr_ = nullptr;
  }

  // Destroy PD
  if (ibvPd_) {
    ibv_dealloc_pd(ibvPd_);
    ibvPd_ = nullptr;
  }

  // Close device
  if (ibvCtx_) {
    ibv_close_device(ibvCtx_);
    ibvCtx_ = nullptr;
  }

  // Destroy DOCA GPU context
  if (docaGpu_ != nullptr) {
    doca_gpu_destroy(docaGpu_);
    docaGpu_ = nullptr;
  }
}

void MultipeerIbgdaTransport::exchange() {
  const int numPeers = nRanks_ - 1;

  // Validate rank count for allGather-based exchange
  if (nRanks_ > kMaxRanksForAllGather) {
    throw std::runtime_error(
        fmt::format(
            "Too many ranks ({}) for allGather-based exchange, max is {}",
            nRanks_,
            kMaxRanksForAllGather));
  }

  // Build local exchange info for allGather
  // Allocate buffer for allGather: one entry per rank
  std::vector<IbgdaTransportExchInfoAll> allInfo(nRanks_);

  // Fill in my info at my rank's slot
  IbgdaTransportExchInfoAll& myInfo = allInfo[myRank_];
  memcpy(myInfo.gid, localGid_.raw, sizeof(myInfo.gid));
  myInfo.gidIndex = gidIndex_;
  myInfo.signalAddr = reinterpret_cast<uint64_t>(signalBuffer_);
  myInfo.signalRkey = HostRKey(signalMr_->rkey);
  myInfo.mtu = localMtu_;

  // Query port for LID (IB only)
  ibv_port_attr exchPortAttr{};
  if (ibv_query_port(ibvCtx_, 1, &exchPortAttr) != 0) {
    LOG(WARNING) << "Failed to query port for LID";
  } else {
    myInfo.lid = exchPortAttr.lid;
  }

  // Fill in per-target QPNs
  // qpnForRank[j] = QPN I use to connect to rank j
  for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
    int peerRank = peerIndexToRank(peerIndex);
    myInfo.qpnForRank[peerRank] =
        doca_verbs_qp_get_qpn(qpHlList_[peerIndex]->qp);
  }
  myInfo.qpnForRank[myRank_] = 0; // Unused (self)

  LOG(INFO) << "MultipeerIbgdaTransport: rank " << myRank_
            << " performing allGather exchange";

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

  // Convert allGather results to per-peer IbgdaExchInfo
  // For each peer, extract their info and the QPN they use to connect to me
  peerExchInfo_.resize(numPeers);
  for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
    int peerRank = peerIndexToRank(peerIndex);
    const IbgdaTransportExchInfoAll& peerInfo = allInfo[peerRank];

    // Extract transport info
    // The QPN we need is the one peer uses to connect to us:
    // peerInfo.qpnForRank[myRank_]
    peerExchInfo_[peerIndex].transport.qpn = peerInfo.qpnForRank[myRank_];
    memcpy(
        peerExchInfo_[peerIndex].transport.gid,
        peerInfo.gid,
        sizeof(peerInfo.gid));
    peerExchInfo_[peerIndex].transport.gidIndex = peerInfo.gidIndex;
    peerExchInfo_[peerIndex].transport.lid = peerInfo.lid;
    peerExchInfo_[peerIndex].transport.mtu = peerInfo.mtu;

    // Extract signal buffer info
    peerExchInfo_[peerIndex].signal.addr = peerInfo.signalAddr;
    peerExchInfo_[peerIndex].signal.rkey = peerInfo.signalRkey;

    LOG(INFO) << "MultipeerIbgdaTransport: received from peer " << peerRank
              << " qpn=" << peerExchInfo_[peerIndex].transport.qpn
              << " signalAddr=0x" << std::hex
              << peerExchInfo_[peerIndex].signal.addr << std::dec;
  }

  // Connect QPs to peers
  for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
    connectQp(qpHlList_[peerIndex], peerExchInfo_[peerIndex].transport);
  }

  // Build device transports on GPU
  std::vector<P2pIbgdaTransportBuildParams> buildParams(numPeers);
  for (int i = 0; i < numPeers; i++) {
    // Get GPU-accessible QP handle
    doca_gpu_dev_verbs_qp* gpuQp = nullptr;
    doca_error_t err =
        doca_gpu_verbs_get_qp_dev(qpHlList_[i]->qp_gverbs, &gpuQp);
    checkDocaError(err, "Failed to get GPU QP handle");

    // Local signal buffer: use my peer index i for my own buffer layout
    std::size_t localSignalOffset = i * config_.signalCount * sizeof(uint64_t);

    // Remote signal buffer: use the peer's index for ME, not my index
    // for the peer. The remote rank partitions its signal buffer using
    // its own peer indexing (skip-self), so the slice reserved for us
    // is at the index the remote rank assigns to myRank_.
    int peerRank = peerIndexToRank(i);
    int myIndexOnPeer = (myRank_ < peerRank) ? myRank_ : (myRank_ - 1);
    std::size_t remoteSignalOffset =
        myIndexOnPeer * config_.signalCount * sizeof(uint64_t);

    auto* remoteSignalPtr = reinterpret_cast<void*>( // NOLINT(performance-no-int-to-ptr)
        peerExchInfo_[i].signal.addr + remoteSignalOffset);

    buildParams[i] = P2pIbgdaTransportBuildParams{
        gpuQp,
        IbgdaLocalBuffer(
            static_cast<char*>(signalBuffer_) + localSignalOffset,
            HostLKey(signalMr_->lkey)),
        IbgdaRemoteBuffer(remoteSignalPtr, peerExchInfo_[i].signal.rkey),
        static_cast<int>(config_.signalCount)};
  }

  peerTransportsGpu_ = buildDeviceTransportsOnGpu(buildParams.data(), numPeers);
  peerTransportSize_ = getP2pIbgdaTransportDeviceSize();

  LOG(INFO) << "MultipeerIbgdaTransport: rank " << myRank_
            << " exchange complete, connected to " << numPeers << " peers";
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
  // Use byte-level arithmetic since P2pIbgdaTransportDevice is incomplete here
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

IbgdaLocalBuffer MultipeerIbgdaTransport::registerBuffer(
    void* ptr,
    std::size_t size) {
  if (ptr == nullptr || size == 0) {
    throw std::invalid_argument("Invalid buffer pointer or size");
  }

  // If already registered, return existing registration (no-op)
  auto existingIt = registeredBuffers_.find(ptr);
  if (existingIt != registeredBuffers_.end()) {
    return IbgdaLocalBuffer(ptr, HostLKey(existingIt->second->lkey));
  }

  int accessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

  // Try DMABUF registration first, fall back to regular reg_mr
  ibv_mr* mr = nullptr;
  int dmabufFd = -1;
  doca_error_t err = doca_gpu_dmabuf_fd(docaGpu_, ptr, size, &dmabufFd);
  if (err == DOCA_SUCCESS && dmabufFd >= 0) {
    mr = ibv_reg_dmabuf_mr(
        ibvPd_,
        0,
        size,
        reinterpret_cast<uint64_t>(ptr),
        dmabufFd,
        accessFlags);
  }
  if (!mr) {
    mr = ibv_reg_mr(ibvPd_, ptr, size, accessFlags);
    if (!mr) {
      throw std::runtime_error("Failed to register buffer with RDMA");
    }
  }

  uint32_t lkey = mr->lkey;
  uint32_t rkey = mr->rkey;
  registeredBuffers_.emplace(ptr, mr);

  LOG(INFO) << "MultipeerIbgdaTransport: registered user buffer ptr=" << ptr
            << " size=" << size << " lkey=" << lkey << " rkey=" << rkey;

  return IbgdaLocalBuffer(ptr, HostLKey(lkey));
}

void MultipeerIbgdaTransport::deregisterBuffer(void* ptr) {
  auto it = registeredBuffers_.find(ptr);
  if (it == registeredBuffers_.end()) {
    LOG(WARNING) << "MultipeerIbgdaTransport: buffer not registered: " << ptr;
    return;
  }

  ibv_dereg_mr(it->second);
  registeredBuffers_.erase(it);

  LOG(INFO) << "MultipeerIbgdaTransport: deregistered buffer ptr=" << ptr;
}

std::vector<IbgdaRemoteBuffer> MultipeerIbgdaTransport::exchangeBuffer(
    const IbgdaLocalBuffer& localBuf) {
  const int numPeers = nRanks_ - 1;

  // Find the MR for this buffer
  auto it = registeredBuffers_.find(localBuf.ptr);
  if (it == registeredBuffers_.end()) {
    throw std::runtime_error(
        "Buffer not registered - call registerBuffer() first");
  }

  // Allocate buffer for allGather: one entry per rank
  std::vector<IbgdaBufferExchInfo> allInfo(nRanks_);

  // Write my info at my rank's slot
  allInfo[myRank_] = IbgdaBufferExchInfo{
      reinterpret_cast<uint64_t>(localBuf.ptr),
      HostRKey(it->second->rkey),
  };

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
  //                          myRank_..numPeers-1 -> ranks myRank_+1..nRanks_-1
  std::vector<IbgdaRemoteBuffer> peerBuffers(numPeers);
  for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
    int peerRank = peerIndexToRank(peerIndex);
    peerBuffers[peerIndex] = allInfo[peerRank].toRemoteBuffer();
  }

  LOG(INFO) << "MultipeerIbgdaTransport: exchanged buffer info with "
            << numPeers << " peers";

  return peerBuffers;
}
int MultipeerIbgdaTransport::nRanks() const {
  return nRanks_;
}

} // namespace comms::pipes
