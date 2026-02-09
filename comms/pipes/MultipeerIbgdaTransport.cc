// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultipeerIbgdaTransport.h"

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <unistd.h>

#include <climits>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>

#include "comms/pipes/MultipeerIbgdaDeviceTransport.cuh"
#include "comms/pipes/MultipeerIbgdaTransportCuda.cuh"

namespace comms::pipes {

namespace {

constexpr int kDefaultGidIndex = 3; // Default GID index (sample uses 0)
constexpr int kQueueSize = 128; // Number of WQEs (matches ncclx default)
constexpr int kHopLimit = 255;

// Sysfs paths for device discovery
constexpr std::string_view kSysClassInfiniband = "/sys/class/infiniband/";

// Helper to get PCIe bus ID string from CUDA device
// Uses cudaDeviceGetPCIBusId to get the exact format DOCA expects
std::string getCudaPciBusId(int cudaDevice) {
  std::string busId(32, '\0');
  cudaError_t err = cudaDeviceGetPCIBusId(
      busId.data(), static_cast<int>(busId.size()), cudaDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        fmt::format(
            "Failed to get CUDA device PCIe bus ID: {}",
            cudaGetErrorString(err)));
  }
  // Trim null terminator
  busId.resize(std::strlen(busId.c_str()));
  return busId;
}

// Convert DOCA error to string
const char* docaErrorToString(doca_error_t err) {
  switch (err) {
    case DOCA_SUCCESS:
      return "DOCA_SUCCESS";
    case DOCA_ERROR_UNKNOWN:
      return "DOCA_ERROR_UNKNOWN";
    case DOCA_ERROR_NOT_PERMITTED:
      return "DOCA_ERROR_NOT_PERMITTED";
    case DOCA_ERROR_IN_USE:
      return "DOCA_ERROR_IN_USE";
    case DOCA_ERROR_NOT_SUPPORTED:
      return "DOCA_ERROR_NOT_SUPPORTED";
    case DOCA_ERROR_AGAIN:
      return "DOCA_ERROR_AGAIN";
    case DOCA_ERROR_INVALID_VALUE:
      return "DOCA_ERROR_INVALID_VALUE";
    case DOCA_ERROR_NO_MEMORY:
      return "DOCA_ERROR_NO_MEMORY";
    case DOCA_ERROR_INITIALIZATION:
      return "DOCA_ERROR_INITIALIZATION";
    case DOCA_ERROR_TIME_OUT:
      return "DOCA_ERROR_TIME_OUT";
    case DOCA_ERROR_SHUTDOWN:
      return "DOCA_ERROR_SHUTDOWN";
    case DOCA_ERROR_CONNECTION_RESET:
      return "DOCA_ERROR_CONNECTION_RESET";
    case DOCA_ERROR_CONNECTION_ABORTED:
      return "DOCA_ERROR_CONNECTION_ABORTED";
    case DOCA_ERROR_CONNECTION_INPROGRESS:
      return "DOCA_ERROR_CONNECTION_INPROGRESS";
    case DOCA_ERROR_NOT_CONNECTED:
      return "DOCA_ERROR_NOT_CONNECTED";
    case DOCA_ERROR_NO_LOCK:
      return "DOCA_ERROR_NO_LOCK";
    case DOCA_ERROR_NOT_FOUND:
      return "DOCA_ERROR_NOT_FOUND";
    case DOCA_ERROR_IO_FAILED:
      return "DOCA_ERROR_IO_FAILED";
    case DOCA_ERROR_BAD_STATE:
      return "DOCA_ERROR_BAD_STATE";
    case DOCA_ERROR_UNSUPPORTED_VERSION:
      return "DOCA_ERROR_UNSUPPORTED_VERSION";
    case DOCA_ERROR_OPERATING_SYSTEM:
      return "DOCA_ERROR_OPERATING_SYSTEM";
    case DOCA_ERROR_DRIVER:
      return "DOCA_ERROR_DRIVER (check DOCA GPUNetIO driver/infrastructure)";
    case DOCA_ERROR_UNEXPECTED:
      return "DOCA_ERROR_UNEXPECTED";
    case DOCA_ERROR_ALREADY_EXIST:
      return "DOCA_ERROR_ALREADY_EXIST";
    case DOCA_ERROR_FULL:
      return "DOCA_ERROR_FULL";
    case DOCA_ERROR_EMPTY:
      return "DOCA_ERROR_EMPTY";
    case DOCA_ERROR_IN_PROGRESS:
      return "DOCA_ERROR_IN_PROGRESS";
    case DOCA_ERROR_TOO_BIG:
      return "DOCA_ERROR_TOO_BIG";
    case DOCA_ERROR_AUTHENTICATION:
      return "DOCA_ERROR_AUTHENTICATION";
    case DOCA_ERROR_BAD_CONFIG:
      return "DOCA_ERROR_BAD_CONFIG";
    case DOCA_ERROR_SKIPPED:
      return "DOCA_ERROR_SKIPPED";
    case DOCA_ERROR_DEVICE_FATAL_ERROR:
      return "DOCA_ERROR_DEVICE_FATAL_ERROR";
    default:
      return "DOCA_ERROR_UNKNOWN_CODE";
  }
}

// Check DOCA error and throw on failure
void checkDocaError(doca_error_t err, const char* msg) {
  if (err != DOCA_SUCCESS) {
    throw std::runtime_error(std::string(msg) + ": " + docaErrorToString(err));
  }
}

// Get PCIe bus ID for an IB device (e.g., "0000:18:00.0")
std::string getPcieForIbDev(std::string_view devName) {
  auto devPath = fmt::format("{}{}/device", kSysClassInfiniband, devName);
  std::string linkBuf(PATH_MAX, '\0');
  ssize_t len = readlink(devPath.c_str(), linkBuf.data(), linkBuf.size() - 1);
  if (len <= 0) {
    return "";
  }
  linkBuf.resize(len);
  // linkBuf is like "../../../0000:18:00.0", extract the last component
  auto pos = linkBuf.rfind('/');
  if (pos != std::string::npos) {
    return linkBuf.substr(pos + 1);
  }
  return linkBuf;
}

// Calculate PCIe "distance" based on bus ID similarity
// Lower is better; 0 = same switch, higher = different switches
int getPcieBusDistance(const std::string& pcie1, const std::string& pcie2) {
  // Extract bus number (e.g., "1b" from "0000:1b:00.0" or "0000:1B:00.0")
  // Format: domain:bus:device.function
  auto extractBus = [](const std::string& pcie) -> int {
    auto colonPos = pcie.find(':');
    if (colonPos == std::string::npos) {
      return -1;
    }
    auto secondColon = pcie.find(':', colonPos + 1);
    if (secondColon == std::string::npos) {
      return -1;
    }
    std::string busStr = pcie.substr(colonPos + 1, secondColon - colonPos - 1);
    // stoi with base 16 handles both uppercase and lowercase hex
    try {
      return std::stoi(busStr, nullptr, 16);
    } catch (...) {
      return -1;
    }
  };

  int bus1 = extractBus(pcie1);
  int bus2 = extractBus(pcie2);
  if (bus1 < 0 || bus2 < 0) {
    return INT_MAX;
  }

  // Check if on same PCIe switch (top nibble of bus matches)
  // e.g., GPU at 0x1b and NIC at 0x18 are both in 0x1x range
  int switch1 = bus1 >> 4;
  int switch2 = bus2 >> 4;
  if (switch1 == switch2) {
    return std::abs(bus1 - bus2); // Small difference = PIX
  }
  return 1000 + std::abs(bus1 - bus2); // Different switch = SYS
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

  gpuPciBusId_ = getCudaPciBusId(config_.cudaDevice);

  LOG(INFO) << "MultipeerIbgdaTransport: GPU " << config_.cudaDevice << " PCIe "
            << gpuPciBusId_;

  doca_error_t err = doca_gpu_create(gpuPciBusId_.c_str(), &docaGpu_);
  checkDocaError(err, "Failed to create DOCA GPU context");

  LOG(INFO) << "MultipeerIbgdaTransport: DOCA GPU context created: "
            << (void*)docaGpu_;

  gidIndex_ = config_.gidIndex.value_or(kDefaultGidIndex);
}

void MultipeerIbgdaTransport::openIbDevice() {
  int numDevices = 0;
  ibv_device** deviceList = ibv_get_device_list(&numDevices);
  if (deviceList == nullptr || numDevices == 0) {
    throw std::runtime_error("No IB devices found");
  }

  // Find device with best PCIe topology affinity (PIX = same switch)
  ibv_device* bestDevice = nullptr;
  int bestDistance = INT_MAX;
  std::string bestNicPcie;

  for (int i = 0; i < numDevices; i++) {
    const char* devName = ibv_get_device_name(deviceList[i]);

    // If user specified a NIC, look for it
    if (config_.nicDeviceName.has_value()) {
      if (config_.nicDeviceName.value() == devName) {
        bestDevice = deviceList[i];
        nicDeviceName_ = devName;
        bestNicPcie = getPcieForIbDev(devName);
        break;
      }
      continue;
    }

    // Get NIC PCIe address and calculate distance to GPU
    std::string nicPcie = getPcieForIbDev(devName);
    int distance = getPcieBusDistance(gpuPciBusId_, nicPcie);

    LOG(INFO) << "MultipeerIbgdaTransport: IB device " << devName << " PCIe "
              << nicPcie << " distance " << distance
              << (distance < 100 ? " (PIX)" : " (SYS)");

    if (distance < bestDistance) {
      bestDistance = distance;
      bestDevice = deviceList[i];
      nicDeviceName_ = devName;
      bestNicPcie = nicPcie;
    }
  }

  if (bestDevice == nullptr) {
    ibv_free_device_list(deviceList);
    throw std::runtime_error("No suitable IB device found");
  }

  LOG(INFO) << "MultipeerIbgdaTransport: selected NIC " << nicDeviceName_
            << " PCIe " << bestNicPcie << " for GPU " << gpuPciBusId_
            << " (distance=" << bestDistance << ")";

  ibvContext_ = ibv_open_device(bestDevice);
  ibv_free_device_list(deviceList);

  if (ibvContext_ == nullptr) {
    throw std::runtime_error("Failed to open IB device");
  }

  // Allocate protection domain
  ibvPd_ = ibv_alloc_pd(ibvContext_);
  if (ibvPd_ == nullptr) {
    throw std::runtime_error("Failed to allocate protection domain");
  }

  // Query local GID
  int ret = ibv_query_gid(ibvContext_, 1, gidIndex_, &localGid_);
  if (ret != 0) {
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
  ret = ibv_query_port(ibvContext_, 1, &portAttr);
  if (ret != 0) {
    throw std::runtime_error("Failed to query port attributes");
  }

  LOG(INFO) << "MultipeerIbgdaTransport: port 1 state=" << portAttr.state
            << " link_layer=" << (int)portAttr.link_layer
            << " (1=IB, 2=Ethernet)";

  if (portAttr.state != IBV_PORT_ACTIVE) {
    throw std::runtime_error(
        "Port 1 is not active (state=" + std::to_string(portAttr.state) + ")");
  }

  // Determine address type based on link layer
  // For RoCE (Ethernet), use IPv6 (our hosts use RoCEv2 with IPv6-only)
  doca_verbs_addr_type addrType =
      (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND)
      ? DOCA_VERBS_ADDR_TYPE_IB_NO_GRH
      : DOCA_VERBS_ADDR_TYPE_IPv6;

  doca_error_t err = doca_verbs_ah_attr_create(ibvContext_, &ahAttr_);
  checkDocaError(err, "Failed to create AH attributes");

  err = doca_verbs_ah_attr_set_addr_type(ahAttr_, addrType);
  checkDocaError(err, "Failed to set address type");

  err = doca_verbs_ah_attr_set_sgid_index(ahAttr_, gidIndex_);
  checkDocaError(err, "Failed to set SGID index");

  err = doca_verbs_ah_attr_set_hop_limit(ahAttr_, kHopLimit);
  checkDocaError(err, "Failed to set hop limit");
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
  if (signalMr_ == nullptr) {
    signalMr_ =
        ibv_reg_mr(ibvPd_, signalBuffer_, signalBufferSize_, accessFlags);
  }
  if (signalMr_ == nullptr) {
    throw std::runtime_error("Failed to register signal memory region");
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
  int ret = ibv_query_device(ibvContext_, &devAttr);
  if (ret == 0) {
    LOG(INFO) << "MultipeerIbgdaTransport: IB device - max_qp="
              << devAttr.max_qp << " max_cq=" << devAttr.max_cq
              << " max_mr=" << devAttr.max_mr
              << " max_qp_wr=" << devAttr.max_qp_wr;
  }

  doca_gpu_verbs_qp_init_attr_hl initAttr{};
  initAttr.gpu_dev = docaGpu_;
  initAttr.ibpd = ibvPd_;
  initAttr.sq_nwqe = kQueueSize;
  initAttr.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
  initAttr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;

  LOG(INFO) << "MultipeerIbgdaTransport: creating " << numPeers << " QPs"
            << " gpu_dev=" << (void*)docaGpu_ << " ibpd=" << (void*)ibvPd_
            << " ibpd->context=" << (void*)(ibvPd_ ? ibvPd_->context : nullptr)
            << " sq_nwqe=" << kQueueSize
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
  ibv_query_port(ibvContext_, 1, &portAttr);
  if (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
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
          DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_PKEY_INDEX |
          DOCA_VERBS_QP_ATTR_PORT_NUM);
  checkDocaError(err, "Failed to modify QP to INIT");

  // Transition to RTR state
  err = doca_verbs_qp_attr_set_next_state(qpAttr, DOCA_VERBS_QP_STATE_RTR);
  checkDocaError(err, "Failed to set next state RTR");
  err = doca_verbs_qp_attr_set_path_mtu(qpAttr, DOCA_VERBS_MTU_SIZE_1K_BYTES);
  checkDocaError(err, "Failed to set MTU");
  err = doca_verbs_qp_attr_set_rq_psn(qpAttr, 0);
  checkDocaError(err, "Failed to set RQ PSN");
  err = doca_verbs_qp_attr_set_dest_qp_num(qpAttr, peerInfo.qpn);
  checkDocaError(err, "Failed to set dest QP number");
  err = doca_verbs_qp_attr_set_ah_attr(qpAttr, ahAttr_);
  checkDocaError(err, "Failed to set AH attributes");
  err = doca_verbs_qp_attr_set_min_rnr_timer(qpAttr, 1);
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
  err = doca_verbs_qp_attr_set_ack_timeout(qpAttr, 14);
  checkDocaError(err, "Failed to set ACK timeout");
  err = doca_verbs_qp_attr_set_retry_cnt(qpAttr, 7);
  checkDocaError(err, "Failed to set retry count");
  err = doca_verbs_qp_attr_set_rnr_retry(qpAttr, 1);
  checkDocaError(err, "Failed to set RNR retry");

  err = doca_verbs_qp_modify(
      qpHl->qp,
      qpAttr,
      DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
          DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
          DOCA_VERBS_QP_ATTR_RNR_RETRY);
  checkDocaError(err, "Failed to modify QP to RTS");

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

  LOG(INFO) << "MultipeerIbgdaTransport: rank " << myRank_ << "/" << nRanks_
            << " initialized on GPU " << gpuPciBusId_ << " with NIC "
            << nicDeviceName_;
}

MultipeerIbgdaTransport::~MultipeerIbgdaTransport() {
  // Free GPU transport memory
  if (peerTransportsGpu_ != nullptr) {
    freeDeviceTransportsOnGpu(peerTransportsGpu_);
  }

  // Destroy high-level QPs
  for (auto* qpHl : qpHlList_) {
    if (qpHl != nullptr) {
      doca_gpu_verbs_destroy_qp_hl(qpHl);
    }
  }

  // Deregister user-registered buffers
  for (auto& [ptr, mr] : registeredBuffers_) {
    if (mr != nullptr) {
      ibv_dereg_mr(mr);
    }
  }
  registeredBuffers_.clear();

  // Deregister signal memory region
  if (signalMr_ != nullptr) {
    ibv_dereg_mr(signalMr_);
  }

  // Free signal buffer (transport-managed)
  if (signalBuffer_ != nullptr) {
    doca_gpu_mem_free(docaGpu_, signalBuffer_);
  }

  // Destroy AH attributes
  if (ahAttr_ != nullptr) {
    doca_verbs_ah_attr_destroy(ahAttr_);
  }

  // Deallocate PD
  if (ibvPd_ != nullptr) {
    ibv_dealloc_pd(ibvPd_);
  }

  // Close IB device
  if (ibvContext_ != nullptr) {
    ibv_close_device(ibvContext_);
  }

  // Destroy DOCA GPU context
  if (docaGpu_ != nullptr) {
    doca_gpu_destroy(docaGpu_);
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

  // Query port for LID (IB only)
  ibv_port_attr portAttr{};
  ibv_query_port(ibvContext_, 1, &portAttr);
  myInfo.lid = portAttr.lid;

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

    // Calculate signal buffer offsets for this peer
    std::size_t signalOffset = i * config_.signalCount * sizeof(uint64_t);

    // Calculate remote signal buffer address with offset
    auto* remoteSignalPtr = reinterpret_cast<void*>( // NOLINT(performance-no-int-to-ptr)
        peerExchInfo_[i].signal.addr + signalOffset);

    buildParams[i] = P2pIbgdaTransportBuildParams{
        gpuQp,
        IbgdaLocalBuffer(
            static_cast<char*>(signalBuffer_) + signalOffset,
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
    ibv_mr* existingMr = existingIt->second;
    return IbgdaLocalBuffer(ptr, HostLKey(existingMr->lkey));
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
  if (mr == nullptr) {
    mr = ibv_reg_mr(ibvPd_, ptr, size, accessFlags);
  }
  if (mr == nullptr) {
    throw std::runtime_error("Failed to register buffer with RDMA");
  }

  registeredBuffers_[ptr] = mr;

  LOG(INFO) << "MultipeerIbgdaTransport: registered user buffer ptr=" << ptr
            << " size=" << size << " lkey=" << mr->lkey << " rkey=" << mr->rkey;

  return IbgdaLocalBuffer(ptr, HostLKey(mr->lkey));
}

void MultipeerIbgdaTransport::deregisterBuffer(void* ptr) {
  auto it = registeredBuffers_.find(ptr);
  if (it == registeredBuffers_.end()) {
    LOG(WARNING) << "MultipeerIbgdaTransport: buffer not registered: " << ptr;
    return;
  }

  if (it->second != nullptr) {
    ibv_dereg_mr(it->second);
  }
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
  ibv_mr* mr = it->second;

  // Allocate buffer for allGather: one entry per rank
  std::vector<IbgdaBufferExchInfo> allInfo(nRanks_);

  // Write my info at my rank's slot
  allInfo[myRank_] = IbgdaBufferExchInfo{
      reinterpret_cast<uint64_t>(localBuf.ptr),
      HostRKey(mr->rkey),
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

std::string MultipeerIbgdaTransport::getNicDeviceName() const {
  return nicDeviceName_;
}

int MultipeerIbgdaTransport::nRanks() const {
  return nRanks_;
}

} // namespace comms::pipes
