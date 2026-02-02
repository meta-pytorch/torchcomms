// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultipeerIbgdaTransport.h"

#include <cuda_runtime.h>
#include <doca_gpunetio_host.h>
#include <glog/logging.h>
#include <infiniband/verbs.h>
#include <unistd.h>

#include <climits>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "comms/pipes/MultipeerIbgdaTransportCuda.cuh"

namespace comms::pipes {

namespace {

constexpr int kDefaultGidIndex = 0; // Default GID index (sample uses 0)
constexpr int kQueueSize = 128; // Number of WQEs (matches ncclx default)
constexpr int kHopLimit = 255;

// Helper to get PCIe bus ID string from CUDA device
// Uses cudaDeviceGetPCIBusId to get the exact format DOCA expects
std::string getCudaPciBusId(int cudaDevice) {
  char busId[32];
  cudaError_t err = cudaDeviceGetPCIBusId(busId, sizeof(busId), cudaDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to get CUDA device PCIe bus ID: " +
        std::string(cudaGetErrorString(err)));
  }
  return std::string(busId);
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

// Get NUMA node for a PCIe device
int getNumaNodeForPcie(const std::string& pciBusId) {
  std::string numaPath = "/sys/bus/pci/devices/" + pciBusId + "/numa_node";
  std::ifstream numaFile(numaPath);
  if (!numaFile.is_open()) {
    return -1;
  }
  int numaNode = -1;
  numaFile >> numaNode;
  return numaNode;
}

// Get NUMA node for an IB device
int getNumaNodeForIbDev(const char* devName) {
  std::string numaPath =
      std::string("/sys/class/infiniband/") + devName + "/device/numa_node";
  std::ifstream numaFile(numaPath);
  if (!numaFile.is_open()) {
    return -1;
  }
  int numaNode = -1;
  numaFile >> numaNode;
  return numaNode;
}

// Get PCIe bus ID for an IB device (e.g., "0000:18:00.0")
std::string getPcieForIbDev(const char* devName) {
  std::string devPath =
      std::string("/sys/class/infiniband/") + devName + "/device";
  char linkBuf[PATH_MAX];
  ssize_t len = readlink(devPath.c_str(), linkBuf, sizeof(linkBuf) - 1);
  if (len <= 0) {
    return "";
  }
  linkBuf[len] = '\0';
  // linkBuf is like "../../../0000:18:00.0", extract the last component
  std::string path(linkBuf);
  auto pos = path.rfind('/');
  if (pos != std::string::npos) {
    return path.substr(pos + 1);
  }
  return path;
}

// Calculate PCIe "distance" based on bus ID similarity
// Lower is better; 0 = same switch, higher = different switches
int getPcieBusDistance(const std::string& pcie1, const std::string& pcie2) {
  // Extract bus number (e.g., "1b" from "0000:1b:00.0" or "0000:1B:00.0")
  // Format: domain:bus:device.function
  auto extractBus = [](const std::string& pcie) -> int {
    auto colonPos = pcie.find(':');
    if (colonPos == std::string::npos)
      return -1;
    auto secondColon = pcie.find(':', colonPos + 1);
    if (secondColon == std::string::npos)
      return -1;
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

// Implementation struct (Pimpl)
struct MultipeerIbgdaTransport::Impl {
  // Rank information
  int myRank;
  int nRanks;

  // Bootstrap for collective operations
  std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap;

  // Configuration
  MultipeerIbgdaTransportConfig config;

  // DOCA GPU context
  doca_gpu* docaGpu{nullptr};

  // IB verbs resources
  ibv_context* ibvContext{nullptr};
  ibv_pd* ibvPd{nullptr};
  doca_verbs_ah_attr* ahAttr{nullptr};
  ibv_gid localGid{};

  // High-level QPs (one per peer)
  std::vector<doca_gpu_verbs_qp_hl*> qpHlList;

  // GPU memory
  void* dataBuffer{nullptr};
  void* signalBuffer{nullptr};
  std::size_t dataBufferSize{0};
  std::size_t signalBufferSize{0};

  // Memory regions
  ibv_mr* dataMr{nullptr};
  ibv_mr* signalMr{nullptr};

  // GPU PCIe bus ID and NIC device name
  std::string gpuPciBusId;
  std::string nicDeviceName;
  int gidIndex{kDefaultGidIndex};

  // Per-peer device transports (GPU accessible)
  P2pIbgdaTransportDevice* peerTransportsGpu{nullptr};
  std::size_t peerTransportSize{0};

  // Exchange info received from peers
  std::vector<IbgdaExchInfo> peerExchInfo;

  // Helper methods
  void initDocaGpu();
  void openIbDevice();
  void allocateResources();
  void registerMemory();
  void createQps();
  void connectQp(doca_gpu_verbs_qp_hl* qpHl, const IbgdaExchInfo& peerInfo);
  int rankToPeerIndex(int rank) const;
  int peerIndexToRank(int peerIndex) const;
};

void MultipeerIbgdaTransport::Impl::initDocaGpu() {
  // CRITICAL: Set CUDA device before any DOCA GPU operations
  cudaError_t cudaErr = cudaSetDevice(config.cudaDevice);
  if (cudaErr != cudaSuccess) {
    throw std::runtime_error(
        "Failed to set CUDA device: " +
        std::string(cudaGetErrorString(cudaErr)));
  }

  gpuPciBusId = getCudaPciBusId(config.cudaDevice);

  LOG(INFO) << "MultipeerIbgdaTransport: GPU " << config.cudaDevice << " PCIe "
            << gpuPciBusId;

  doca_error_t err = doca_gpu_create(gpuPciBusId.c_str(), &docaGpu);
  checkDocaError(err, "Failed to create DOCA GPU context");

  LOG(INFO) << "MultipeerIbgdaTransport: DOCA GPU context created: "
            << (void*)docaGpu;

  gidIndex = config.gidIndex.value_or(kDefaultGidIndex);
}

void MultipeerIbgdaTransport::Impl::openIbDevice() {
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
    if (config.nicDeviceName.has_value()) {
      if (config.nicDeviceName.value() == devName) {
        bestDevice = deviceList[i];
        nicDeviceName = devName;
        bestNicPcie = getPcieForIbDev(devName);
        break;
      }
      continue;
    }

    // Get NIC PCIe address and calculate distance to GPU
    std::string nicPcie = getPcieForIbDev(devName);
    int distance = getPcieBusDistance(gpuPciBusId, nicPcie);

    LOG(INFO) << "MultipeerIbgdaTransport: IB device " << devName << " PCIe "
              << nicPcie << " distance " << distance
              << (distance < 100 ? " (PIX)" : " (SYS)");

    if (distance < bestDistance) {
      bestDistance = distance;
      bestDevice = deviceList[i];
      nicDeviceName = devName;
      bestNicPcie = nicPcie;
    }
  }

  if (bestDevice == nullptr) {
    ibv_free_device_list(deviceList);
    throw std::runtime_error("No suitable IB device found");
  }

  LOG(INFO) << "MultipeerIbgdaTransport: selected NIC " << nicDeviceName
            << " PCIe " << bestNicPcie << " for GPU " << gpuPciBusId
            << " (distance=" << bestDistance << ")";

  ibvContext = ibv_open_device(bestDevice);
  ibv_free_device_list(deviceList);

  if (ibvContext == nullptr) {
    throw std::runtime_error("Failed to open IB device");
  }

  // Allocate protection domain
  ibvPd = ibv_alloc_pd(ibvContext);
  if (ibvPd == nullptr) {
    throw std::runtime_error("Failed to allocate protection domain");
  }

  // Query local GID
  int ret = ibv_query_gid(ibvContext, 1, gidIndex, &localGid);
  if (ret != 0) {
    throw std::runtime_error(
        "Failed to query GID at index " + std::to_string(gidIndex));
  }

  // Print GID value for debugging
  char gidStr[64];
  snprintf(
      gidStr,
      sizeof(gidStr),
      "%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x",
      localGid.raw[0],
      localGid.raw[1],
      localGid.raw[2],
      localGid.raw[3],
      localGid.raw[4],
      localGid.raw[5],
      localGid.raw[6],
      localGid.raw[7],
      localGid.raw[8],
      localGid.raw[9],
      localGid.raw[10],
      localGid.raw[11],
      localGid.raw[12],
      localGid.raw[13],
      localGid.raw[14],
      localGid.raw[15]);
  LOG(INFO) << "MultipeerIbgdaTransport: GID index " << gidIndex << " = "
            << gidStr;

  // Query port to determine link layer (IB vs Ethernet)
  ibv_port_attr portAttr{};
  ret = ibv_query_port(ibvContext, 1, &portAttr);
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

  // Create AH attributes
  doca_verbs_addr_type addrType =
      (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND)
      ? DOCA_VERBS_ADDR_TYPE_IB_NO_GRH
      : DOCA_VERBS_ADDR_TYPE_IPv4;

  doca_error_t err = doca_verbs_ah_attr_create(ibvContext, &ahAttr);
  checkDocaError(err, "Failed to create AH attributes");

  err = doca_verbs_ah_attr_set_addr_type(ahAttr, addrType);
  checkDocaError(err, "Failed to set address type");

  err = doca_verbs_ah_attr_set_sgid_index(ahAttr, gidIndex);
  checkDocaError(err, "Failed to set SGID index");

  err = doca_verbs_ah_attr_set_hop_limit(ahAttr, kHopLimit);
  checkDocaError(err, "Failed to set hop limit");
}

void MultipeerIbgdaTransport::Impl::allocateResources() {
  const int numPeers = nRanks - 1;

  dataBufferSize = config.dataBufferSize * numPeers;
  signalBufferSize = config.signalCount * sizeof(uint64_t) * numPeers;

  // Allocate GPU memory for data buffer
  void* dataBufferCpu = nullptr;
  doca_error_t err = doca_gpu_mem_alloc(
      docaGpu,
      dataBufferSize,
      4096,
      DOCA_GPU_MEM_TYPE_GPU,
      &dataBuffer,
      &dataBufferCpu);
  checkDocaError(err, "Failed to allocate GPU data buffer");

  // Allocate GPU memory for signal buffer
  void* signalBufferCpu = nullptr;
  err = doca_gpu_mem_alloc(
      docaGpu,
      signalBufferSize,
      4096,
      DOCA_GPU_MEM_TYPE_GPU,
      &signalBuffer,
      &signalBufferCpu);
  checkDocaError(err, "Failed to allocate GPU signal buffer");

  // Zero-initialize signal buffer
  cudaError_t cudaErr = cudaMemset(signalBuffer, 0, signalBufferSize);
  if (cudaErr != cudaSuccess) {
    throw std::runtime_error("Failed to zero signal buffer");
  }
}

void MultipeerIbgdaTransport::Impl::registerMemory() {
  int accessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

  // Try DMABUF registration first, fall back to regular reg_mr
  int dmabufFd = -1;
  doca_error_t err =
      doca_gpu_dmabuf_fd(docaGpu, dataBuffer, dataBufferSize, &dmabufFd);
  if (err == DOCA_SUCCESS && dmabufFd >= 0) {
    dataMr = ibv_reg_dmabuf_mr(
        ibvPd,
        0,
        dataBufferSize,
        reinterpret_cast<uint64_t>(dataBuffer),
        dmabufFd,
        accessFlags);
  }
  if (dataMr == nullptr) {
    dataMr = ibv_reg_mr(ibvPd, dataBuffer, dataBufferSize, accessFlags);
  }
  if (dataMr == nullptr) {
    throw std::runtime_error("Failed to register data memory region");
  }

  // Register signal buffer
  dmabufFd = -1;
  err = doca_gpu_dmabuf_fd(docaGpu, signalBuffer, signalBufferSize, &dmabufFd);
  if (err == DOCA_SUCCESS && dmabufFd >= 0) {
    signalMr = ibv_reg_dmabuf_mr(
        ibvPd,
        0,
        signalBufferSize,
        reinterpret_cast<uint64_t>(signalBuffer),
        dmabufFd,
        accessFlags);
  }
  if (signalMr == nullptr) {
    signalMr = ibv_reg_mr(ibvPd, signalBuffer, signalBufferSize, accessFlags);
  }
  if (signalMr == nullptr) {
    throw std::runtime_error("Failed to register signal memory region");
  }

  LOG(INFO) << "MultipeerIbgdaTransport: registered memory - data lkey="
            << dataMr->lkey << " rkey=" << dataMr->rkey
            << " signal lkey=" << signalMr->lkey << " rkey=" << signalMr->rkey;
}

void MultipeerIbgdaTransport::Impl::createQps() {
  const int numPeers = nRanks - 1;
  qpHlList.resize(numPeers, nullptr);

  // Verify CUDA device is still set correctly
  int currentDevice = -1;
  cudaGetDevice(&currentDevice);
  LOG(INFO) << "MultipeerIbgdaTransport::createQps: current CUDA device="
            << currentDevice << " expected=" << config.cudaDevice;

  // Query IB device capabilities for debugging
  ibv_device_attr devAttr{};
  int ret = ibv_query_device(ibvContext, &devAttr);
  if (ret == 0) {
    LOG(INFO) << "MultipeerIbgdaTransport: IB device - max_qp="
              << devAttr.max_qp << " max_cq=" << devAttr.max_cq
              << " max_mr=" << devAttr.max_mr
              << " max_qp_wr=" << devAttr.max_qp_wr;
  }

  doca_gpu_verbs_qp_init_attr_hl initAttr{};
  initAttr.gpu_dev = docaGpu;
  initAttr.ibpd = ibvPd;
  initAttr.sq_nwqe = kQueueSize;
  initAttr.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
  initAttr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;

  LOG(INFO) << "MultipeerIbgdaTransport: creating " << numPeers << " QPs"
            << " gpu_dev=" << (void*)docaGpu << " ibpd=" << (void*)ibvPd
            << " ibpd->context=" << (void*)(ibvPd ? ibvPd->context : nullptr)
            << " sq_nwqe=" << kQueueSize
            << " nic_handler=AUTO mreg_type=DEFAULT";

  for (int i = 0; i < numPeers; i++) {
    LOG(INFO) << "MultipeerIbgdaTransport: creating QP " << i;
    doca_error_t err = doca_gpu_verbs_create_qp_hl(&initAttr, &qpHlList[i]);
    if (err != DOCA_SUCCESS) {
      LOG(ERROR) << "MultipeerIbgdaTransport: QP " << i
                 << " creation failed: " << docaErrorToString(err) << " (code "
                 << (int)err << ")";
      checkDocaError(err, "Failed to create high-level QP");
    }

    LOG(INFO) << "MultipeerIbgdaTransport: created QP " << i
              << " qpn=" << doca_verbs_qp_get_qpn(qpHlList[i]->qp);
  }
}

void MultipeerIbgdaTransport::Impl::connectQp(
    doca_gpu_verbs_qp_hl* qpHl,
    const IbgdaExchInfo& peerInfo) {
  // Set remote GID in AH attributes
  doca_verbs_gid remoteGid{};
  memcpy(remoteGid.raw, peerInfo.gid, sizeof(remoteGid.raw));
  doca_error_t err = doca_verbs_ah_attr_set_gid(ahAttr, remoteGid);
  checkDocaError(err, "Failed to set remote GID");

  // Query port for IB-specific parameters
  ibv_port_attr portAttr{};
  ibv_query_port(ibvContext, 1, &portAttr);
  if (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    err = doca_verbs_ah_attr_set_dlid(ahAttr, peerInfo.lid);
    checkDocaError(err, "Failed to set DLID");
  }

  // Create QP attributes for modification
  doca_verbs_qp_attr* qpAttr = nullptr;
  err = doca_verbs_qp_attr_create(&qpAttr);
  checkDocaError(err, "Failed to create QP attributes");

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
  err = doca_verbs_qp_attr_set_ah_attr(qpAttr, ahAttr);
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

int MultipeerIbgdaTransport::Impl::rankToPeerIndex(int rank) const {
  return (rank < myRank) ? rank : (rank - 1);
}

int MultipeerIbgdaTransport::Impl::peerIndexToRank(int peerIndex) const {
  return (peerIndex < myRank) ? peerIndex : (peerIndex + 1);
}

// Main class implementation

MultipeerIbgdaTransport::MultipeerIbgdaTransport(
    int myRank,
    int nRanks,
    std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
    const MultipeerIbgdaTransportConfig& config)
    : impl_(std::make_unique<Impl>()) {
  if (myRank < 0 || myRank >= nRanks) {
    throw std::invalid_argument("Invalid rank");
  }
  if (nRanks < 2) {
    throw std::invalid_argument("Need at least 2 ranks");
  }
  if (config.dataBufferSize == 0) {
    throw std::invalid_argument("dataBufferSize must be > 0");
  }
  if (config.signalCount == 0) {
    throw std::invalid_argument("signalCount must be > 0");
  }

  impl_->myRank = myRank;
  impl_->nRanks = nRanks;
  impl_->bootstrap = std::move(bootstrap);
  impl_->config = config;

  // Initialize DOCA GPU context
  impl_->initDocaGpu();

  // Open IB device and create PD
  impl_->openIbDevice();

  // Allocate GPU memory
  impl_->allocateResources();

  // Register memory for RDMA
  impl_->registerMemory();

  // Create high-level QPs
  impl_->createQps();

  LOG(INFO) << "MultipeerIbgdaTransport: rank " << impl_->myRank << "/"
            << impl_->nRanks << " initialized on GPU " << impl_->gpuPciBusId
            << " with NIC " << impl_->nicDeviceName;
}

MultipeerIbgdaTransport::~MultipeerIbgdaTransport() {
  // Free GPU transport memory
  if (impl_->peerTransportsGpu != nullptr) {
    freeDeviceTransportsOnGpu(impl_->peerTransportsGpu);
  }

  // Destroy high-level QPs
  for (auto* qpHl : impl_->qpHlList) {
    if (qpHl != nullptr) {
      doca_gpu_verbs_destroy_qp_hl(qpHl);
    }
  }

  // Deregister memory regions
  if (impl_->dataMr != nullptr) {
    ibv_dereg_mr(impl_->dataMr);
  }
  if (impl_->signalMr != nullptr) {
    ibv_dereg_mr(impl_->signalMr);
  }

  // Free GPU memory
  if (impl_->dataBuffer != nullptr) {
    doca_gpu_mem_free(impl_->docaGpu, impl_->dataBuffer);
  }
  if (impl_->signalBuffer != nullptr) {
    doca_gpu_mem_free(impl_->docaGpu, impl_->signalBuffer);
  }

  // Destroy AH attributes
  if (impl_->ahAttr != nullptr) {
    doca_verbs_ah_attr_destroy(impl_->ahAttr);
  }

  // Deallocate PD
  if (impl_->ibvPd != nullptr) {
    ibv_dealloc_pd(impl_->ibvPd);
  }

  // Close IB device
  if (impl_->ibvContext != nullptr) {
    ibv_close_device(impl_->ibvContext);
  }

  // Destroy DOCA GPU context
  if (impl_->docaGpu != nullptr) {
    doca_gpu_destroy(impl_->docaGpu);
  }
}

void MultipeerIbgdaTransport::exchange() {
  const int numPeers = impl_->nRanks - 1;

  // Build local exchange info
  IbgdaExchInfo localInfo{};
  localInfo.dataAddr = reinterpret_cast<uint64_t>(impl_->dataBuffer);
  localInfo.dataRkey = impl_->dataMr->rkey;
  localInfo.signalAddr = reinterpret_cast<uint64_t>(impl_->signalBuffer);
  localInfo.signalRkey = impl_->signalMr->rkey;
  memcpy(localInfo.gid, impl_->localGid.raw, sizeof(localInfo.gid));
  localInfo.gidIndex = impl_->gidIndex;

  // Query port for LID (IB only)
  ibv_port_attr portAttr{};
  ibv_query_port(impl_->ibvContext, 1, &portAttr);
  localInfo.lid = portAttr.lid;

  // Prepare per-peer info with QPN
  impl_->peerExchInfo.resize(numPeers);

  // Exchange info with all peers
  for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
    int peerRank = impl_->peerIndexToRank(peerIndex);

    // Set local QPN for this peer
    localInfo.qpn = doca_verbs_qp_get_qpn(impl_->qpHlList[peerIndex]->qp);

    LOG(INFO) << "MultipeerIbgdaTransport: rank " << impl_->myRank
              << " exchanging with peer " << peerRank << " (index " << peerIndex
              << ")"
              << " local qpn=" << localInfo.qpn;

    // Use unique tag per peer pair to avoid mixing messages
    constexpr int kBaseTag = 0x1234;
    int tag = kBaseTag + peerIndex;

    // Send local info to peer and receive peer info
    // Order matters: lower rank sends first to avoid deadlock
    if (impl_->myRank < peerRank) {
      impl_->bootstrap->send(&localInfo, sizeof(localInfo), peerRank, tag)
          .get();
      impl_->bootstrap
          ->recv(
              &impl_->peerExchInfo[peerIndex],
              sizeof(IbgdaExchInfo),
              peerRank,
              tag)
          .get();
    } else {
      impl_->bootstrap
          ->recv(
              &impl_->peerExchInfo[peerIndex],
              sizeof(IbgdaExchInfo),
              peerRank,
              tag)
          .get();
      impl_->bootstrap->send(&localInfo, sizeof(localInfo), peerRank, tag)
          .get();
    }

    LOG(INFO) << "MultipeerIbgdaTransport: received from peer " << peerRank
              << " qpn=" << impl_->peerExchInfo[peerIndex].qpn << " dataAddr=0x"
              << std::hex << impl_->peerExchInfo[peerIndex].dataAddr
              << std::dec;
  }

  // Connect QPs to peers
  for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
    impl_->connectQp(
        impl_->qpHlList[peerIndex], impl_->peerExchInfo[peerIndex]);
  }

  // Build device transports on GPU
  std::vector<P2pIbgdaTransportBuildParams> buildParams(numPeers);
  for (int i = 0; i < numPeers; i++) {
    // Get GPU-accessible QP handle
    doca_gpu_dev_verbs_qp* gpuQp = nullptr;
    doca_error_t err =
        doca_gpu_verbs_get_qp_dev(impl_->qpHlList[i]->qp_gverbs, &gpuQp);
    checkDocaError(err, "Failed to get GPU QP handle");

    // Calculate signal buffer offsets for this peer
    std::size_t signalOffset = i * impl_->config.signalCount * sizeof(uint64_t);

    buildParams[i] = P2pIbgdaTransportBuildParams{
        gpuQp,
        IbgdaLocalBuffer(
            static_cast<char*>(impl_->signalBuffer) + signalOffset,
            impl_->signalMr->lkey),
        IbgdaRemoteBuffer(
            reinterpret_cast<void*>(
                impl_->peerExchInfo[i].signalAddr + signalOffset),
            impl_->peerExchInfo[i].signalRkey)};
  }

  impl_->peerTransportsGpu =
      buildDeviceTransportsOnGpu(buildParams.data(), numPeers);
  impl_->peerTransportSize = getP2pIbgdaTransportDeviceSize();

  LOG(INFO) << "MultipeerIbgdaTransport: rank " << impl_->myRank
            << " exchange complete, connected to " << numPeers << " peers";
}

P2pIbgdaTransportDevice* MultipeerIbgdaTransport::getDeviceTransportPtr()
    const {
  return impl_->peerTransportsGpu;
}

int MultipeerIbgdaTransport::numPeers() const {
  return impl_->nRanks - 1;
}

int MultipeerIbgdaTransport::myRank() const {
  return impl_->myRank;
}

int MultipeerIbgdaTransport::nRanks() const {
  return impl_->nRanks;
}

IbgdaLocalBuffer MultipeerIbgdaTransport::getDataBuffer(int peerRank) const {
  int peerIndex = impl_->rankToPeerIndex(peerRank);
  std::size_t offset = peerIndex * impl_->config.dataBufferSize;
  return IbgdaLocalBuffer(
      static_cast<char*>(impl_->dataBuffer) + offset, impl_->dataMr->lkey);
}

IbgdaRemoteBuffer MultipeerIbgdaTransport::getRemoteDataBuffer(
    int peerRank) const {
  int peerIndex = impl_->rankToPeerIndex(peerRank);
  if (peerIndex < 0 ||
      peerIndex >= static_cast<int>(impl_->peerExchInfo.size())) {
    return IbgdaRemoteBuffer();
  }
  // Calculate offset for this peer's buffer on remote side
  int remotePeerIndex =
      impl_->peerExchInfo[peerIndex].qpn; // Use QPN as identifier
  (void)remotePeerIndex; // For now, return base address - offset calculation
                         // would need remote rank info
  return IbgdaRemoteBuffer(
      reinterpret_cast<void*>(impl_->peerExchInfo[peerIndex].dataAddr),
      impl_->peerExchInfo[peerIndex].dataRkey);
}

std::string MultipeerIbgdaTransport::getNicDeviceName() const {
  return impl_->nicDeviceName;
}

int MultipeerIbgdaTransport::getGidIndex() const {
  return impl_->gidIndex;
}

} // namespace comms::pipes
