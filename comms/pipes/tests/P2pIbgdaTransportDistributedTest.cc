// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/P2pIbgdaTransportDistributedTest.h"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include <host/doca_error.h>
#include <host/doca_gpunetio.h>
#include <host/doca_gpunetio_high_level.h>
#include <host/doca_verbs.h>

#include "comms/pipes/IbgdaBuffer.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"

namespace comms::pipes::tests {

// =============================================================================
// Error Checking Macros
// =============================================================================

#define CUDACHECK_TEST(cmd)          \
  do {                               \
    cudaError_t err = (cmd);         \
    if (err != cudaSuccess) {        \
      FAIL() << fmt::format(         \
          "CUDA error: {} at {}:{}", \
          cudaGetErrorString(err),   \
          __FILE__,                  \
          __LINE__);                 \
    }                                \
  } while (0)

#define DOCA_CHECK_TEST(cmd)         \
  do {                               \
    doca_error_t err = (cmd);        \
    if (err != DOCA_SUCCESS) {       \
      FAIL() << fmt::format(         \
          "DOCA error: {} at {}:{}", \
          static_cast<int>(err),     \
          __FILE__,                  \
          __LINE__);                 \
    }                                \
  } while (0)

#define DOCA_CHECK_RETURN(cmd, msg)                                   \
  do {                                                                \
    doca_error_t _status = (cmd);                                     \
    if (_status != DOCA_SUCCESS) {                                    \
      XLOGF(WARNING, "{}: error {}", msg, static_cast<int>(_status)); \
      return false;                                                   \
    }                                                                 \
  } while (0)

// =============================================================================
// Constants
// =============================================================================

constexpr int kDefaultNumSignals = 4;
constexpr size_t kDefaultDataSize = 4096;
constexpr uint16_t kDefaultQueueSize = 2048;
constexpr uint8_t kDefaultHopLimit = 255;
constexpr uint8_t kDefaultPortNum = 1;
constexpr int kDefaultGidIndex = 3;

// =============================================================================
// IB/DOCA Utility Functions (inlined from DocaVerbsUtils.h)
// =============================================================================

inline std::string readSysfs(const std::string& path) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) {
    return "";
  }
  char buf[256] = {};
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) {
    buf[--n] = '\0';
  }
  return std::string(buf);
}

inline int getNumaNode(const std::string& pciAddr) {
  std::string addr = pciAddr;
  if (addr.length() < 12) {
    addr = "0000:" + addr;
  }
  for (char& c : addr) {
    c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
  }
  std::string path = "/sys/bus/pci/devices/" + addr + "/numa_node";
  std::string content = readSysfs(path);
  if (content.empty()) {
    return -1;
  }
  return std::stoi(content);
}

inline std::string getIbDevicePciAddr(const std::string& ibDevName) {
  std::string symlinkPath = "/sys/class/infiniband/" + ibDevName + "/device";
  char resolvedPath[PATH_MAX] = {};
  if (realpath(symlinkPath.c_str(), resolvedPath) == nullptr) {
    return "";
  }
  std::string path(resolvedPath);
  size_t lastSlash = path.rfind('/');
  if (lastSlash == std::string::npos) {
    return "";
  }
  return path.substr(lastSlash + 1);
}

inline int parsePciBus(const std::string& pciAddr) {
  size_t colonPos = pciAddr.find(':');
  if (colonPos == std::string::npos) {
    return -1;
  }
  std::string busStr;
  size_t secondColon = pciAddr.find(':', colonPos + 1);
  if (secondColon != std::string::npos) {
    busStr = pciAddr.substr(colonPos + 1, secondColon - colonPos - 1);
  } else {
    busStr = pciAddr.substr(0, colonPos);
  }
  try {
    return std::stoi(busStr, nullptr, 16);
  } catch (...) {
    return -1;
  }
}

inline std::string findClosestNic(const std::string& gpuPciAddr) {
  int numDevs = 0;
  struct ibv_device** devList = ibv_get_device_list(&numDevs);
  if (!devList || numDevs == 0) {
    return "";
  }

  int gpuNuma = getNumaNode(gpuPciAddr);
  int gpuBus = parsePciBus(gpuPciAddr);

  std::string bestNic;
  int bestScore = INT_MIN;

  for (int i = 0; i < numDevs; i++) {
    const char* devName = ibv_get_device_name(devList[i]);
    if (strncmp(devName, "mlx5", 4) != 0) {
      continue;
    }
    std::string nicPciAddr = getIbDevicePciAddr(devName);
    if (nicPciAddr.empty()) {
      continue;
    }
    int nicNuma = getNumaNode(nicPciAddr);
    int nicBus = parsePciBus(nicPciAddr);

    int score = 0;
    if (gpuNuma >= 0 && nicNuma >= 0 && gpuNuma == nicNuma) {
      score += 1000;
    }
    if (gpuBus >= 0 && nicBus >= 0) {
      score += 255 - std::abs(gpuBus - nicBus);
    }
    if (score > bestScore) {
      bestScore = score;
      bestNic = devName;
    }
  }

  ibv_free_device_list(devList);
  return bestNic;
}

inline std::string findFirstMlx5Device() {
  int numDevs = 0;
  struct ibv_device** devList = ibv_get_device_list(&numDevs);
  if (!devList || numDevs == 0) {
    return "";
  }
  std::string name;
  for (int i = 0; i < numDevs; i++) {
    const char* devName = ibv_get_device_name(devList[i]);
    if (strncmp(devName, "mlx5", 4) == 0) {
      name = devName;
      break;
    }
  }
  ibv_free_device_list(devList);
  return name;
}

inline struct ibv_context* openIbDevice(const std::string& name) {
  int numDevs = 0;
  struct ibv_device** devList = ibv_get_device_list(&numDevs);
  if (!devList || numDevs == 0) {
    return nullptr;
  }
  struct ibv_context* ctx = nullptr;
  for (int i = 0; i < numDevs; i++) {
    if (name == ibv_get_device_name(devList[i])) {
      ctx = ibv_open_device(devList[i]);
      break;
    }
  }
  ibv_free_device_list(devList);
  return ctx;
}

// =============================================================================
// QP Connection Helper (inlined from DocaVerbsUtils.h)
// =============================================================================

class QpConnector {
 public:
  static bool connect(
      struct ibv_context* ctx,
      doca_verbs_qp* qp,
      uint32_t remoteQpNum,
      const union ibv_gid& remoteGid) {
    // Create and configure AH attributes
    doca_verbs_ah_attr* ahAttr = nullptr;
    if (doca_verbs_ah_attr_create(ctx, &ahAttr) != DOCA_SUCCESS || !ahAttr) {
      return false;
    }

    doca_verbs_gid docaGid{};
    memcpy(docaGid.raw, remoteGid.raw, 16);

    if (doca_verbs_ah_attr_set_addr_type(ahAttr, DOCA_VERBS_ADDR_TYPE_IPv6) !=
            DOCA_SUCCESS ||
        doca_verbs_ah_attr_set_gid(ahAttr, docaGid) != DOCA_SUCCESS ||
        doca_verbs_ah_attr_set_sgid_index(ahAttr, kDefaultGidIndex) !=
            DOCA_SUCCESS ||
        doca_verbs_ah_attr_set_hop_limit(ahAttr, kDefaultHopLimit) !=
            DOCA_SUCCESS) {
      doca_verbs_ah_attr_destroy(ahAttr);
      return false;
    }

    // Create and configure QP attributes
    doca_verbs_qp_attr* qpAttr = nullptr;
    if (doca_verbs_qp_attr_create(&qpAttr) != DOCA_SUCCESS || !qpAttr) {
      doca_verbs_ah_attr_destroy(ahAttr);
      return false;
    }

    if (doca_verbs_qp_attr_set_port_num(qpAttr, kDefaultPortNum) !=
            DOCA_SUCCESS ||
        doca_verbs_qp_attr_set_rq_psn(qpAttr, 0) != DOCA_SUCCESS ||
        doca_verbs_qp_attr_set_sq_psn(qpAttr, 0) != DOCA_SUCCESS ||
        doca_verbs_qp_attr_set_ack_timeout(qpAttr, 20) != DOCA_SUCCESS ||
        doca_verbs_qp_attr_set_retry_cnt(qpAttr, 7) != DOCA_SUCCESS ||
        doca_verbs_qp_attr_set_rnr_retry(qpAttr, 7) != DOCA_SUCCESS ||
        doca_verbs_qp_attr_set_min_rnr_timer(qpAttr, 12) != DOCA_SUCCESS ||
        doca_verbs_qp_attr_set_dest_qp_num(qpAttr, remoteQpNum) !=
            DOCA_SUCCESS ||
        doca_verbs_qp_attr_set_ah_attr(qpAttr, ahAttr) != DOCA_SUCCESS) {
      doca_verbs_qp_attr_destroy(qpAttr);
      doca_verbs_ah_attr_destroy(ahAttr);
      return false;
    }

    // RST -> INIT
    if (doca_verbs_qp_attr_set_next_state(qpAttr, DOCA_VERBS_QP_STATE_INIT) !=
            DOCA_SUCCESS ||
        doca_verbs_qp_attr_set_allow_remote_write(qpAttr, 1) != DOCA_SUCCESS ||
        doca_verbs_qp_attr_set_allow_remote_read(qpAttr, 1) != DOCA_SUCCESS ||
        doca_verbs_qp_attr_set_allow_remote_atomic(
            qpAttr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC) != DOCA_SUCCESS) {
      doca_verbs_qp_attr_destroy(qpAttr);
      doca_verbs_ah_attr_destroy(ahAttr);
      return false;
    }

    doca_error_t err = doca_verbs_qp_modify(
        qp,
        qpAttr,
        DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
            DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ |
            DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM);
    if (err != DOCA_SUCCESS) {
      XLOGF(WARNING, "Failed to modify QP to INIT: {}", static_cast<int>(err));
      doca_verbs_qp_attr_destroy(qpAttr);
      doca_verbs_ah_attr_destroy(ahAttr);
      return false;
    }

    // INIT -> RTR
    doca_verbs_qp_attr_set_next_state(qpAttr, DOCA_VERBS_QP_STATE_RTR);
    err = doca_verbs_qp_modify(
        qp,
        qpAttr,
        DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
            DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
            DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER);
    if (err != DOCA_SUCCESS) {
      XLOGF(WARNING, "Failed to modify QP to RTR: {}", static_cast<int>(err));
      doca_verbs_qp_attr_destroy(qpAttr);
      doca_verbs_ah_attr_destroy(ahAttr);
      return false;
    }

    // RTR -> RTS
    doca_verbs_qp_attr_set_next_state(qpAttr, DOCA_VERBS_QP_STATE_RTS);
    err = doca_verbs_qp_modify(
        qp,
        qpAttr,
        DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
            DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
            DOCA_VERBS_QP_ATTR_RNR_RETRY);
    if (err != DOCA_SUCCESS) {
      XLOGF(WARNING, "Failed to modify QP to RTS: {}", static_cast<int>(err));
      doca_verbs_qp_attr_destroy(qpAttr);
      doca_verbs_ah_attr_destroy(ahAttr);
      return false;
    }

    doca_verbs_qp_attr_destroy(qpAttr);
    doca_verbs_ah_attr_destroy(ahAttr);
    return true;
  }
};

// =============================================================================
// QP Exchange Info
// =============================================================================

struct QpExchangeInfo {
  uint32_t qpNum;
  uint8_t port;
  uint64_t subnetPrefix;
  uint64_t interfaceId;
  int32_t rank;
  uint64_t dataAddr;
  uint64_t signalAddr;
  uint32_t dataRkey;
  uint32_t signalRkey;
};

// =============================================================================
// Test Fixture
// =============================================================================

class P2pIbgdaTransportDistributedTest
    : public meta::comms::MpiBaseTestFixture {
 protected:
  void SetUp() override {
    meta::comms::MpiBaseTestFixture::SetUp();
    cudaGetLastError(); // Clear previous errors

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA devices available";
      return;
    }
    if (localRank >= deviceCount) {
      GTEST_SKIP() << "Not enough CUDA devices";
      return;
    }

    XLOGF(INFO, "Rank {}: Setting CUDA device to {}", globalRank, localRank);
    CUDACHECK_TEST(cudaSetDevice(localRank));
    CUDACHECK_TEST(cudaFree(0)); // Initialize CUDA context

    // Initialize all resources
    if (!initializeResources(kDefaultDataSize)) {
      GTEST_SKIP() << "Failed to initialize DOCA/IB resources";
      return;
    }

    if (!exchangeAndConnect()) {
      cleanup();
      GTEST_SKIP() << "Failed to exchange QP info and connect";
      return;
    }

    // Create device transport
    deviceTransport_ = allocateDeviceTransport(
        qpGpuDev_, localSignalBuf_, remoteSignalBuf_, kDefaultNumSignals);
    if (!deviceTransport_) {
      cleanup();
      GTEST_SKIP() << "Failed to allocate device transport";
      return;
    }

    XLOGF(INFO, "Rank {}: Setup complete", globalRank);
  }

  void TearDown() override {
    cleanup();
    meta::comms::MpiBaseTestFixture::TearDown();
  }

  // ---------------------------------------------------------------------------
  // Resource Initialization
  // ---------------------------------------------------------------------------

  bool initializeResources(size_t dataBufferSize) {
    dataBufferSize_ = dataBufferSize;

    // Detect GPU PCI address
    std::string gpuAddr = detectGpuAddr();
    if (gpuAddr.empty()) {
      XLOG(WARNING) << "Failed to detect GPU PCI address";
      return false;
    }

    // Find closest NIC based on PCIe topology
    std::string nicName = findClosestNic(gpuAddr);
    if (nicName.empty()) {
      nicName = findFirstMlx5Device();
    }
    if (nicName.empty()) {
      XLOG(WARNING) << "No suitable NIC found";
      return false;
    }
    XLOGF(INFO, "Rank {}: Using NIC={} GPU={}", globalRank, nicName, gpuAddr);

    // Create DOCA GPU device
    DOCA_CHECK_RETURN(
        doca_gpu_create(gpuAddr.c_str(), &gpuDev_),
        "Failed to create DOCA GPU");

    // Open IB device and allocate PD
    verbsCtx_ = openIbDevice(nicName);
    if (!verbsCtx_) {
      XLOG(WARNING) << "Failed to open IB device";
      return false;
    }
    verbsPd_ = ibv_alloc_pd(verbsCtx_);
    if (!verbsPd_) {
      XLOG(WARNING) << "Failed to allocate PD";
      return false;
    }

    // Query GID
    if (ibv_query_gid(
            verbsCtx_, kDefaultPortNum, kDefaultGidIndex, &localGid_) != 0) {
      XLOG(WARNING) << "Failed to query GID";
      return false;
    }

    // Create high-level QP
    doca_gpu_verbs_qp_init_attr_hl qpInit{};
    qpInit.gpu_dev = gpuDev_;
    qpInit.ibpd = verbsPd_;
    qpInit.sq_nwqe = kDefaultQueueSize;
    qpInit.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;
    qpInit.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_CUDA_DMABUF;

    DOCA_CHECK_RETURN(
        doca_gpu_verbs_create_qp_hl(&qpInit, &qpHl_), "Failed to create QP");
    localQpNum_ = doca_verbs_qp_get_qpn(qpHl_->qp);
    XLOGF(INFO, "Rank {}: Created QP with qpn={}", globalRank, localQpNum_);

    // Allocate and register buffers
    return allocateBuffers();
  }

  bool exchangeAndConnect() {
    // Prepare local info
    QpExchangeInfo localInfo{};
    localInfo.qpNum = localQpNum_;
    localInfo.port = kDefaultPortNum;
    localInfo.subnetPrefix = localGid_.global.subnet_prefix;
    localInfo.interfaceId = localGid_.global.interface_id;
    localInfo.rank = globalRank;
    localInfo.dataAddr = reinterpret_cast<uint64_t>(dataBuffer_);
    localInfo.signalAddr = reinterpret_cast<uint64_t>(signalBuffer_);
    localInfo.dataRkey = dataMr_->rkey;
    localInfo.signalRkey = signalMr_->rkey;

    // Exchange via MPI
    std::vector<QpExchangeInfo> allCards(numRanks);
    MPI_CHECK(MPI_Allgather(
        &localInfo,
        sizeof(QpExchangeInfo),
        MPI_BYTE,
        allCards.data(),
        sizeof(QpExchangeInfo),
        MPI_BYTE,
        MPI_COMM_WORLD));

    // Get peer info (rank 0 connects to rank 1, rank 1 connects to rank 0)
    int peerRank = (globalRank == 0) ? 1 : 0;
    const auto& remote = allCards[peerRank];
    remoteQpNum_ = remote.qpNum;
    remoteGid_.global.subnet_prefix = remote.subnetPrefix;
    remoteGid_.global.interface_id = remote.interfaceId;
    remoteDataAddr_ = remote.dataAddr;
    remoteSignalAddr_ = remote.signalAddr;
    remoteDataRkey_ = remote.dataRkey;
    remoteSignalRkey_ = remote.signalRkey;

    XLOGF(
        INFO,
        "Rank {}: Remote QPN={} dataAddr={:#x}",
        globalRank,
        remoteQpNum_,
        remoteDataAddr_);

    // Connect QP
    if (!QpConnector::connect(verbsCtx_, qpHl_->qp, remoteQpNum_, remoteGid_)) {
      XLOG(WARNING) << "Failed to connect QP";
      return false;
    }

    // Export to GPU
    DOCA_CHECK_RETURN(
        doca_gpu_verbs_qp_flat_list_create_hl(&qpHl_, 1, &qpGpuDev_),
        "Failed to export QP to GPU");

    XLOGF(
        INFO,
        "Rank {}: QP connected to remote QPN={}",
        globalRank,
        remoteQpNum_);

    // Build buffer descriptors with strong types - implicit conversion handles
    // byte order swap
    localDataBuf_ = IbgdaLocalBuffer(dataBuffer_, HostLKey(dataMr_->lkey));
    localSignalBuf_ =
        IbgdaLocalBuffer(signalBuffer_, HostLKey(signalMr_->lkey));

    remoteDataBuf_ = IbgdaRemoteBuffer(
        reinterpret_cast<void*>(remoteDataAddr_), HostRKey(remoteDataRkey_));
    remoteSignalBuf_ = IbgdaRemoteBuffer(
        reinterpret_cast<void*>(remoteSignalAddr_),
        HostRKey(remoteSignalRkey_));

    return true;
  }

  // ---------------------------------------------------------------------------
  // Helper Methods
  // ---------------------------------------------------------------------------

  void syncRanks() {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  void resetSignalBuffers() {
    CUDACHECK_TEST(
        cudaMemset(signalBuffer_, 0, kDefaultNumSignals * sizeof(uint64_t)));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    syncRanks();
  }

 private:
  std::string detectGpuAddr() {
    char pciBusId[32] = {};
    if (cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), localRank) !=
        cudaSuccess) {
      return "";
    }
    return std::string(pciBusId);
  }

  bool allocateBuffers() {
    cudaError_t err;

    err = cudaMalloc(&dataBuffer_, dataBufferSize_);
    if (err != cudaSuccess) {
      XLOGF(
          WARNING,
          "Failed to allocate data buffer: {}",
          cudaGetErrorString(err));
      return false;
    }

    // Allocate signal buffer for multiple signals
    size_t signalBufferSize = kDefaultNumSignals * sizeof(uint64_t);
    err = cudaMalloc(&signalBuffer_, signalBufferSize);
    if (err != cudaSuccess) {
      XLOGF(
          WARNING,
          "Failed to allocate signal buffer: {}",
          cudaGetErrorString(err));
      return false;
    }

    err = cudaMemset(signalBuffer_, 0, signalBufferSize);
    if (err != cudaSuccess) {
      XLOGF(
          WARNING,
          "Failed to memset signal buffer: {}",
          cudaGetErrorString(err));
      return false;
    }

    int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

    dataMr_ = ibv_reg_mr(verbsPd_, dataBuffer_, dataBufferSize_, flags);
    signalMr_ = ibv_reg_mr(verbsPd_, signalBuffer_, signalBufferSize, flags);

    if (!dataMr_ || !signalMr_) {
      XLOG(WARNING) << "Failed to register buffers with RDMA";
      return false;
    }

    XLOGF(
        INFO,
        "Rank {}: Allocated buffers - data={} signal={}",
        globalRank,
        dataBuffer_,
        signalBuffer_);
    return true;
  }

  void cleanup() {
    if (deviceTransport_) {
      freeDeviceTransport(deviceTransport_);
      deviceTransport_ = nullptr;
    }
    if (qpGpuDev_) {
      doca_gpu_verbs_qp_flat_list_destroy_hl(qpGpuDev_);
      qpGpuDev_ = nullptr;
    }
    if (qpHl_) {
      doca_gpu_verbs_destroy_qp_hl(qpHl_);
      qpHl_ = nullptr;
    }
    if (dataMr_) {
      ibv_dereg_mr(dataMr_);
      dataMr_ = nullptr;
    }
    if (signalMr_) {
      ibv_dereg_mr(signalMr_);
      signalMr_ = nullptr;
    }
    if (dataBuffer_) {
      cudaFree(dataBuffer_);
      dataBuffer_ = nullptr;
    }
    if (signalBuffer_) {
      cudaFree(signalBuffer_);
      signalBuffer_ = nullptr;
    }
    if (verbsPd_) {
      ibv_dealloc_pd(verbsPd_);
      verbsPd_ = nullptr;
    }
    if (verbsCtx_) {
      ibv_close_device(verbsCtx_);
      verbsCtx_ = nullptr;
    }
    if (gpuDev_) {
      doca_gpu_destroy(gpuDev_);
      gpuDev_ = nullptr;
    }
  }

 protected:
  // DOCA/GPU resources
  doca_gpu* gpuDev_{nullptr};
  struct ibv_context* verbsCtx_{nullptr};
  struct ibv_pd* verbsPd_{nullptr};

  // QP resources
  doca_gpu_verbs_qp_hl* qpHl_{nullptr};
  doca_gpu_dev_verbs_qp* qpGpuDev_{nullptr};
  uint32_t localQpNum_{0};
  union ibv_gid localGid_{};

  // Remote QP info
  uint32_t remoteQpNum_{0};
  union ibv_gid remoteGid_{};
  uint64_t remoteDataAddr_{0};
  uint64_t remoteSignalAddr_{0};
  uint32_t remoteDataRkey_{0};
  uint32_t remoteSignalRkey_{0};

  // Buffers
  void* dataBuffer_{nullptr};
  void* signalBuffer_{nullptr};
  size_t dataBufferSize_{0};
  struct ibv_mr* dataMr_{nullptr};
  struct ibv_mr* signalMr_{nullptr};

  // Buffer descriptors for device use
  IbgdaLocalBuffer localDataBuf_;
  IbgdaLocalBuffer localSignalBuf_;
  IbgdaRemoteBuffer remoteDataBuf_;
  IbgdaRemoteBuffer remoteSignalBuf_;

  // Device transport
  P2pIbgdaTransportDevice* deviceTransport_{nullptr};
};

// =============================================================================
// Test Cases
// =============================================================================

TEST_F(P2pIbgdaTransportDistributedTest, PutSignalBasic) {
  // Rank 0 sends data to Rank 1 with signal
  // Rank 1 waits for signal, then verifies data

  constexpr int kSignalId = 0;
  constexpr uint64_t kSignalVal = 1;
  constexpr uint8_t kDataPattern = 0xAB;
  const size_t nbytes = dataBufferSize_;

  resetSignalBuffers();

  if (globalRank == 0) {
    // Sender: fill local data and send to peer
    runFillDataKernel(dataBuffer_, nbytes, kDataPattern);
    XLOGF(INFO, "Rank 0: Sending {} bytes with signal", nbytes);

    runPutSignalKernel(
        deviceTransport_,
        localDataBuf_,
        remoteDataBuf_,
        nbytes,
        kSignalId,
        kSignalVal);

    XLOG(INFO) << "Rank 0: put_signal complete";
  } else {
    // Receiver: wait for signal, then verify data
    bool* d_success = nullptr;
    CUDACHECK_TEST(cudaMalloc(&d_success, sizeof(bool)));
    CUDACHECK_TEST(cudaMemset(d_success, 0, sizeof(bool)));

    XLOG(INFO) << "Rank 1: Waiting for signal...";
    runWaitSignalKernel(deviceTransport_, kSignalId, kSignalVal, d_success);
    XLOG(INFO) << "Rank 1: Signal received";

    // Verify data
    runVerifyDataKernel(dataBuffer_, nbytes, kDataPattern, d_success);

    bool success = false;
    CUDACHECK_TEST(
        cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaFree(d_success));
    EXPECT_TRUE(success) << "Rank 1: Data verification failed";
  }

  syncRanks();
}

TEST_F(P2pIbgdaTransportDistributedTest, PutSignalNonAdaptiveBasic) {
  // Test put_signal_non_adaptive - same as PutSignalBasic but uses
  // the fused put_signal operation instead of the split put + signal

  constexpr int kSignalId = 0;
  constexpr uint64_t kSignalVal = 1;
  constexpr uint8_t kDataPattern = 0xCD;
  const size_t nbytes = dataBufferSize_;

  resetSignalBuffers();

  if (globalRank == 0) {
    // Sender: fill local data and send to peer using non-adaptive version
    runFillDataKernel(dataBuffer_, nbytes, kDataPattern);
    XLOGF(
        INFO, "Rank 0: Sending {} bytes with put_signal_non_adaptive", nbytes);

    runPutSignalNonAdaptiveKernel(
        deviceTransport_,
        localDataBuf_,
        remoteDataBuf_,
        nbytes,
        kSignalId,
        kSignalVal);

    XLOG(INFO) << "Rank 0: put_signal_non_adaptive complete";
  } else {
    // Receiver: wait for signal, then verify data
    bool* d_success = nullptr;
    CUDACHECK_TEST(cudaMalloc(&d_success, sizeof(bool)));
    CUDACHECK_TEST(cudaMemset(d_success, 0, sizeof(bool)));

    XLOG(INFO) << "Rank 1: Waiting for signal...";
    runWaitSignalKernel(deviceTransport_, kSignalId, kSignalVal, d_success);
    XLOG(INFO) << "Rank 1: Signal received";

    // Verify data
    runVerifyDataKernel(dataBuffer_, nbytes, kDataPattern, d_success);

    bool success = false;
    CUDACHECK_TEST(
        cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaFree(d_success));
    EXPECT_TRUE(success) << "Rank 1: Data verification failed (non-adaptive)";
  }

  syncRanks();
}

TEST_F(P2pIbgdaTransportDistributedTest, SignalOnly) {
  // Test signal-only operation without data transfer
  // Rank 0 sends signal to Rank 1
  // Rank 1 waits for signal

  constexpr int kSignalId = 1;
  constexpr uint64_t kSignalVal = 42;

  resetSignalBuffers();

  if (globalRank == 0) {
    XLOG(INFO) << "Rank 0: Sending signal only";
    runSignalOnlyKernel(deviceTransport_, kSignalId, kSignalVal);
    XLOG(INFO) << "Rank 0: signal complete";
  } else {
    bool* d_success = nullptr;
    CUDACHECK_TEST(cudaMalloc(&d_success, sizeof(bool)));
    CUDACHECK_TEST(cudaMemset(d_success, 0, sizeof(bool)));

    XLOG(INFO) << "Rank 1: Waiting for signal...";
    runWaitSignalKernel(deviceTransport_, kSignalId, kSignalVal, d_success);

    bool success = false;
    CUDACHECK_TEST(
        cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaFree(d_success));
    EXPECT_TRUE(success) << "Rank 1: wait_signal failed";
    XLOG(INFO) << "Rank 1: Signal received";
  }

  syncRanks();
}

// =============================================================================
// Parameterized Tests for Different Transfer Sizes
// =============================================================================

class P2pIbgdaTransportSizeTest : public P2pIbgdaTransportDistributedTest,
                                  public ::testing::WithParamInterface<size_t> {
};

TEST_P(P2pIbgdaTransportSizeTest, PutSignalVaryingSize) {
  const size_t nbytes = GetParam();

  // Skip if requested size exceeds allocated buffer
  if (nbytes > dataBufferSize_) {
    GTEST_SKIP() << "Requested size " << nbytes << " exceeds buffer size "
                 << dataBufferSize_;
    return;
  }

  constexpr int kSignalId = 2;
  constexpr uint64_t kSignalVal = 1;
  constexpr uint8_t kDataPattern = 0xCD;

  resetSignalBuffers();

  if (globalRank == 0) {
    runFillDataKernel(dataBuffer_, nbytes, kDataPattern);
    XLOGF(INFO, "Rank 0: Sending {} bytes", nbytes);

    runPutSignalKernel(
        deviceTransport_,
        localDataBuf_,
        remoteDataBuf_,
        nbytes,
        kSignalId,
        kSignalVal);
  } else {
    bool* d_success = nullptr;
    CUDACHECK_TEST(cudaMalloc(&d_success, sizeof(bool)));
    CUDACHECK_TEST(cudaMemset(d_success, 0, sizeof(bool)));

    runWaitSignalKernel(deviceTransport_, kSignalId, kSignalVal, d_success);
    runVerifyDataKernel(dataBuffer_, nbytes, kDataPattern, d_success);

    bool success = false;
    CUDACHECK_TEST(
        cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaFree(d_success));
    EXPECT_TRUE(success) << "Rank 1: Data verification failed for size "
                         << nbytes;
  }

  syncRanks();
}

INSTANTIATE_TEST_SUITE_P(
    TransferSizes,
    P2pIbgdaTransportSizeTest,
    ::testing::Values(256, 1024, 4096),
    [](const ::testing::TestParamInfo<size_t>& info) {
      return fmt::format("{}B", info.param);
    });

} // namespace comms::pipes::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto mpi_env = std::make_unique<meta::comms::MPIEnvironmentBase>();
  ::testing::AddGlobalTestEnvironment(mpi_env.get());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
