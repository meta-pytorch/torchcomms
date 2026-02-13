// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime.h>
#include <folly/container/F14Map.h>
#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <gtest/gtest.h>
#include <numeric>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/checks.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ibverbx;
using namespace meta::comms;

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

namespace {
// use broadcom nic for AMD platform, use mellanox nic for NV platform
#if defined(__HIP_PLATFORM_AMD__) && !defined(USE_FE_NIC)
const std::string kNicPrefix("bnxt_re");
#else
const std::string kNicPrefix("mlx5_");
#endif

constexpr uint8_t kPortNum = 1;

#if defined(USE_FE_NIC)
constexpr int kGidIndex = 1;
#else
constexpr int kGidIndex = 3;
#endif

struct BusinessCard {
  enum ibv_mtu mtu { IBV_MTU_4096 };
  uint32_t qpNum{0};
  uint8_t port{0};
  uint64_t subnetPrefixes[2]{0, 0}; // GB200: Support 2 devices with 2 GIDs
  uint64_t interfaceIds[2]{0, 0}; // GB200: Support 2 devices with 2 GIDs
  int32_t rank{-1};
  uint64_t remoteAddr{0};
  uint32_t rkeys[2]{0, 0}; // GB200: Support 2 devices with 2 rkeys
};

std::ostream& operator<<(std::ostream& out, BusinessCard const& card) {
  out << fmt::format(
      "<rank {} qp-num {}, port {}, gids [{:x}/{:x}, {:x}/{:x}] remoteAddr {:x}, rkeys [{:x}, {:x}]>",
      card.rank,
      card.qpNum,
      card.port,
      card.subnetPrefixes[0],
      card.interfaceIds[0],
      card.subnetPrefixes[1],
      card.interfaceIds[1],
      card.remoteAddr,
      card.rkeys[0],
      card.rkeys[1]);
  return out;
}

// helper functions
ibv_qp_init_attr makeIbvQpInitAttr(ibv_cq* cq) {
  ibv_qp_init_attr initAttr{};
  memset(&initAttr, 0, sizeof(ibv_qp_init_attr));
  initAttr.send_cq = cq;
  initAttr.recv_cq = cq;
  initAttr.qp_type = IBV_QPT_RC; // Reliable Connection
  initAttr.sq_sig_all = 0;
  initAttr.cap.max_send_wr = 1024; // maximum outstanding send WRs
  initAttr.cap.max_recv_wr = 1024; // maximum outstanding recv WRs
  initAttr.cap.max_send_sge = 1;
  initAttr.cap.max_recv_sge = 1;
  initAttr.cap.max_inline_data = 0;
  return initAttr;
}

ibv_qp_attr makeQpAttrInit(const BusinessCard& localCard) {
  ibv_qp_attr qpAttr = {
      .qp_state = IBV_QPS_INIT,
      .qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
          IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_REMOTE_WRITE,
      .pkey_index = 0,
      .port_num = localCard.port,
  };
  return qpAttr;
}

ibv_qp_attr makeQpAttrRtr(ibv_gid remoteGid) {
  // The Service Level to be used
  uint8_t kServiceLevel = 0;
  int kTrafficClass = 0;

  ibv_qp_attr qpAttr{};
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));

  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = IBV_MTU_4096;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;

  // assume IBV_LINK_LAYER_ETHERNET
  qpAttr.ah_attr.is_global = 1;
  qpAttr.ah_attr.grh.dgid.global.subnet_prefix = remoteGid.global.subnet_prefix;
  qpAttr.ah_attr.grh.dgid.global.interface_id = remoteGid.global.interface_id;
  qpAttr.ah_attr.grh.flow_label = 0;
  qpAttr.ah_attr.grh.sgid_index = kGidIndex;
  qpAttr.ah_attr.grh.hop_limit = 255;
  qpAttr.ah_attr.grh.traffic_class = kTrafficClass;
  qpAttr.ah_attr.sl = kServiceLevel;
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = kPortNum;
  return qpAttr;
}

ibv_qp_attr makeQpAttrRts() {
  // 4us x 2^20 = 4s
  const uint8_t kTimeout = 20;
  const uint8_t kRetryCnt = 1;

  struct ibv_qp_attr qpAttr{};
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));

  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = kTimeout;
  qpAttr.retry_cnt = kRetryCnt;

  // The value 7 is special and specify to retry infinite times in case of RNR
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  return qpAttr;
}

void changeVirtualQpStateToRts(
    IbvVirtualQp& virtualQp,
    const BusinessCard& localCard,
    const BusinessCard& remoteCard,
    const IbvVirtualQpBusinessCard& remoteVirtualQpBusinessCard) {
  size_t totalQps = virtualQp.getTotalQps();

  // Determine how many devices the remote has based on non-zero GIDs
  size_t numRemoteDevices = 0;
  for (size_t i = 0; i < 2; i++) {
    if (remoteCard.subnetPrefixes[i] != 0 || remoteCard.interfaceIds[i] != 0) {
      numRemoteDevices++;
    }
  }

  if (numRemoteDevices == 0) {
    throw std::runtime_error("Remote card has no valid GIDs");
  }

  // Get references to physical QPs and notify QP
  auto& physicalQps = virtualQp.getQpsRef();
  auto& notifyQp = virtualQp.getNotifyQpRef();

  // Modify all QPs one by one
  for (size_t i = 0; i < totalQps; i++) {
    // Distribute QPs evenly across remote devices using block distribution
    // First half of QPs connect to first remote device, second half to second
    // device
    size_t remoteDeviceIdx = (i * numRemoteDevices) / totalQps;

    // Construct remote GID from BusinessCard
    ibv_gid remoteGid{};
    remoteGid.global.subnet_prefix = remoteCard.subnetPrefixes[remoteDeviceIdx];
    remoteGid.global.interface_id = remoteCard.interfaceIds[remoteDeviceIdx];

    // Change QP state to INIT
    {
      auto qpAttr = makeQpAttrInit(localCard);
      auto result = physicalQps.at(i).modifyQp(
          &qpAttr,
          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
      if (!result) {
        throw std::runtime_error(
            fmt::format("Failed to modify QP {} to INIT state", i));
      }
    }

    // Change QP state to RTR
    {
      auto qpAttr = makeQpAttrRtr(remoteGid);
      qpAttr.dest_qp_num = remoteVirtualQpBusinessCard.qpNums_.at(i);
      auto result = physicalQps.at(i).modifyQp(
          &qpAttr,
          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
              IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
      if (!result) {
        throw std::runtime_error(
            fmt::format("Failed to modify QP {} to RTR state", i));
      }
    }

    // Change QP state to RTS
    {
      auto qpAttr = makeQpAttrRts();
      auto result = physicalQps.at(i).modifyQp(
          &qpAttr,
          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
              IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
      if (!result) {
        throw std::runtime_error(
            fmt::format("Failed to modify QP {} to RTS state", i));
      }
    }
  }

  // Modify notify QP (using the first remote device's GID)
  ibv_gid notifyRemoteGid{};
  notifyRemoteGid.global.subnet_prefix = remoteCard.subnetPrefixes[0];
  notifyRemoteGid.global.interface_id = remoteCard.interfaceIds[0];

  // Change notify QP state to INIT
  {
    auto qpAttr = makeQpAttrInit(localCard);
    auto result = notifyQp.modifyQp(
        &qpAttr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (!result) {
      throw std::runtime_error("Failed to modify notify QP to INIT state");
    }
  }

  // Change notify QP state to RTR
  {
    auto qpAttr = makeQpAttrRtr(notifyRemoteGid);
    qpAttr.dest_qp_num = remoteVirtualQpBusinessCard.notifyQpNum_;
    auto result = notifyQp.modifyQp(
        &qpAttr,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (!result) {
      throw std::runtime_error("Failed to modify notify QP to RTR state");
    }
  }

  // Change notify QP state to RTS
  {
    auto qpAttr = makeQpAttrRts();
    auto result = notifyQp.modifyQp(
        &qpAttr,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
            IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (!result) {
      throw std::runtime_error("Failed to modify notify QP to RTS state");
    }
  }
}

} // namespace

class IbverbxVirtualQpTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    ncclCvarInit();
    ASSERT_TRUE(ibvInit());
  }

  // Common setup class to hold all initialized resources
  class VirtualQpSetup {
   public:
    VirtualQpSetup(
        std::vector<IbvDevice> devices_,
        std::vector<IbvPd> pds_,
        IbvVirtualCq virtualCq_,
        IbvVirtualQp virtualQp_,
        void* devBuf_,
        size_t devBufSize_,
        std::vector<IbvMr> mrs_,
        BusinessCard localCard_,
        BusinessCard remoteCard_,
        IbvVirtualQpBusinessCard remoteVirtualQpBusinessCard_,
        folly::F14FastMap<int32_t, MemoryRegionKeys> deviceIdToKeys_)
        : devices(std::move(devices_)),
          pds(std::move(pds_)),
          virtualCq(std::move(virtualCq_)),
          virtualQp(std::move(virtualQp_)),
          devBuf(devBuf_),
          devBufSize(devBufSize_),
          mrs(std::move(mrs_)),
          localCard(localCard_),
          remoteCard(remoteCard_),
          remoteVirtualQpBusinessCard(std::move(remoteVirtualQpBusinessCard_)),
          deviceIdToKeys(std::move(deviceIdToKeys_)) {}

    std::vector<IbvDevice> devices;
    std::vector<IbvPd> pds;
    IbvVirtualCq virtualCq;
    IbvVirtualQp virtualQp;
    void* devBuf;
    size_t devBufSize;
    std::vector<IbvMr> mrs;
    BusinessCard localCard;
    BusinessCard remoteCard;
    IbvVirtualQpBusinessCard remoteVirtualQpBusinessCard;
    folly::F14FastMap<int32_t, MemoryRegionKeys> deviceIdToKeys;
  };

  // Common setup function for parameterized tests
  template <typename T>
  VirtualQpSetup setupVirtualQp(
      int devBufSize,
      int numQp,
      int maxMsgPerQp = -1,
      int maxMsgBytes = -1,
      LoadBalancingScheme loadBalancingScheme = LoadBalancingScheme::SPRAY) {
    CUDA_CHECK(cudaSetDevice(localRank));

    int myDevId{-1};
    CUDA_CHECK(cudaGetDevice(&myDevId));

    // get device
    auto maybeDevices =
        IbvDevice::ibvGetDeviceList(NCCL_IB_HCA, NCCL_IB_HCA_PREFIX);
    EXPECT_TRUE(maybeDevices);
    auto devices = std::move(*maybeDevices);

    // For GB200: select 2 devices (device pair) per GPU
    // Assuming devices are arranged in pairs: [dev0, dev1], [dev2, dev3], ...
    std::vector<IbvDevice> selectedDevices;
    if (myDevId * 2 + 1 < static_cast<int>(devices.size())) {
      selectedDevices.push_back(std::move(devices.at(myDevId * 2)));
      selectedDevices.push_back(std::move(devices.at(myDevId * 2 + 1)));
    } else {
      // Fallback: use single device if pair not available
      selectedDevices.push_back(std::move(devices.at(myDevId)));
    }

    // Create one PD per device
    std::vector<IbvPd> pds;
    pds.reserve(selectedDevices.size());
    for (auto& device : selectedDevices) {
      auto maybePd = device.allocPd();
      EXPECT_TRUE(maybePd);
      pds.push_back(std::move(*maybePd));
    }

    // Create one CQ per device and wrap in IbvVirtualCq
    int cqe = 2 * numQp * maxMsgPerQp;
    std::vector<IbvCq> cqList;
    cqList.reserve(selectedDevices.size());
    for (auto& device : selectedDevices) {
      auto maybeCq = device.createCq(cqe, nullptr, nullptr, 0);
      EXPECT_TRUE(maybeCq);
      cqList.push_back(std::move(*maybeCq));
    }
    auto virtualCq = IbvVirtualCq(std::move(cqList), cqe);

    // Create physical QPs distributed across PDs and CQs
    uint32_t totalQps = numQp;
    std::vector<IbvQp> qps;
    qps.reserve(totalQps);

    size_t numPhysicalCqs = virtualCq.getPhysicalCqsRef().size();
    size_t numPds = pds.size();

    for (uint32_t i = 0; i < totalQps; i++) {
      // Distribute QPs evenly across PDs and physical CQs
      // First half of QPs use first PD/CQ, second half use second PD/CQ
      size_t pdIdx = (i * numPds) / totalQps;
      size_t cqIdx = (i * numPhysicalCqs) / totalQps;

      auto initAttr =
          makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(cqIdx).cq());
      auto maybeQp = pds.at(pdIdx).createQp(&initAttr);
      EXPECT_TRUE(maybeQp);
      qps.push_back(std::move(*maybeQp));
    }

    // Create notify QP using the first PD and first physical CQ
    auto notifyInitAttr =
        makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
    auto maybeNotifyQp = pds.at(0).createQp(&notifyInitAttr);
    EXPECT_TRUE(maybeNotifyQp);

    // Create IbvVirtualQp from vector of QPs
    auto virtualQp = IbvVirtualQp(
        std::move(qps),
        &virtualCq,
        maxMsgPerQp,
        maxMsgBytes,
        loadBalancingScheme,
        std::move(*maybeNotifyQp));

    // init device buffer
    void* devBuf{nullptr};
    CUDA_CHECK(cudaMalloc(&devBuf, devBufSize));

    // Initialize host buffer and copy to device
    size_t numElements = devBufSize / sizeof(T);
    std::vector<T> hostBuf(numElements);

    if (globalRank == 0) {
      // receiver: fill up with 0s
      std::fill(hostBuf.begin(), hostBuf.end(), T{});
    } else if (globalRank == 1) {
      // sender/writer: initialize with sequence number (1, 2, 3, ...)
      if constexpr (std::is_integral_v<T>) {
        std::iota(hostBuf.begin(), hostBuf.end(), T(1));
      } else {
        // For floating point types, use incremental values
        T val = T(1);
        for (auto& elem : hostBuf) {
          elem = val;
          val += T(1);
        }
      }
    }
    CUDA_CHECK(
        cudaMemcpy(devBuf, hostBuf.data(), devBufSize, cudaMemcpyDefault));
    CUDA_CHECK(cudaDeviceSynchronize());

    // register memory region with all PDs
    ibv_access_flags access = static_cast<ibv_access_flags>(
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ);
    std::vector<IbvMr> mrs;
    mrs.reserve(pds.size());
    for (size_t i = 0; i < pds.size(); i++) {
      auto maybeMr = pds.at(i).regMr(devBuf, devBufSize, access);
      EXPECT_TRUE(maybeMr);
      mrs.push_back(std::move(*maybeMr));
    }

    // create local business card and exchange
    // Query GIDs from both devices and collect rkeys from all MRs
    BusinessCard localCard = {
        .mtu = IBV_MTU_4096,
        .port = kPortNum,
        .subnetPrefixes = {0, 0},
        .interfaceIds = {0, 0},
        .rank = globalRank,
        .remoteAddr = reinterpret_cast<uint64_t>(devBuf),
        .rkeys = {0, 0},
    };

    // Populate GIDs and rkeys for each device
    for (size_t i = 0; i < selectedDevices.size() && i < 2; i++) {
      auto gid = selectedDevices.at(i).queryGid(kPortNum, kGidIndex);
      EXPECT_TRUE(gid);
      localCard.subnetPrefixes[i] = gid->global.subnet_prefix;
      localCard.interfaceIds[i] = gid->global.interface_id;
      if (i < mrs.size()) {
        localCard.rkeys[i] = mrs.at(i).mr()->rkey;
      }
    }
    std::vector<BusinessCard> cards(numRanks);
    MPI_CHECK(MPI_Allgather(
        &localCard,
        sizeof(BusinessCard),
        MPI_BYTE,
        cards.data(),
        sizeof(BusinessCard),
        MPI_BYTE,
        MPI_COMM_WORLD));
    for (int i = 0; i < numRanks; ++i) {
      const auto& card = cards.at(i);
      XLOG(DBG1) << "rank " << globalRank << ": got card " << card;
    }
    const auto& remoteCard = globalRank == 0 ? cards.at(1) : cards.at(0);

    // Get the business card and serialize it to JSON
    std::string serializedCard =
        virtualQp.getVirtualQpBusinessCard().serialize();

    // Since all hosts have the same number of QPs, the serialized string size
    // should be consistent Use the local string size directly
    size_t bufferSize = serializedCard.size();

    // Gather all serialized cards
    std::vector<char> allSerializedCards(bufferSize * numRanks);
    MPI_CHECK(MPI_Allgather(
        serializedCard.data(),
        bufferSize,
        MPI_CHAR,
        allSerializedCards.data(),
        bufferSize,
        MPI_CHAR,
        MPI_COMM_WORLD));

    // Extract remote card's serialized string
    std::string remoteSerializedCard(
        allSerializedCards.data() + (globalRank == 0 ? bufferSize : 0),
        bufferSize);

    auto maybeRemoteVirtualQpBusinessCard =
        IbvVirtualQpBusinessCard::deserialize(remoteSerializedCard);
    EXPECT_TRUE(maybeRemoteVirtualQpBusinessCard);
    auto remoteVirtualQpBusinessCard =
        std::move(*maybeRemoteVirtualQpBusinessCard);

    // init qp group
    changeVirtualQpStateToRts(
        virtualQp, localCard, remoteCard, remoteVirtualQpBusinessCard);

    // Construct deviceIdToKeys map for GB200
    // This maps device ID to the lkey/rkey pairs used for RDMA operations
    folly::F14FastMap<int32_t, MemoryRegionKeys> deviceIdToKeys;
    for (size_t i = 0; i < selectedDevices.size(); i++) {
      int32_t deviceId = selectedDevices.at(i).getDeviceId();
      deviceIdToKeys[deviceId] = MemoryRegionKeys{
          .lkey = mrs.at(i).mr()->lkey, .rkey = remoteCard.rkeys[i]};
    }

    return VirtualQpSetup(
        std::move(selectedDevices),
        std::move(pds),
        std::move(virtualCq),
        std::move(virtualQp),
        devBuf,
        static_cast<size_t>(devBufSize),
        std::move(mrs),
        localCard,
        remoteCard,
        std::move(remoteVirtualQpBusinessCard),
        std::move(deviceIdToKeys));
  }
};

// Enum for different data types
enum class DataType { INT8, INT16, INT32, INT64, FLOAT, DOUBLE };

// Helper function to convert DataType enum to string (shared by both fixtures)
std::string dataTypeToString(DataType dataType) {
  switch (dataType) {
    case DataType::INT8:
      return "INT8";
    case DataType::INT16:
      return "INT16";
    case DataType::INT32:
      return "INT32";
    case DataType::INT64:
      return "INT64";
    case DataType::FLOAT:
      return "FLOAT";
    case DataType::DOUBLE:
      return "DOUBLE";
    default:
      return "UNKNOWN";
  }
}

// Helper function to get size of data type enum (shared by both fixtures)
size_t getDataTypeSize(DataType dataType) {
  switch (dataType) {
    case DataType::INT8:
      return sizeof(int8_t);
    case DataType::INT16:
      return sizeof(int16_t);
    case DataType::INT32:
      return sizeof(int32_t);
    case DataType::INT64:
      return sizeof(int64_t);
    case DataType::FLOAT:
      return sizeof(float);
    case DataType::DOUBLE:
      return sizeof(double);
    default:
      return sizeof(int64_t);
  }
}

// Parameterized test class for virtual QP RDMA write tests
class IbverbxVirtualQpRdmaWriteTestFixture
    : public IbverbxVirtualQpTestFixture,
      public ::testing::WithParamInterface<
          std::tuple<int, DataType, int, int, int, LoadBalancingScheme>> {
 public:
  // Parameterized test name generator function for virtual QP RDMA write tests
  static std::string getTestName(
      const testing::TestParamInfo<ParamType>& info) {
    std::string baseName = fmt::format(
        "{}_devBufSize_{}_dataType_{}_numQp_",
        std::get<0>(info.param),
        dataTypeToString(std::get<1>(info.param)),
        std::get<2>(info.param));

    // Always include both maxMsgPerQp and maxMsgBytes values to avoid
    // duplicates
    std::string maxMsgPerQpStr = std::get<3>(info.param) > 0
        ? std::to_string(std::get<3>(info.param))
        : "nolimit";
    std::string maxMsgBytesStr = std::get<4>(info.param) > 0
        ? std::to_string(std::get<4>(info.param))
        : "nolimit";

    std::string loadBalancingStr =
        std::get<5>(info.param) == LoadBalancingScheme::DQPLB ? "DQPLB"
                                                              : "SPRAY";

    baseName += fmt::format(
        "{}_maxMsgPerQp_{}_maxMsgBytes_{}_scheme",
        maxMsgPerQpStr,
        maxMsgBytesStr,
        loadBalancingStr);
    return baseName;
  }

 protected:
  void SetUp() override {
    IbverbxVirtualQpTestFixture::SetUp();
  }

  // Helper template function to run virtual QP RDMA write test with specific
  // data type
  template <typename T>
  void runRdmaWriteVirtualQpTest(
      int devBufSize,
      int numQp,
      int maxMsgPerQp = -1,
      int maxMsgBytes = -1,
      LoadBalancingScheme loadBalancingScheme = LoadBalancingScheme::SPRAY);
};

// Parameterized test class for virtual QP RDMA read tests
class IbverbxVirtualQpRdmaReadTestFixture
    : public IbverbxVirtualQpTestFixture,
      public ::testing::WithParamInterface<
          std::tuple<int, DataType, int, int, int, LoadBalancingScheme>> {
 public:
  // Parameterized test name generator function for virtual QP RDMA read tests
  static std::string getTestName(
      const testing::TestParamInfo<ParamType>& info) {
    std::string baseName = fmt::format(
        "{}_devBufSize_{}_dataType_{}_numQp_",
        std::get<0>(info.param),
        dataTypeToString(std::get<1>(info.param)),
        std::get<2>(info.param));

    // Always include both maxMsgPerQp and maxMsgBytes values to avoid
    // duplicates
    std::string maxMsgPerQpStr = std::get<3>(info.param) > 0
        ? std::to_string(std::get<3>(info.param))
        : "nolimit";
    std::string maxMsgBytesStr = std::get<4>(info.param) > 0
        ? std::to_string(std::get<4>(info.param))
        : "nolimit";

    std::string loadBalancingStr =
        std::get<5>(info.param) == LoadBalancingScheme::DQPLB ? "DQPLB"
                                                              : "SPRAY";

    baseName += fmt::format(
        "{}_maxMsgPerQp_{}_maxMsgBytes_{}_scheme",
        maxMsgPerQpStr,
        maxMsgBytesStr,
        loadBalancingStr);
    return baseName;
  }

 protected:
  void SetUp() override {
    IbverbxVirtualQpTestFixture::SetUp();
  }

  // Helper template function to run virtual QP RDMA read test with specific
  // data type
  template <typename T>
  void runRdmaReadVirtualQpTest(
      int devBufSize,
      int numQp,
      int maxMsgPerQp = -1,
      int maxMsgBytes = -1,
      LoadBalancingScheme loadBalancingScheme = LoadBalancingScheme::SPRAY);
};

// Parameterized test class for virtual QP send/recv tests
class IbverbxVirtualQpSendRecvTestFixture
    : public IbverbxVirtualQpTestFixture,
      public ::testing::WithParamInterface<
          std::tuple<int, DataType, int, int, int>> {
 public:
  // Parameterized test name generator function for virtual QP send/recv tests
  static std::string getTestName(
      const testing::TestParamInfo<ParamType>& info) {
    std::string baseName = fmt::format(
        "{}_devBufSize_{}_dataType_{}_numQp_",
        std::get<0>(info.param),
        dataTypeToString(std::get<1>(info.param)),
        std::get<2>(info.param));

    // Always include both maxMsgPerQp and maxMsgBytes values to avoid
    // duplicates
    std::string maxMsgPerQpStr = std::get<3>(info.param) > 0
        ? std::to_string(std::get<3>(info.param))
        : "nolimit";
    std::string maxMsgBytesStr = std::get<4>(info.param) > 0
        ? std::to_string(std::get<4>(info.param))
        : "nolimit";

    baseName += fmt::format(
        "{}_maxMsgPerQp_{}_maxMsgBytes", maxMsgPerQpStr, maxMsgBytesStr);
    return baseName;
  }

 protected:
  void SetUp() override {
    IbverbxVirtualQpTestFixture::SetUp();
  }

  // Helper template function to run virtual QP send/recv test with specific
  // data type
  template <typename T>
  void runSendRecvVirtualQpTest(
      int devBufSize,
      int numQp,
      int maxMsgPerQp = -1,
      int maxMsgBytes = -1);
};

// Template helper function implementation for Virtual QP RDMA Write
template <typename T>
void IbverbxVirtualQpRdmaWriteTestFixture::runRdmaWriteVirtualQpTest(
    int devBufSize,
    int numQp,
    int maxMsgPerQp,
    int maxMsgBytes,
    LoadBalancingScheme loadBalancingScheme) {
  // Use common setup function to initialize IB resources and devBuf
  auto setup = setupVirtualQp<T>(
      devBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);

  // post send/recv and poll cq
  int wr_id = 0;
  int imm_data = 16384;
  if (globalRank == 0) {
    // receiver

    // post a dummy IbvVirtualRecvWr as this is one-sided comm
    IbvVirtualRecvWr recvWr;
    recvWr.wrId = wr_id;
    ASSERT_TRUE(setup.virtualQp.postRecv_v2(recvWr));
  } else if (globalRank == 1) {
    // writer

    IbvVirtualSendWr sendWr;
    sendWr.wrId = wr_id;
    sendWr.localAddr = setup.devBuf;
    sendWr.length = static_cast<uint32_t>(devBufSize);
    sendWr.remoteAddr = setup.remoteCard.remoteAddr;
    sendWr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    sendWr.sendFlags = IBV_SEND_SIGNALED;
    sendWr.immData = imm_data;
    sendWr.deviceKeys = setup.deviceIdToKeys;
    ASSERT_TRUE(setup.virtualQp.postSend_v2(sendWr));
  }

  // poll cq and check cq
  bool stop = false;
  while (!stop) {
    auto maybeWcsVector = setup.virtualCq.pollCq_v2();
    ASSERT_TRUE(maybeWcsVector);
    auto numWc = maybeWcsVector->size();
    ASSERT_GE(numWc, 0);
    if (numWc == 0) {
      // CQ empty, sleep and retry
      XLOGF(WARN, "rank {}: cq empty, retry in 500ms", globalRank);
      /* sleep override */
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }

    ASSERT_EQ(numWc, 1);

    // got a WC
    const auto wc = maybeWcsVector->at(0);
    ASSERT_EQ(wc.wrId, wr_id);
    ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
    // NOTE: IbvVirtualWc does not propagate immData from physical
    // completions. The IMM field is consumed internally by the VirtualQp
    // layer (SPRAY uses it for notify signaling, DQPLB uses it for
    // sequence tracking). User-level immData is not forwarded through
    // the virtual completion path.
    XLOGF(DBG1, "Rank {} got a wc: wr_id {}", globalRank, wc.wrId);
    stop = true;
  }

  // receiver check data
  if (globalRank == 0) {
    // check data
    size_t numElements = devBufSize / sizeof(T);
    std::vector<T> hostExpectedBuf(numElements);
    if constexpr (std::is_integral_v<T>) {
      std::iota(hostExpectedBuf.begin(), hostExpectedBuf.end(), T(1));
    } else {
      // For floating point types, use incremental values
      T val = T(1);
      for (auto& elem : hostExpectedBuf) {
        elem = val;
        val += T(1);
      }
    }

    std::vector<T> hostRecvBuf(numElements);
    CUDA_CHECK(cudaMemcpy(
        hostRecvBuf.data(), setup.devBuf, devBufSize, cudaMemcpyDefault));
    CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_EQ(hostExpectedBuf, hostRecvBuf);
  }
  XLOGF(DBG1, "rank {} RDMA-WRITE OK", globalRank);

  // Clean up device buffer
  CUDA_CHECK(cudaFree(setup.devBuf));
}

// Template helper function implementation for Virtual QP RDMA Read
template <typename T>
void IbverbxVirtualQpRdmaReadTestFixture::runRdmaReadVirtualQpTest(
    int devBufSize,
    int numQp,
    int maxMsgPerQp,
    int maxMsgBytes,
    LoadBalancingScheme loadBalancingScheme) {
  // Use common setup function to initialize IB resources and devBuf
  auto setup = setupVirtualQp<T>(
      devBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);

  // RDMA Read is a one-sided operation, so both ranks must be synchronized
  // before proceeding. Typically, a separate control QP is used for
  // coordination, but in this case, we rely on MPI setup. We use MPI_Barrier to
  // ensure both ranks are ready before continuing, since a control QP is not
  // available.
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Post a send operation and poll the completion queue (CQ).
  // Note: RDMA READ is a one-sided operation, meaning only the receiver
  // (the side initiating the read from the remote memory) is actively involved.
  // The remote side does not participate in the postSend and pollCq operations.
  int wr_id = 0;
  if (globalRank == 0) {
    // reader - does the work in RDMA Read

    IbvVirtualSendWr sendWr;
    sendWr.wrId = wr_id;
    sendWr.localAddr = setup.devBuf;
    sendWr.length = static_cast<uint32_t>(devBufSize);
    sendWr.remoteAddr = setup.remoteCard.remoteAddr;
    sendWr.opcode = IBV_WR_RDMA_READ;
    sendWr.sendFlags = IBV_SEND_SIGNALED;
    sendWr.deviceKeys = setup.deviceIdToKeys;
    ASSERT_TRUE(setup.virtualQp.postSend_v2(sendWr));

    // poll cq and check cq
    while (true) {
      auto maybeWcsVector = setup.virtualCq.pollCq_v2();
      ASSERT_TRUE(maybeWcsVector);
      auto numWc = maybeWcsVector->size();
      ASSERT_GE(numWc, 0);
      if (numWc == 0) {
        // CQ empty, sleep and retry
        XLOGF(WARN, "rank {}: cq empty, retry in 500ms", globalRank);
        /* sleep override */
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        continue;
      }

      ASSERT_EQ(numWc, 1);

      // got a WC
      const auto wc = maybeWcsVector->at(0);
      ASSERT_EQ(wc.wrId, wr_id);
      ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
      XLOGF(DBG1, "Rank {} got a wc: wr_id {}", globalRank, wc.wrId);
      break;
    }

    // check data
    size_t numElements = devBufSize / sizeof(T);
    std::vector<T> hostExpectedBuf(numElements);
    if constexpr (std::is_integral_v<T>) {
      std::iota(hostExpectedBuf.begin(), hostExpectedBuf.end(), T(1));
    } else {
      // For floating point types, use incremental values
      T val = T(1);
      for (auto& elem : hostExpectedBuf) {
        elem = val;
        val += T(1);
      }
    }

    std::vector<T> hostRecvBuf(numElements);
    CUDA_CHECK(cudaMemcpy(
        hostRecvBuf.data(), setup.devBuf, devBufSize, cudaMemcpyDefault));
    CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_EQ(hostExpectedBuf, hostRecvBuf);
  }

  // RDMA Read is a one-sided operation, so both ranks must be synchronized
  // before finishing the test, otherwise, the IB resources on one rank might
  // get lost. Typically, a separate control QP is used for coordination, but in
  // this case, we rely on MPI setup. We use MPI_Barrier to ensure both ranks
  // are ready before continuing, since a control QP is not available.
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  XLOGF(DBG1, "rank {} RDMA-READ OK", globalRank);

  // Clean up device buffer
  CUDA_CHECK(cudaFree(setup.devBuf));
}

// Template helper function implementation for Virtual QP Send/Recv
template <typename T>
void IbverbxVirtualQpSendRecvTestFixture::runSendRecvVirtualQpTest(
    int devBufSize,
    int numQp,
    int maxMsgPerQp,
    int maxMsgBytes) {
  // Use common setup function to initialize IB resources and devBuf
  auto setup = setupVirtualQp<T>(devBufSize, numQp, maxMsgPerQp, maxMsgBytes);

  // post send/recv within each virtual qp and poll cq
  int wr_id = 0;
  if (globalRank == 0) {
    // receiver
    IbvVirtualRecvWr recvWr;
    recvWr.wrId = wr_id;
    recvWr.localAddr = setup.devBuf;
    recvWr.length = static_cast<uint32_t>(devBufSize);
    recvWr.deviceKeys = setup.deviceIdToKeys;
    ASSERT_TRUE(setup.virtualQp.postRecv_v2(recvWr));
  } else if (globalRank == 1) {
    // sender
    IbvVirtualSendWr sendWr;
    sendWr.wrId = wr_id;
    sendWr.localAddr = setup.devBuf;
    sendWr.length = static_cast<uint32_t>(devBufSize);
    sendWr.opcode = IBV_WR_SEND;
    sendWr.sendFlags = IBV_SEND_SIGNALED;
    sendWr.deviceKeys = setup.deviceIdToKeys;
    ASSERT_TRUE(setup.virtualQp.postSend_v2(sendWr));
  }

  // poll cq and check cq
  bool stop = false;
  while (!stop) {
    auto maybeWcsVector = setup.virtualCq.pollCq_v2();
    ASSERT_TRUE(maybeWcsVector);
    auto numWc = maybeWcsVector->size();
    ASSERT_GE(numWc, 0);
    if (numWc == 0) {
      // CQ empty, sleep and retry
      XLOGF(WARN, "rank {}: cq empty, retry in 500ms", globalRank);
      /* sleep override */
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }

    ASSERT_EQ(numWc, 1);

    // got a WC
    const auto wc = maybeWcsVector->at(0);
    ASSERT_EQ(wc.wrId, wr_id);
    ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
    if (globalRank == 0) {
      // According to ibverblib, this value is relevant for the Receive Queue
      // when handling incoming Send or RDMA Write with immediate operations.
      // Note that this value excludes the length of any immediate data present.
      // For the Send Queue, this value applies to RDMA Read and Atomic
      // operations. Therefore, only check the receiver's work completion (wc)
      // byte_len here.
      ASSERT_EQ(wc.byteLen, devBufSize);
    }
    XLOGF(DBG1, "Rank {} got a wc: wr_id {}", globalRank, wc.wrId);
    stop = true;
  }

  // receiver check data
  if (globalRank == 0) {
    // check data
    size_t numElements = devBufSize / sizeof(T);
    std::vector<T> hostExpectedBuf(numElements);
    if constexpr (std::is_integral_v<T>) {
      std::iota(hostExpectedBuf.begin(), hostExpectedBuf.end(), T(1));
    } else {
      // For floating point types, use incremental values
      T val = T(1);
      for (auto& elem : hostExpectedBuf) {
        elem = val;
        val += T(1);
      }
    }

    std::vector<T> hostRecvBuf(numElements);
    CUDA_CHECK(cudaMemcpy(
        hostRecvBuf.data(), setup.devBuf, devBufSize, cudaMemcpyDefault));
    CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_EQ(hostExpectedBuf, hostRecvBuf);
  }
  XLOGF(DBG1, "rank {} send/recv OK", globalRank);

  // Clean up device buffer
  CUDA_CHECK(cudaFree(setup.devBuf));
}

// RDMA Read Virtual QP test using template helper
TEST_P(IbverbxVirtualQpRdmaReadTestFixture, RdmaReadVirtualQpWithParam) {
  const auto& [devBufSize, dataType, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme] =
      GetParam();

  // Dispatch to the appropriate template function based on data type
  switch (dataType) {
    case DataType::INT8:
      runRdmaReadVirtualQpTest<int8_t>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
    case DataType::INT16:
      runRdmaReadVirtualQpTest<int16_t>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
    case DataType::INT32:
      runRdmaReadVirtualQpTest<int32_t>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
    case DataType::INT64:
      runRdmaReadVirtualQpTest<int64_t>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
    case DataType::FLOAT:
      runRdmaReadVirtualQpTest<float>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
    case DataType::DOUBLE:
      runRdmaReadVirtualQpTest<double>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
  }
}

// RDMA Write Virtual QP test using template helper
TEST_P(IbverbxVirtualQpRdmaWriteTestFixture, RdmaWriteVirtualQpWithParam) {
  const auto& [devBufSize, dataType, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme] =
      GetParam();

  // Dispatch to the appropriate template function based on data type
  switch (dataType) {
    case DataType::INT8:
      runRdmaWriteVirtualQpTest<int8_t>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
    case DataType::INT16:
      runRdmaWriteVirtualQpTest<int16_t>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
    case DataType::INT32:
      runRdmaWriteVirtualQpTest<int32_t>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
    case DataType::INT64:
      runRdmaWriteVirtualQpTest<int64_t>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
    case DataType::FLOAT:
      runRdmaWriteVirtualQpTest<float>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
    case DataType::DOUBLE:
      runRdmaWriteVirtualQpTest<double>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
  }
}

// Send/Recv Virtual QP test using template helper
TEST_P(IbverbxVirtualQpSendRecvTestFixture, SendRecvVirtualQpWithParam) {
  const auto& [devBufSize, dataType, numQp, maxMsgPerQp, maxMsgBytes] =
      GetParam();

  // Dispatch to the appropriate template function based on data type
  switch (dataType) {
    case DataType::INT8:
      runSendRecvVirtualQpTest<int8_t>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes);
      break;
    case DataType::INT16:
      runSendRecvVirtualQpTest<int16_t>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes);
      break;
    case DataType::INT32:
      runSendRecvVirtualQpTest<int32_t>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes);
      break;
    case DataType::INT64:
      runSendRecvVirtualQpTest<int64_t>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes);
      break;
    case DataType::FLOAT:
      runSendRecvVirtualQpTest<float>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes);
      break;
    case DataType::DOUBLE:
      runSendRecvVirtualQpTest<double>(
          devBufSize, numQp, maxMsgPerQp, maxMsgBytes);
      break;
  }
}

// Instantiate Virtual QP Rdma Read test with different buffer sizes, data
// Small buffer configurations - 1KB and 8KB
INSTANTIATE_TEST_SUITE_P(
    VirtualQpRdmaReadTestSmallBuffer,
    IbverbxVirtualQpRdmaReadTestFixture,
    ::testing::Combine(
        testing::Values(1024, 8192), // Small buffer sizes: 1KB, 8KB
        testing::Values(DataType::INT8, DataType::INT32, DataType::FLOAT),
        testing::Values(1, 4), // QP numbers: 1, 4
        testing::Values(64, 128), // maxMsgPerQp: 64, 128
        testing::Values(128, 256), // maxMsgBytes: 128, 256
        testing::Values(
            LoadBalancingScheme::SPRAY)), // LoadBalancingScheme: Spray and
                                          // DQPLB should provide same behavior
                                          // for RDMA READ since there's no
                                          // notification in RDMA READ
    IbverbxVirtualQpRdmaReadTestFixture::getTestName);

// Medium buffer configurations - 1MB and 8MB
INSTANTIATE_TEST_SUITE_P(
    VirtualQpRdmaReadTestMediumBuffer,
    IbverbxVirtualQpRdmaReadTestFixture,
    ::testing::Combine(
        testing::Values(
            1048576,
            8388608), // Medium buffer sizes: 1MB, 8MB
        testing::Values(DataType::INT8, DataType::INT32, DataType::FLOAT),
        testing::Values(16, 128), // QP numbers: 16, 128
        testing::Values(128, 1024), // maxMsgPerQp: 128, 1024
        testing::Values(1024, 16384), // maxMsgBytes: 1024, 16384
        testing::Values(
            LoadBalancingScheme::SPRAY)), // LoadBalancingScheme: Spray and
                                          // DQPLB should provide same behavior
                                          // for RDMA READ since there's no
                                          // notification in RDMA READ
    IbverbxVirtualQpRdmaReadTestFixture::getTestName);

// Large buffer configurations - 1GB
INSTANTIATE_TEST_SUITE_P(
    VirtualQpRdmaReadTestLargeBuffer,
    IbverbxVirtualQpRdmaReadTestFixture,
    ::testing::Combine(
        testing::Values(1073741824), // Large buffer size: 1GB
        testing::Values(DataType::INT8, DataType::INT32, DataType::FLOAT),
        testing::Values(16, 128), // High QP number for maximum parallelism
        testing::Values(128, 1024), // maxMsgPerQp: 128, 1024
        testing::Values(16384, 1048576), // maxMsgBytes: 16KB, 1MB
        testing::Values(
            LoadBalancingScheme::SPRAY)), // LoadBalancingScheme: Spray and
                                          // DQPLB should provide same behavior
                                          // for RDMA READ since there's no
                                          // notification in RDMA READ
    IbverbxVirtualQpRdmaReadTestFixture::getTestName);

// Large buffer configurations - 1GB with DQPLB mode
INSTANTIATE_TEST_SUITE_P(
    VirtualQpRdmaReadTestLargeBufferDqplb,
    IbverbxVirtualQpRdmaReadTestFixture,
    ::testing::Combine(
        testing::Values(1073741824), // Large buffer size: 1GB
        testing::Values(DataType::INT8),
        testing::Values(128), // High QP number for maximum parallelism
        testing::Values(1024), // maxMsgPerQp: 128, 1024
        testing::Values(1048576), // maxMsgBytes: 16KB, 1MB
        testing::Values(
            LoadBalancingScheme::DQPLB)), // check DQPLB scheme is providing
                                          // same behavior as Spray
    IbverbxVirtualQpRdmaReadTestFixture::getTestName);

// Instantiate Virtual QP Rdma Write test with different buffer sizes, data
// Small buffer configurations - 1KB and 8KB
INSTANTIATE_TEST_SUITE_P(
    VirtualQpRdmaWriteTestSmallBuffer,
    IbverbxVirtualQpRdmaWriteTestFixture,
    ::testing::Combine(
        testing::Values(1024, 8192), // Small buffer sizes: 1KB, 8KB
        testing::Values(DataType::INT8, DataType::INT32, DataType::FLOAT),
        testing::Values(1, 4), // QP numbers: 1, 4
        testing::Values(64, 128), // maxMsgPerQp: 64, 128
        testing::Values(128, 256), // maxMsgBytes: 128, 256
        testing::Values(
            LoadBalancingScheme::DQPLB,
            LoadBalancingScheme::SPRAY)), // LoadBalancingScheme
    IbverbxVirtualQpRdmaWriteTestFixture::getTestName);

// Medium buffer configurations - 1MB and 8MB
INSTANTIATE_TEST_SUITE_P(
    VirtualQpRdmaWriteTestMediumBuffer,
    IbverbxVirtualQpRdmaWriteTestFixture,
    ::testing::Combine(
        testing::Values(
            1048576,
            8388608), // Medium buffer sizes: 1MB, 8MB
        testing::Values(DataType::INT8, DataType::INT32, DataType::FLOAT),
        testing::Values(16, 128), // QP numbers: 16, 128
        testing::Values(128, 1024), // maxMsgPerQp: 128, 1024
        testing::Values(1024, 16384), // maxMsgBytes: 1024, 16384
        testing::Values(
            LoadBalancingScheme::DQPLB,
            LoadBalancingScheme::SPRAY)), // LoadBalancingScheme
    IbverbxVirtualQpRdmaWriteTestFixture::getTestName);

// Large buffer configurations - 1GB
INSTANTIATE_TEST_SUITE_P(
    VirtualQpRdmaWriteTestLargeBuffer,
    IbverbxVirtualQpRdmaWriteTestFixture,
    ::testing::Combine(
        testing::Values(1073741824), // Large buffer size: 1GB
        testing::Values(DataType::INT8, DataType::INT32, DataType::FLOAT),
        testing::Values(16, 128), // High QP number for maximum parallelism
        testing::Values(128, 1024), // maxMsgPerQp: 128, 1024
        testing::Values(16384, 1048576), // maxMsgBytes: 16KB, 1MB
        testing::Values(
            LoadBalancingScheme::DQPLB,
            LoadBalancingScheme::SPRAY)), // LoadBalancingScheme
    IbverbxVirtualQpRdmaWriteTestFixture::getTestName);

// Instantiate Virtual QP Send Recv test with different buffer sizes, data
// We use different buffer sizes for the send/recv unit tests compared to the
// RDMA Write tests. This is because send/recv currently operates with a single
// QP, making large buffer tests (up to 1GB) very slow. To mitigate this, we
// limit the number of large buffer test cases in send/recv, whereas RDMA Write
// tests include more extensive coverage. Additionally, in practical use cases,
// RDMA Write is typically used to transfer large chunks of data, while
// send/recv is mainly used for exchanging fixed-size metadata. Therefore, it
// makes sense to focus large buffer testing on RDMA Write, and keep send/recv
// tests limited to a few representative large buffer cases.

// Small buffer configurations for Send/Recv - 1KB and 8KB
INSTANTIATE_TEST_SUITE_P(
    VirtualQpSendRecvTestSmallBuffer,
    IbverbxVirtualQpSendRecvTestFixture,
    ::testing::Combine(
        testing::Values(1024, 8192), // Small buffer sizes: 1KB, 8KB
        testing::Values(DataType::INT8, DataType::INT32, DataType::FLOAT),
        testing::Values(1, 4), // QP numbers: 1, 4
        testing::Values(64, 128), // maxMsgPerQp: 64, 128
        testing::Values(128, 256)), // maxMsgBytes: 128, 256
    IbverbxVirtualQpSendRecvTestFixture::getTestName);

// Medium buffer configurations for Send/Recv - 1MB
INSTANTIATE_TEST_SUITE_P(
    VirtualQpSendRecvTestMediumBuffer1MB,
    IbverbxVirtualQpSendRecvTestFixture,
    ::testing::Combine(
        testing::Values(1048576), // Medium buffer sizes: 1MB
        testing::Values(DataType::INT8, DataType::INT32, DataType::FLOAT),
        testing::Values(16, 128), // QP numbers: 16, 128
        testing::Values(128, 1024), // maxMsgPerQp: 128, 1024
        testing::Values(1024, 16384)), // maxMsgBytes: 1024, 16384
    IbverbxVirtualQpSendRecvTestFixture::getTestName);

// Medium buffer configurations for Send/Recv - 8MG
INSTANTIATE_TEST_SUITE_P(
    VirtualQpSendRecvTestMediumBuffer8MB,
    IbverbxVirtualQpSendRecvTestFixture,
    ::testing::Combine(
        testing::Values(8388608), // Large buffer size: 8MB
        testing::Values(DataType::INT32, DataType::FLOAT),
        testing::Values(16, 128), // High QP number for maximum parallelism
        testing::Values(1024), // maxMsgPerQp: 1024
        testing::Values(16384)), // maxMsgBytes: 1MB
    IbverbxVirtualQpSendRecvTestFixture::getTestName);

// Large buffer configurations for Send/Recv - 1GB
INSTANTIATE_TEST_SUITE_P(
    VirtualQpSendRecvTestLargeBuffer,
    IbverbxVirtualQpSendRecvTestFixture,
    ::testing::Combine(
        testing::Values(1073741824), // Large buffer size: 1GB
        testing::Values(DataType::INT32),
        testing::Values(16, 128), // High QP number for maximum parallelism
        testing::Values(1024), // maxMsgPerQp: 1024
        testing::Values(1048576)), // maxMsgBytes: 1MB
    IbverbxVirtualQpSendRecvTestFixture::getTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
