// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <numeric>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/testinfra/MpiTestUtils.h"
#include "comms/utils/checks.h"

using namespace ibverbx;
using namespace meta::comms;

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
  uint64_t subnetPrefix{0};
  uint64_t interfaceId{0};
  int32_t rank{-1};
  // following fields are for RDMA_WRITE
  uint64_t remoteAddr{0};
  uint32_t rkey{0};
};

std::ostream& operator<<(std::ostream& out, BusinessCard const& card) {
  out << fmt::format(
      "<rank {} qp-num {}, port {}, gid {:x}/{:x} remoteAddr {:x}, rkey {:x}>",
      card.rank,
      card.qpNum,
      card.port,
      card.subnetPrefix,
      card.interfaceId,
      card.remoteAddr,
      card.rkey);
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

ibv_qp_attr makeQpAttrRtr(const BusinessCard& remoteCard) {
  // The Service Level to be used
  uint8_t kServiceLevel = 0;
  int kTrafficClass = 0;

  ibv_qp_attr qpAttr{};
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));

  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = remoteCard.mtu;
  qpAttr.dest_qp_num = remoteCard.qpNum;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;

  // assume IBV_LINK_LAYER_ETHERNET
  qpAttr.ah_attr.is_global = 1;
  qpAttr.ah_attr.grh.dgid.global.subnet_prefix = remoteCard.subnetPrefix;
  qpAttr.ah_attr.grh.dgid.global.interface_id = remoteCard.interfaceId;
  qpAttr.ah_attr.grh.flow_label = 0;
  qpAttr.ah_attr.grh.sgid_index = kGidIndex;
  qpAttr.ah_attr.grh.hop_limit = 255;
  qpAttr.ah_attr.grh.traffic_class = kTrafficClass;
  qpAttr.ah_attr.sl = kServiceLevel;
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = remoteCard.port;
  return qpAttr;
}

ibv_qp_attr makeQpAttrRts() {
  // 4us x 2^20 = 4s
  const uint8_t kTimeout = 20;
  const uint8_t kRetryCnt = 1;

  struct ibv_qp_attr qpAttr {};
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
  {
    // change QP group state to INIT
    auto qpAttr = makeQpAttrInit(localCard);
    ASSERT_TRUE(virtualQp.modifyVirtualQp(
        &qpAttr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  }
  {
    // change QP group state to RTR
    auto qpAttr = makeQpAttrRtr(remoteCard);
    ASSERT_TRUE(virtualQp.modifyVirtualQp(
        &qpAttr,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER,
        remoteVirtualQpBusinessCard));
  }
  {
    // change QP group state to RTS
    auto qpAttr = makeQpAttrRts();
    ASSERT_TRUE(virtualQp.modifyVirtualQp(
        &qpAttr,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
            IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
  }
}

} // namespace

class IbverbxVirtualQpTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    ASSERT_TRUE(ibvInit());
  }
};

TEST_F(IbverbxVirtualQpTestFixture, IbvVirtualQpModifyVirtualQp) {
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  // make cq
  int cqe = 100;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  ASSERT_NE(maybeVirtualCq->getPhysicalCqRef().cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  // make qp group
  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqRef().cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  uint32_t totalQps = 16;
  auto virtualQp =
      pd->createVirtualQp(totalQps, &initAttr, &virtualCq, &virtualCq);
  ASSERT_TRUE(virtualQp);

  // create local business card and exchange
  auto gid = device.queryGid(kPortNum, kGidIndex);
  ASSERT_TRUE(gid);

  BusinessCard localCard = {
      .mtu = IBV_MTU_4096,
      .port = kPortNum,
      .subnetPrefix = gid->global.subnet_prefix,
      .interfaceId = gid->global.interface_id,
      .rank = globalRank,
  };
  auto virtualQpBusinessCard = virtualQp->getVirtualQpBusinessCard();
  ASSERT_EQ(virtualQpBusinessCard.qpNums_.size(), totalQps);

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
      virtualQp->getVirtualQpBusinessCard().serialize();

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
  ASSERT_TRUE(maybeRemoteVirtualQpBusinessCard);
  auto remoteVirtualQpBusinessCard =
      std::move(*maybeRemoteVirtualQpBusinessCard);
  ASSERT_EQ(remoteVirtualQpBusinessCard.qpNums_.size(), totalQps);

  changeVirtualQpStateToRts(
      *virtualQp, localCard, remoteCard, remoteVirtualQpBusinessCard);
}

TEST_F(IbverbxVirtualQpTestFixture, IbvVirtualQpMultipleRdmaWrites) {
  CUDA_CHECK(cudaSetDevice(localRank));

  int myDevId{0};
  CUDA_CHECK(cudaGetDevice(&myDevId));

  // get device
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(myDevId);
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  // make cq
  int cqe = 1600;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  ASSERT_NE(maybeVirtualCq->getPhysicalCqRef().cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  // make qp group
  int totalQps = 16;
  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqRef().cq());
  auto virtualQp = pd->createVirtualQp(
      totalQps, &initAttr, &virtualCq, &virtualCq, 128, 1024);
  ASSERT_TRUE(virtualQp);

  // init device buffer for receiver
  void* devBuf{nullptr};
  size_t devBufSize = 1048576;
  CUDA_CHECK(cudaMalloc(&devBuf, devBufSize));

  // Calculate number of elements based on buffer size
  size_t numElements = devBufSize / sizeof(int64_t);

  // init sender buffers (2 buffers with different data)
  void* devBuf1{nullptr};
  void* devBuf2{nullptr};
  std::vector<int64_t> hostBuf1(numElements);
  std::vector<int64_t> hostBuf2(numElements);

  if (globalRank == 0) {
    // receiver: initialize with zeros
    std::vector<int64_t> hostBuf(numElements);
    std::fill(hostBuf.begin(), hostBuf.end(), 0);
    CUDA_CHECK(
        cudaMemcpy(devBuf, hostBuf.data(), devBufSize, cudaMemcpyDefault));
  } else if (globalRank == 1) {
    // sender: allocate 2 buffers with different data
    CUDA_CHECK(cudaMalloc(&devBuf1, devBufSize));
    CUDA_CHECK(cudaMalloc(&devBuf2, devBufSize));

    // First buffer: fill with pattern starting from 100
    std::iota(hostBuf1.begin(), hostBuf1.end(), 100);
    CUDA_CHECK(
        cudaMemcpy(devBuf1, hostBuf1.data(), devBufSize, cudaMemcpyDefault));

    // Second buffer: fill with pattern starting from 200
    std::iota(hostBuf2.begin(), hostBuf2.end(), 200);
    CUDA_CHECK(
        cudaMemcpy(devBuf2, hostBuf2.data(), devBufSize, cudaMemcpyDefault));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // register memory regions
  ibv_access_flags access = static_cast<ibv_access_flags>(
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
      IBV_ACCESS_REMOTE_READ);

  auto mr = pd->regMr(devBuf, devBufSize, access);
  ASSERT_TRUE(mr);

  // create local business card and exchange
  auto gid = device.queryGid(kPortNum, kGidIndex);
  ASSERT_TRUE(gid);

  BusinessCard localCard = {
      .mtu = IBV_MTU_4096,
      .port = kPortNum,
      .subnetPrefix = gid->global.subnet_prefix,
      .interfaceId = gid->global.interface_id,
      .rank = globalRank,
      .remoteAddr = reinterpret_cast<uint64_t>(devBuf),
      .rkey = mr->mr()->rkey,
  };
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
      virtualQp->getVirtualQpBusinessCard().serialize();

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

  ASSERT_TRUE(maybeRemoteVirtualQpBusinessCard);
  auto remoteVirtualQpBusinessCard =
      std::move(*maybeRemoteVirtualQpBusinessCard);

  // init qp group
  changeVirtualQpStateToRts(
      *virtualQp, localCard, remoteCard, remoteVirtualQpBusinessCard);

  // post RDMA writes and poll cq
  int imm_data1 = 16384;
  int imm_data2 = 32768;

  if (globalRank == 0) {
    // receiver: post 2 dummy recv WRs for the 2 RDMA writes with immediate
    for (int i = 0; i < 2; ++i) {
      ibv_sge sgList = {};
      ibv_recv_wr recvWr = {
          .wr_id = static_cast<uint64_t>(i), .sg_list = &sgList, .num_sge = 0};
      ibv_recv_wr recvWrBad{};
      ASSERT_TRUE(virtualQp->postRecv(&recvWr, &recvWrBad));
    }
  } else if (globalRank == 1) {
    // sender: register memory for the 2 buffers and post 2 RDMA writes

    auto mr1 = pd->regMr(devBuf1, devBufSize, access);
    ASSERT_TRUE(mr1);

    auto mr2 = pd->regMr(devBuf2, devBufSize, access);
    ASSERT_TRUE(mr2);

    // First RDMA write
    ibv_sge sgList1 = {
        .addr = (uint64_t)devBuf1,
        .length = static_cast<uint32_t>(devBufSize),
        .lkey = mr1->mr()->lkey};
    ibv_send_wr sendWr1 = {
        .wr_id = 0,
        .next = nullptr,
        .sg_list = &sgList1,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_WRITE_WITH_IMM,
        .send_flags = IBV_SEND_SIGNALED};
    sendWr1.wr.rdma.remote_addr = remoteCard.remoteAddr;
    sendWr1.wr.rdma.rkey = remoteCard.rkey;
    sendWr1.imm_data = imm_data1;

    // Second RDMA write
    ibv_sge sgList2 = {
        .addr = (uint64_t)devBuf2,
        .length = static_cast<uint32_t>(devBufSize),
        .lkey = mr2->mr()->lkey};
    ibv_send_wr sendWr2 = {
        .wr_id = 1,
        .next = nullptr,
        .sg_list = &sgList2,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_WRITE_WITH_IMM,
        .send_flags = IBV_SEND_SIGNALED};
    sendWr2.wr.rdma.remote_addr = remoteCard.remoteAddr;
    sendWr2.wr.rdma.rkey = remoteCard.rkey;
    sendWr2.imm_data = imm_data2;

    // Post both RDMA writes
    ibv_send_wr sendWrBad{};
    ASSERT_TRUE(virtualQp->postSend(&sendWr1, &sendWrBad));
    ASSERT_TRUE(virtualQp->postSend(&sendWr2, &sendWrBad));
  }

  // poll cq and check completion entries
  int numEntries{20};
  int completedOps = 0;
  int expectedOps = 2; // 2 operations per rank

  while (completedOps < expectedOps) {
    auto maybeWcsVector = virtualCq.pollCq(numEntries);
    ASSERT_TRUE(maybeWcsVector);
    auto numWc = maybeWcsVector->size();

    if (numWc == 0) {
      // CQ empty, sleep and retry
      XLOGF(WARN, "rank {}: cq empty, retry in 500ms", globalRank);
      /* sleep override */
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }

    for (size_t i = 0; i < numWc; ++i) {
      const auto& wc = maybeWcsVector->at(i);
      ASSERT_EQ(wc.status, IBV_WC_SUCCESS);

      if (globalRank == 0) {
        // receiver: check immediate data values
        if (wc.wr_id == 0) {
          ASSERT_EQ(wc.imm_data, imm_data1);
        } else if (wc.wr_id == 1) {
          ASSERT_EQ(wc.imm_data, imm_data2);
        }
      } else if (globalRank == 1) {
        // sender: verify completion of our writes
        ASSERT_TRUE(wc.wr_id == 0 || wc.wr_id == 1);
      }

      XLOGF(INFO, "Rank {} got WC: wr_id {}", globalRank, wc.wr_id);
      completedOps++;
    }
  }

  // receiver check data - should match the second RDMA write (buffer2)
  if (globalRank == 0) {
    std::vector<int64_t> hostExpectedBuf(numElements);
    std::iota(
        hostExpectedBuf.begin(),
        hostExpectedBuf.end(),
        200); // Second buffer data

    std::vector<int64_t> hostRecvBuf(numElements);
    CUDA_CHECK(
        cudaMemcpy(hostRecvBuf.data(), devBuf, devBufSize, cudaMemcpyDefault));
    CUDA_CHECK(cudaDeviceSynchronize());

    ASSERT_EQ(hostExpectedBuf, hostRecvBuf);
    XLOGF(INFO, "rank {} verified data matches second RDMA write", globalRank);
  }

  XLOGF(
      DBG1,
      "rank {} multiple RDMA writes test completed successfully",
      globalRank);

  // Clean up sender buffers
  if (globalRank == 1) {
    CUDA_CHECK(cudaFree(devBuf1));
    CUDA_CHECK(cudaFree(devBuf2));
  }
}

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
  CUDA_CHECK(cudaSetDevice(localRank));

  int myDevId{-1};
  CUDA_CHECK(cudaGetDevice(&myDevId));

  // get device
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(myDevId);
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  // make cq
  int cqe = 2 * numQp * maxMsgPerQp;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  ASSERT_NE(maybeVirtualCq->getPhysicalCqRef().cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  // make qp group
  uint32_t totalQps = numQp;
  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqRef().cq());
  auto virtualQp = pd->createVirtualQp(
      totalQps,
      &initAttr,
      &virtualCq,
      &virtualCq,
      maxMsgPerQp,
      maxMsgBytes,
      loadBalancingScheme);
  ASSERT_TRUE(virtualQp);

  // init device buffer
  void* devBuf{nullptr};
  CUDA_CHECK(cudaMalloc(&devBuf, devBufSize));
  size_t numElements = devBufSize / sizeof(T);
  std::vector<T> hostBuf(numElements);

  if (globalRank == 0) {
    // receiver: fill up with 0s
    std::fill(hostBuf.begin(), hostBuf.end(), T{});
  } else if (globalRank == 1) {
    // writer: initialize with sequence number (1, 2, 3, ...)
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
  CUDA_CHECK(cudaMemcpy(devBuf, hostBuf.data(), devBufSize, cudaMemcpyDefault));
  CUDA_CHECK(cudaDeviceSynchronize());

  // register mr
  ibv_access_flags access = static_cast<ibv_access_flags>(
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
      IBV_ACCESS_REMOTE_READ);
  auto mr = pd->regMr(devBuf, devBufSize, access);
  ASSERT_TRUE(mr);

  // create local business card and exchange
  auto gid = device.queryGid(kPortNum, kGidIndex);
  ASSERT_TRUE(gid);

  BusinessCard localCard = {
      .mtu = IBV_MTU_4096,
      .port = kPortNum,
      .subnetPrefix = gid->global.subnet_prefix,
      .interfaceId = gid->global.interface_id,
      .rank = globalRank,
      .remoteAddr = reinterpret_cast<uint64_t>(devBuf),
      .rkey = mr->mr()->rkey,
  };
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
      virtualQp->getVirtualQpBusinessCard().serialize();

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
  ASSERT_TRUE(maybeRemoteVirtualQpBusinessCard);
  auto remoteVirtualQpBusinessCard =
      std::move(*maybeRemoteVirtualQpBusinessCard);

  // init qp group
  changeVirtualQpStateToRts(
      *virtualQp, localCard, remoteCard, remoteVirtualQpBusinessCard);

  // post send/recv and poll cq
  int wr_id = 0;
  int imm_data = 16384;
  if (globalRank == 0) {
    // receiver

    // post a dummy ibv_recv_wr as this is one-sided comm
    ibv_sge sgList = {};
    ibv_recv_wr recvWr = {
        .wr_id = static_cast<uint64_t>(wr_id),
        .sg_list = &sgList,
        .num_sge = 0};
    ibv_recv_wr recvWrBad{};
    ASSERT_TRUE(virtualQp->postRecv(&recvWr, &recvWrBad));
  } else if (globalRank == 1) {
    // writer

    ibv_sge sgList = {
        .addr = (uint64_t)devBuf,
        .length = static_cast<uint32_t>(devBufSize),
        .lkey = mr->mr()->lkey};
    ibv_send_wr sendWr = {
        .wr_id = static_cast<uint64_t>(wr_id),
        .next = nullptr,
        .sg_list = &sgList,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_WRITE_WITH_IMM,
        .send_flags = IBV_SEND_SIGNALED};
    // set rdma remote fields for WRITE operation
    sendWr.wr.rdma.remote_addr = remoteCard.remoteAddr;
    sendWr.wr.rdma.rkey = remoteCard.rkey;
    sendWr.imm_data = imm_data;

    ibv_send_wr sendWrBad{};
    ASSERT_TRUE(virtualQp->postSend(&sendWr, &sendWrBad));
  }

  // poll cq and check cq
  int numEntries{20};
  bool stop = false;
  while (!stop) {
    auto maybeWcsVector = virtualCq.pollCq(numEntries);
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
    ASSERT_EQ(wc.wr_id, wr_id);
    ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
    if (globalRank == 0 && loadBalancingScheme == LoadBalancingScheme::SPRAY) {
      // In spray mode, receive checks the value of the IMM field to determine
      // status. In DQPLB mode, the IMM field is used to track sequence number
      // completion, so it should not be checked here.
      ASSERT_EQ(wc.imm_data, imm_data);
    }
    XLOGF(DBG1, "Rank {} got a wc: wr_id {}", globalRank, wc.wr_id);
    stop = true;
  }

  // receiver check data
  if (globalRank == 0) {
    // check data
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
    CUDA_CHECK(
        cudaMemcpy(hostRecvBuf.data(), devBuf, devBufSize, cudaMemcpyDefault));
    CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_EQ(hostExpectedBuf, hostRecvBuf);
  }
  XLOGF(DBG1, "rank {} RDMA-WRITE OK", globalRank);
}

// Template helper function implementation for Virtual QP Send/Recv
template <typename T>
void IbverbxVirtualQpSendRecvTestFixture::runSendRecvVirtualQpTest(
    int devBufSize,
    int numQp,
    int maxMsgPerQp,
    int maxMsgBytes) {
  CUDA_CHECK(cudaSetDevice(localRank));

  int myDevId{-1};
  CUDA_CHECK(cudaGetDevice(&myDevId));

  // get device
  auto devices = IbvDevice::ibvGetDeviceList({kNicPrefix});
  ASSERT_TRUE(devices);
  auto& device = devices->at(myDevId);
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);

  // make cq
  int cqe = 2 * numQp * maxMsgPerQp;
  auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeVirtualCq);
  ASSERT_NE(maybeVirtualCq->getPhysicalCqRef().cq(), nullptr);
  auto virtualCq = std::move(*maybeVirtualCq);

  // make qp group
  uint32_t totalQps = numQp;
  auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqRef().cq());
  auto virtualQp = pd->createVirtualQp(
      totalQps, &initAttr, &virtualCq, &virtualCq, maxMsgPerQp, maxMsgBytes);
  ASSERT_TRUE(virtualQp);

  // init device buffer
  void* devBuf{nullptr};
  CUDA_CHECK(cudaMalloc(&devBuf, devBufSize));
  size_t numElements = devBufSize / sizeof(T);
  std::vector<T> hostBuf(numElements);

  if (globalRank == 0) {
    // receiver: fill up with 0s
    std::fill(hostBuf.begin(), hostBuf.end(), T{});
  } else if (globalRank == 1) {
    // sender: initialize with sequence number (1, 2, 3, ...)
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
  CUDA_CHECK(cudaMemcpy(devBuf, hostBuf.data(), devBufSize, cudaMemcpyDefault));
  CUDA_CHECK(cudaDeviceSynchronize());

  // register mr
  ibv_access_flags access = static_cast<ibv_access_flags>(
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
      IBV_ACCESS_REMOTE_READ);
  auto mr = pd->regMr(devBuf, devBufSize, access);
  ASSERT_TRUE(mr);

  // create local business card and exchange
  auto gid = device.queryGid(kPortNum, kGidIndex);
  ASSERT_TRUE(gid);

  BusinessCard localCard = {
      .mtu = IBV_MTU_4096,
      .port = kPortNum,
      .subnetPrefix = gid->global.subnet_prefix,
      .interfaceId = gid->global.interface_id,
      .rank = globalRank,
      .remoteAddr = reinterpret_cast<uint64_t>(devBuf),
      .rkey = mr->mr()->rkey,
  };
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
      virtualQp->getVirtualQpBusinessCard().serialize();

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
  ASSERT_TRUE(maybeRemoteVirtualQpBusinessCard);
  auto remoteVirtualQpBusinessCard =
      std::move(*maybeRemoteVirtualQpBusinessCard);

  // init qp group
  changeVirtualQpStateToRts(
      *virtualQp, localCard, remoteCard, remoteVirtualQpBusinessCard);

  // post send/recv within each virtual qp and poll cq
  int wr_id = 0;
  if (globalRank == 0) {
    // receiver
    ibv_recv_wr recvWr{};
    ibv_recv_wr recvWrBad{};
    struct ibv_sge sge = {
        .addr = (uint64_t)devBuf,
        .length = static_cast<uint32_t>(devBufSize),
        .lkey = mr->mr()->lkey,
    };
    recvWr.wr_id = wr_id;
    recvWr.sg_list = &sge;
    recvWr.num_sge = 1;
    recvWr.next = nullptr;
    ASSERT_TRUE(virtualQp->postRecv(&recvWr, &recvWrBad));
  } else if (globalRank == 1) {
    // sender
    ibv_send_wr sendWr{};
    ibv_send_wr sendWrBad{};
    struct ibv_sge sge = {
        .addr = (uint64_t)devBuf,
        .length = static_cast<uint32_t>(devBufSize),
        .lkey = mr->mr()->lkey,
    };
    sendWr.wr_id = wr_id;
    sendWr.sg_list = &sge;
    sendWr.num_sge = 1;
    sendWr.opcode = IBV_WR_SEND;
    sendWr.send_flags = IBV_SEND_SIGNALED;
    sendWr.next = nullptr;
    ASSERT_TRUE(virtualQp->postSend(&sendWr, &sendWrBad));
  }

  // poll cq and check cq
  int numEntries{20};
  bool stop = false;
  while (!stop) {
    auto maybeWcsVector = virtualCq.pollCq(numEntries);
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
    ASSERT_EQ(wc.wr_id, wr_id);
    ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
    if (globalRank == 0) {
      // According to ibverblib, this value is relevant for the Receive Queue
      // when handling incoming Send or RDMA Write with immediate operations.
      // Note that this value excludes the length of any immediate data present.
      // For the Send Queue, this value applies to RDMA Read and Atomic
      // operations. Therefore, only check the receiver's work completion (wc)
      // byte_len here.
      ASSERT_EQ(wc.byte_len, devBufSize);
    }
    XLOGF(DBG1, "Rank {} got a wc: wr_id {}", globalRank, wc.wr_id);
    stop = true;
  }

  // receiver check data
  if (globalRank == 0) {
    // check data
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
    CUDA_CHECK(
        cudaMemcpy(hostRecvBuf.data(), devBuf, devBufSize, cudaMemcpyDefault));
    CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_EQ(hostExpectedBuf, hostRecvBuf);
  }
  XLOGF(DBG1, "rank {} send/recv OK", globalRank);
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
