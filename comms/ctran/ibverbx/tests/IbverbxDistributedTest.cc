// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <numeric>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
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
  initAttr.cap.max_send_wr = 256; // maximum outstanding send WRs
  initAttr.cap.max_recv_wr = 256; // maximum outstanding recv WRs
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

void changeQpStateToRts(
    IbvQp& qp,
    const BusinessCard& localCard,
    const BusinessCard& remoteCard) {
  {
    // change QP state to INIT
    auto qpAttr = makeQpAttrInit(localCard);
    ASSERT_TRUE(qp.modifyQp(
        &qpAttr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  }
  {
    // change QP state to RTR
    auto qpAttr = makeQpAttrRtr(remoteCard);
    ASSERT_TRUE(qp.modifyQp(
        &qpAttr,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));
  }
  {
    // change QP state to RTS
    auto qpAttr = makeQpAttrRts();
    ASSERT_TRUE(qp.modifyQp(
        &qpAttr,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
            IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
  }
}

void pollOneWcFromCq(int rank, IbvCq& cq) {
  int numEntries{20};

  bool stop = false;
  while (!stop) {
    auto maybeWcsVector = cq.pollCq(numEntries);
    if (maybeWcsVector.hasError()) {
      XLOGF(
          FATAL,
          "rank {}: cq poll failed with error {}",
          rank,
          maybeWcsVector.error().errStr);
      return;
    }

    auto numWc = maybeWcsVector->size();
    ASSERT_GE(numWc, 0);
    if (numWc == 0) {
      // CQ empty, sleep and retry
      XLOGF(WARN, "rank {}: cq empty, retry in 500ms", rank);
      /* sleep override */
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }

    ASSERT_EQ(numWc, 1);

    // got a WC
    const auto wc = maybeWcsVector->at(0);
    XLOGF(
        DBG1,
        "Rank {} got a wc: wr_id {}, status {}, opcode {}, byte_len {}",
        rank,
        wc.wr_id,
        wc.status,
        wc.opcode,
        wc.byte_len);
    ASSERT_EQ(wc.status, IBV_WC_SUCCESS);
    stop = true;
  }
}

} // namespace

class IbverbxSingleQpTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    ASSERT_TRUE(ibvInit());
  }
};

// Enum for different data types
enum class DataType { INT8, INT16, INT32, INT64, FLOAT, DOUBLE };

// Helper function to convert DataType enum to string
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

// Helper function to get size of data type enum
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

// Parameterized test class for single QP tests (SendRecv, RdmaWrite)
class IbverbxSingleQpTestFixtureWithParam
    : public IbverbxSingleQpTestFixture,
      public ::testing::WithParamInterface<std::tuple<int, DataType>> {
 public:
  // Parameterized test name generator function for single QP tests
  static std::string getTestName(
      const testing::TestParamInfo<ParamType>& info) {
    return fmt::format(
        "{}_devBufSize_{}_dataType",
        std::get<0>(info.param),
        dataTypeToString(std::get<1>(info.param)));
  }

 protected:
  void SetUp() override {
    IbverbxSingleQpTestFixture::SetUp();
  }

  // Helper template function to run test with specific data type
  template <typename T>
  void runSendRecvTest(int devBufSize);

  // Helper template function to run RDMA write test with specific data type
  template <typename T>
  void runRdmaWriteTest(int devBufSize);
};

// Template helper function implementation for Single QP tests
template <typename T>
void IbverbxSingleQpTestFixtureWithParam::runSendRecvTest(int devBufSize) {
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
  int cqe = 100;
  auto cq = device.createCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(cq);

  // make qp
  auto initAttr = makeIbvQpInitAttr(cq->cq());
  auto qp = pd->createQp(&initAttr);
  ASSERT_TRUE(qp);

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
      .qpNum = qp->qp()->qp_num,
      .port = kPortNum,
      .subnetPrefix = gid->global.subnet_prefix,
      .interfaceId = gid->global.interface_id,
      .rank = globalRank,
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

  // init qp
  changeQpStateToRts(*qp, localCard, remoteCard);

  // post send/recv and poll cq
  if (globalRank == 0) {
    // receiver

    ibv_sge sgList = {
        .addr = (uint64_t)devBuf,
        .length = static_cast<uint32_t>(devBufSize),
        .lkey = mr->mr()->lkey};
    ibv_recv_wr recvWr = {
        .wr_id = 0, .next = nullptr, .sg_list = &sgList, .num_sge = 1};
    ibv_recv_wr recvWrBad{};
    ASSERT_TRUE(qp->postRecv(&recvWr, &recvWrBad));
    pollOneWcFromCq(globalRank, *cq);

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
  } else if (globalRank == 1) {
    // sender

    ibv_sge sgList = {
        .addr = (uint64_t)devBuf,
        .length = static_cast<uint32_t>(devBufSize),
        .lkey = mr->mr()->lkey};
    ibv_send_wr sendWr = {
        .wr_id = 0,
        .next = nullptr,
        .sg_list = &sgList,
        .num_sge = 1,
        .opcode = IBV_WR_SEND,
        .send_flags = IBV_SEND_SIGNALED};
    ibv_send_wr sendWrBad{};
    ASSERT_TRUE(qp->postSend(&sendWr, &sendWrBad));
    pollOneWcFromCq(globalRank, *cq);
  }
  XLOGF(DBG1, "rank {} send/recv OK", globalRank);
}

// Template helper function implementation for RDMA Write
template <typename T>
void IbverbxSingleQpTestFixtureWithParam::runRdmaWriteTest(int devBufSize) {
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
  int cqe = 100;
  auto cq = device.createCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(cq);

  // make qp
  auto initAttr = makeIbvQpInitAttr(cq->cq());
  auto qp = pd->createQp(&initAttr);
  ASSERT_TRUE(qp);

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
      .qpNum = qp->qp()->qp_num,
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

  // init qp
  changeQpStateToRts(*qp, localCard, remoteCard);

  // post send/recv and poll cq
  if (globalRank == 0) {
    // receiver

    // post a dummy ibv_recv_wr as this is one-sided comm
    ibv_sge sgList = {};
    ibv_recv_wr recvWr = {.wr_id = 0, .sg_list = &sgList, .num_sge = 1};
    ibv_recv_wr recvWrBad{};
    ASSERT_TRUE(qp->postRecv(&recvWr, &recvWrBad));

    pollOneWcFromCq(globalRank, *cq);

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
  } else if (globalRank == 1) {
    // writer

    ibv_sge sgList = {
        .addr = (uint64_t)devBuf,
        .length = static_cast<uint32_t>(devBufSize),
        .lkey = mr->mr()->lkey};
    ibv_send_wr sendWr = {
        .wr_id = 0,
        .next = nullptr,
        .sg_list = &sgList,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_WRITE_WITH_IMM,
        .send_flags = IBV_SEND_SIGNALED};
    // set rdma remote fields for WRITE operation
    sendWr.wr.rdma.remote_addr = remoteCard.remoteAddr;
    sendWr.wr.rdma.rkey = remoteCard.rkey;
    sendWr.imm_data = 1;

    ibv_send_wr sendWrBad{};
    ASSERT_TRUE(qp->postSend(&sendWr, &sendWrBad));

    pollOneWcFromCq(globalRank, *cq);
    // the remote NIC/PCIe has completed memory transaction and sent an ACK to
    // us
  }
  XLOGF(DBG1, "rank {} RDMA-WRITE OK", globalRank);
}

TEST_F(IbverbxSingleQpTestFixture, IbvQpModifyQp) {
  auto devices = IbvDevice::ibvGetDeviceList();
  ASSERT_TRUE(devices);
  auto& device = devices->at(0);

  // make cq
  int cqe = 100;
  auto cq = device.createCq(cqe, nullptr, nullptr, 0);
  ASSERT_TRUE(cq);

  // make qp
  auto initAttr = makeIbvQpInitAttr(cq->cq());
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);
  auto qp = pd->createQp(&initAttr);
  ASSERT_TRUE(qp);

  // create local business card and exchange
  auto gid = device.queryGid(kPortNum, kGidIndex);
  ASSERT_TRUE(gid);

  BusinessCard localCard = {
      .mtu = IBV_MTU_4096,
      .qpNum = qp->qp()->qp_num,
      .port = kPortNum,
      .subnetPrefix = gid->global.subnet_prefix,
      .interfaceId = gid->global.interface_id,
      .rank = globalRank,
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

  changeQpStateToRts(*qp, localCard, remoteCard);
}

// Sender specifies <src-buf, src-len, lkey> with IBV_WR_SEND op
// Receiver specified <dst-buf, dst-len, lkey>
TEST_P(IbverbxSingleQpTestFixtureWithParam, SendRecvWithParam) {
  const auto& [devBufSize, dataType] = GetParam();

  // Dispatch to the appropriate template function based on data type
  switch (dataType) {
    case DataType::INT8:
      runSendRecvTest<int8_t>(devBufSize);
      break;
    case DataType::INT16:
      runSendRecvTest<int16_t>(devBufSize);
      break;
    case DataType::INT32:
      runSendRecvTest<int32_t>(devBufSize);
      break;
    case DataType::INT64:
      runSendRecvTest<int64_t>(devBufSize);
      break;
    case DataType::FLOAT:
      runSendRecvTest<float>(devBufSize);
      break;
    case DataType::DOUBLE:
      runSendRecvTest<double>(devBufSize);
      break;
  }
}

// Sender specifies <src-buf, dst-buf, lkey, src-len, rkey> with
// IBV_WR_RDMA_WRITE op
TEST_P(IbverbxSingleQpTestFixtureWithParam, RdmaWriteWithParam) {
  const auto& [devBufSize, dataType] = GetParam();

  // Dispatch to the appropriate template function based on data type
  switch (dataType) {
    case DataType::INT8:
      runRdmaWriteTest<int8_t>(devBufSize);
      break;
    case DataType::INT16:
      runRdmaWriteTest<int16_t>(devBufSize);
      break;
    case DataType::INT32:
      runRdmaWriteTest<int32_t>(devBufSize);
      break;
    case DataType::INT64:
      runRdmaWriteTest<int64_t>(devBufSize);
      break;
    case DataType::FLOAT:
      runRdmaWriteTest<float>(devBufSize);
      break;
    case DataType::DOUBLE:
      runRdmaWriteTest<double>(devBufSize);
      break;
  }
}

INSTANTIATE_TEST_SUITE_P(
    SingleQpTest,
    IbverbxSingleQpTestFixtureWithParam,
    ::testing::Combine(
        testing::Values(1024, 1048576, 1073741824), // Test 1 KB, 1 MB, 1 GB
        testing::Values(
            DataType::INT8,
            DataType::INT32,
            DataType::FLOAT)), // Test representative data types
    IbverbxSingleQpTestFixtureWithParam::getTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
