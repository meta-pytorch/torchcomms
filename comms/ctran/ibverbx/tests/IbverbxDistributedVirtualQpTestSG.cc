// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime.h>
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

// Number of scatter-gather buffers for all tests
constexpr int kNumSgBuffers = 4;

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
  initAttr.cap.max_send_sge = kNumSgBuffers; // support scatter-gather
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

class IbverbxVirtualQpSGTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    ncclCvarInit();
    ASSERT_TRUE(ibvInit());
  }

  // Setup structure to hold all initialized resources for scatter-gather tests
  struct VirtualQpSGSetup {
    std::vector<IbvDevice> devices;
    IbvPd pd;
    IbvVirtualCq virtualCq;
    IbvVirtualQp virtualQp;
    // Receiver's single destination buffer
    void* devBufRecv;
    size_t devBufRecvSize;
    IbvMr mrRecv;
    // Sender's 4 scatter-gather source buffers
    std::array<void*, kNumSgBuffers> sgDevBufs;
    size_t sgBufSize; // Size of each SG buffer
    std::vector<IbvMr> sgMrs;
    BusinessCard localCard;
    BusinessCard remoteCard;
    IbvVirtualQpBusinessCard remoteVirtualQpBusinessCard;
  };

  // Common setup function for scatter-gather tests
  // perBufferSize: size of each of the 4 scatter-gather buffers
  template <typename T>
  VirtualQpSGSetup setupVirtualQpSG(
      int perBufferSize,
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
    auto& device = devices.at(myDevId);
    auto maybePd = device.allocPd();
    EXPECT_TRUE(maybePd);
    auto pd = std::move(*maybePd);

    // make cq
    int cqe = 2 * numQp * maxMsgPerQp;
    auto maybeVirtualCq = device.createVirtualCq(cqe, nullptr, nullptr, 0);
    EXPECT_TRUE(maybeVirtualCq);
    EXPECT_NE(maybeVirtualCq->getPhysicalCqsRef().at(0).cq(), nullptr);
    auto virtualCq = std::move(*maybeVirtualCq);

    // make qp group
    uint32_t totalQps = numQp;
    auto initAttr = makeIbvQpInitAttr(virtualCq.getPhysicalCqsRef().at(0).cq());
    auto maybeVirtualQp = pd.createVirtualQp(
        totalQps,
        &initAttr,
        &virtualCq,
        &virtualCq,
        maxMsgPerQp,
        maxMsgBytes,
        loadBalancingScheme);
    EXPECT_TRUE(maybeVirtualQp);
    auto virtualQp = std::move(*maybeVirtualQp);

    // Total size for receiver = 4 * perBufferSize
    size_t totalRecvSize = kNumSgBuffers * perBufferSize;
    size_t numElementsPerBuf = perBufferSize / sizeof(T);

    // init device buffers
    void* devBufRecv{nullptr};
    CUDA_CHECK(cudaMalloc(&devBufRecv, totalRecvSize));

    std::array<void*, kNumSgBuffers> sgDevBufs{};
    std::vector<IbvMr> sgMrs;

    ibv_access_flags access = static_cast<ibv_access_flags>(
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ);

    if (globalRank == 0) {
      // receiver: fill destination buffer with 0s
      CUDA_CHECK(cudaMemset(devBufRecv, 0, totalRecvSize));
    } else if (globalRank == 1) {
      // sender: allocate and initialize 4 scatter-gather buffers
      for (int i = 0; i < kNumSgBuffers; ++i) {
        CUDA_CHECK(cudaMalloc(&sgDevBufs[i], perBufferSize));

        // Initialize each buffer with sequential values starting from
        // i*numElementsPerBuf + 1
        std::vector<T> hostBuf(numElementsPerBuf);
        if constexpr (std::is_integral_v<T>) {
          std::iota(
              hostBuf.begin(),
              hostBuf.end(),
              static_cast<T>(i * numElementsPerBuf + 1));
        } else {
          T val = static_cast<T>(i * numElementsPerBuf + 1);
          for (auto& elem : hostBuf) {
            elem = val;
            val += T(1);
          }
        }
        CUDA_CHECK(cudaMemcpy(
            sgDevBufs[i], hostBuf.data(), perBufferSize, cudaMemcpyDefault));

        // register memory for each SG buffer
        auto maybeMr = pd.regMr(sgDevBufs[i], perBufferSize, access);
        EXPECT_TRUE(maybeMr);
        sgMrs.push_back(std::move(*maybeMr));
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // register memory region for receiver
    auto maybeMrRecv = pd.regMr(devBufRecv, totalRecvSize, access);
    EXPECT_TRUE(maybeMrRecv);
    auto mrRecv = std::move(*maybeMrRecv);

    // create local business card and exchange
    auto gid = device.queryGid(kPortNum, kGidIndex);
    EXPECT_TRUE(gid);

    BusinessCard localCard = {
        .mtu = IBV_MTU_4096,
        .port = kPortNum,
        .subnetPrefix = gid->global.subnet_prefix,
        .interfaceId = gid->global.interface_id,
        .rank = globalRank,
        .remoteAddr = reinterpret_cast<uint64_t>(devBufRecv),
        .rkey = mrRecv.mr()->rkey,
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

    return VirtualQpSGSetup{
        .devices = std::move(devices),
        .pd = std::move(pd),
        .virtualCq = std::move(virtualCq),
        .virtualQp = std::move(virtualQp),
        .devBufRecv = devBufRecv,
        .devBufRecvSize = totalRecvSize,
        .mrRecv = std::move(mrRecv),
        .sgDevBufs = sgDevBufs,
        .sgBufSize = static_cast<size_t>(perBufferSize),
        .sgMrs = std::move(sgMrs),
        .localCard = localCard,
        .remoteCard = remoteCard,
        .remoteVirtualQpBusinessCard = std::move(remoteVirtualQpBusinessCard),
    };
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

// Parameterized test class for virtual QP scatter-gather RDMA write tests
class IbverbxVirtualQpRdmaWriteSGTestFixture
    : public IbverbxVirtualQpSGTestFixture,
      public ::testing::WithParamInterface<
          std::tuple<int, DataType, int, int, int, LoadBalancingScheme>> {
 public:
  // Parameterized test name generator function
  static std::string getTestName(
      const testing::TestParamInfo<ParamType>& info) {
    std::string baseName = fmt::format(
        "{}_perBufSize_{}_dataType_{}_numQp_",
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
    IbverbxVirtualQpSGTestFixture::SetUp();
  }

  // Helper template function to run scatter-gather RDMA write test
  template <typename T>
  void runRdmaWriteSGTest(
      int perBufferSize,
      int numQp,
      int maxMsgPerQp = -1,
      int maxMsgBytes = -1,
      LoadBalancingScheme loadBalancingScheme = LoadBalancingScheme::SPRAY);
};

// Template helper function implementation for Virtual QP Scatter-Gather RDMA
// Write
template <typename T>
void IbverbxVirtualQpRdmaWriteSGTestFixture::runRdmaWriteSGTest(
    int perBufferSize,
    int numQp,
    int maxMsgPerQp,
    int maxMsgBytes,
    LoadBalancingScheme loadBalancingScheme) {
  // Use common setup function to initialize IB resources with 4 SG buffers
  auto setup = setupVirtualQpSG<T>(
      perBufferSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);

  // post send/recv and poll cq
  int wr_id = 0;
  int imm_data = 16384;
  if (globalRank == 0) {
    // receiver - post a dummy ibv_recv_wr as this is one-sided comm
    ibv_sge sgList = {};
    ibv_recv_wr recvWr = {
        .wr_id = static_cast<uint64_t>(wr_id),
        .sg_list = &sgList,
        .num_sge = 0};
    ibv_recv_wr recvWrBad{};
    ASSERT_TRUE(setup.virtualQp.postRecv(&recvWr, &recvWrBad));
  } else if (globalRank == 1) {
    // sender - use postSendScatterGather with 4 buffers

    // Build scatter-gather list with 4 buffers
    std::array<ibv_sge, kNumSgBuffers> sgList{};
    for (int i = 0; i < kNumSgBuffers; ++i) {
      sgList[i] = {
          .addr = reinterpret_cast<uint64_t>(setup.sgDevBufs[i]),
          .length = static_cast<uint32_t>(setup.sgBufSize),
          .lkey = setup.sgMrs[i].mr()->lkey};
    }

    ibv_send_wr sendWr = {
        .wr_id = static_cast<uint64_t>(wr_id),
        .next = nullptr,
        .sg_list = sgList.data(),
        .num_sge = kNumSgBuffers,
        .opcode = IBV_WR_RDMA_WRITE_WITH_IMM,
        .send_flags = IBV_SEND_SIGNALED};
    // set rdma remote fields for WRITE operation
    sendWr.wr.rdma.remote_addr = setup.remoteCard.remoteAddr;
    sendWr.wr.rdma.rkey = setup.remoteCard.rkey;
    sendWr.imm_data = imm_data;

    // Build per-buffer keys (empty since we're using single NIC)
    std::vector<ScatterGatherBufferKeys> perBufferKeys;
    // For single NIC case, perBufferKeys can be empty

    ibv_send_wr sendWrBad{};
    ASSERT_TRUE(setup.virtualQp.postSendScatterGather(
        &sendWr, &sendWrBad, perBufferKeys));
  }

  // poll cq and check cq
  int numEntries{20};
  bool stop = false;
  while (!stop) {
    auto maybeWcsVector = setup.virtualCq.pollCq(numEntries);
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
    // check data - receiver should have received all 4 buffers concatenated
    size_t totalElements = setup.devBufRecvSize / sizeof(T);
    std::vector<T> hostExpectedBuf(totalElements);
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

    std::vector<T> hostRecvBuf(totalElements);
    CUDA_CHECK(cudaMemcpy(
        hostRecvBuf.data(),
        setup.devBufRecv,
        setup.devBufRecvSize,
        cudaMemcpyDefault));
    CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_EQ(hostExpectedBuf, hostRecvBuf);
  }
  XLOGF(DBG1, "rank {} RDMA-WRITE with scatter-gather OK", globalRank);

  // Clean up device buffers
  CUDA_CHECK(cudaFree(setup.devBufRecv));
  if (globalRank == 1) {
    for (int i = 0; i < kNumSgBuffers; ++i) {
      CUDA_CHECK(cudaFree(setup.sgDevBufs[i]));
    }
  }
}

// Scatter-Gather RDMA Write Virtual QP test using template helper
TEST_P(IbverbxVirtualQpRdmaWriteSGTestFixture, RdmaWriteSGVirtualQpWithParam) {
  const auto& [perBufSize, dataType, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme] =
      GetParam();

  // Dispatch to the appropriate template function based on data type
  switch (dataType) {
    case DataType::INT8:
      runRdmaWriteSGTest<int8_t>(
          perBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
    case DataType::INT16:
      runRdmaWriteSGTest<int16_t>(
          perBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
    case DataType::INT32:
      runRdmaWriteSGTest<int32_t>(
          perBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
    case DataType::INT64:
      runRdmaWriteSGTest<int64_t>(
          perBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
    case DataType::FLOAT:
      runRdmaWriteSGTest<float>(
          perBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
    case DataType::DOUBLE:
      runRdmaWriteSGTest<double>(
          perBufSize, numQp, maxMsgPerQp, maxMsgBytes, loadBalancingScheme);
      break;
  }
}

// Instantiate Scatter-Gather RDMA Write tests with different configurations
// Small buffer configurations - 1KB and 8KB per SG buffer (4KB and 32KB total)
INSTANTIATE_TEST_SUITE_P(
    VirtualQpRdmaWriteSGTestSmallBuffer,
    IbverbxVirtualQpRdmaWriteSGTestFixture,
    ::testing::Combine(
        testing::Values(1024, 8192), // Small per-buffer sizes: 1KB, 8KB
        testing::Values(DataType::INT8, DataType::INT32, DataType::FLOAT),
        testing::Values(1, 4), // QP numbers: 1, 4
        testing::Values(64, 128), // maxMsgPerQp: 64, 128
        testing::Values(128, 256), // maxMsgBytes: 128, 256
        testing::Values(
            LoadBalancingScheme::DQPLB,
            LoadBalancingScheme::SPRAY)), // LoadBalancingScheme
    IbverbxVirtualQpRdmaWriteSGTestFixture::getTestName);

// Medium buffer configurations - 256KB and 1MB per SG buffer (1MB and 4MB
// total)
INSTANTIATE_TEST_SUITE_P(
    VirtualQpRdmaWriteSGTestMediumBuffer,
    IbverbxVirtualQpRdmaWriteSGTestFixture,
    ::testing::Combine(
        testing::Values(
            262144,
            1048576), // Medium per-buffer sizes: 256KB, 1MB
        testing::Values(DataType::INT8, DataType::INT32, DataType::FLOAT),
        testing::Values(16, 128), // QP numbers: 16, 128
        testing::Values(128, 1024), // maxMsgPerQp: 128, 1024
        testing::Values(1024, 16384), // maxMsgBytes: 1024, 16384
        testing::Values(
            LoadBalancingScheme::DQPLB,
            LoadBalancingScheme::SPRAY)), // LoadBalancingScheme
    IbverbxVirtualQpRdmaWriteSGTestFixture::getTestName);

// Large buffer configurations - 256MB per SG buffer (1GB total)
// Note: FLOAT is excluded from large buffer tests because single-precision
// floats (32-bit) can only represent consecutive integers exactly up to 2^24
// (16,777,216). With 256MB per buffer (67M+ floats), the total sequence exceeds
// float precision limits, causing comparison failures.
INSTANTIATE_TEST_SUITE_P(
    VirtualQpRdmaWriteSGTestLargeBuffer,
    IbverbxVirtualQpRdmaWriteSGTestFixture,
    ::testing::Combine(
        testing::Values(268435456), // Large per-buffer size: 256MB (1GB total)
        testing::Values(DataType::INT8, DataType::INT32),
        testing::Values(16, 128), // High QP number for maximum parallelism
        testing::Values(128, 1024), // maxMsgPerQp: 128, 1024
        testing::Values(16384, 1048576), // maxMsgBytes: 16KB, 1MB
        testing::Values(
            LoadBalancingScheme::DQPLB,
            LoadBalancingScheme::SPRAY)), // LoadBalancingScheme
    IbverbxVirtualQpRdmaWriteSGTestFixture::getTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
