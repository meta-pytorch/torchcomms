// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <vector>

#include <folly/init/Init.h>
#include <folly/logging/Init.h>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ibverbx;

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

// use broadcom nic for AMD platform, use mellanox nic for NV platform
#if defined(__HIP_PLATFORM_AMD__) && !defined(USE_FE_NIC)
const std::string kNicPrefix("bnxt_re");
#else
const std::string kNicPrefix("mlx5_");
#endif

constexpr uint8_t kPortNum = 1;
constexpr int kGidIndex = 3;
constexpr uint32_t kTotalQps = 16;
constexpr uint32_t kMaxMsgCntPerQp = 128;
constexpr uint32_t kMaxMsgSize = 524288;
constexpr uint32_t kMaxOutstandingWrs = 128;
constexpr uint32_t kMaxSge = 1;

namespace {

class IbvEndPoint {
 public:
  IbvEndPoint(
      int nicDevId,
      LoadBalancingScheme loadBalancingScheme = LoadBalancingScheme::SPRAY);
  ~IbvEndPoint();
  ibv_qp_init_attr makeIbvQpInitAttr(ibv_cq* cq);
  ibv_qp_attr makeQpAttrInit();
  ibv_qp_attr makeQpAttrRtr(ibv_gid remoteGid);
  void changeVirtualQpStateToRts(
      const std::vector<ibv_gid>& remoteGids,
      const IbvVirtualQpBusinessCard& remoteVirtualQpBusinessCard);

  std::vector<IbvDevice> devices;
  std::vector<IbvPd> pds;
  IbvVirtualCq cq;
  IbvVirtualQp qp;
};

IbvEndPoint::IbvEndPoint(int nicDevId, LoadBalancingScheme loadBalancingScheme)
    : devices(([nicDevId]() {
        // Initialize ibverbx first
        auto initResult = ibvInit();
        if (!initResult) {
          throw std::runtime_error("ibvInit() failed");
        }

        // TODO: Currently, we use NCCL_IB_HCA to obtain the list of InfiniBand
        // devices. In the future, since Ibverbx is a standalone IB library, it
        // should provide its own interface to enumerate available IB devices
        // and return the device list.
        auto deviceList =
            IbvDevice::ibvGetDeviceList(NCCL_IB_HCA, NCCL_IB_HCA_PREFIX);
        if (!deviceList) {
          throw std::runtime_error("Failed to get device list");
        }

        if (deviceList->empty()) {
          throw std::runtime_error("No InfiniBand devices available");
        }

        if (nicDevId * 2 + 1 >= static_cast<int>(deviceList->size())) {
          throw std::out_of_range("nicDevId out of range");
        }

        std::vector<IbvDevice> selectedDevices;
        auto& dev0 = deviceList->at(nicDevId * 2);
        auto& dev1 = deviceList->at(nicDevId * 2 + 1);
        selectedDevices.push_back(std::move(dev0));
        selectedDevices.push_back(std::move(dev1));
        return selectedDevices;
      })()),
      pds(([this]() {
        std::vector<IbvPd> pdList;
        for (auto& device : devices) {
          auto maybePd = device.allocPd();
          if (!maybePd) {
            throw std::runtime_error("Failed to allocate protection domain");
          }
          pdList.push_back(std::move(*maybePd));
        }
        return pdList;
      })()),
      cq(([this]() {
        std::vector<IbvCq> cqList;
        for (auto& device : devices) {
          auto maybeCq = device.createCq(32768, nullptr, nullptr, 0);
          if (!maybeCq) {
            throw std::runtime_error("Failed to create completion queue");
          }
          cqList.push_back(std::move(*maybeCq));
        }
        return IbvVirtualCq(std::move(cqList), 32768);
      })()),
      qp([this, loadBalancingScheme]() {
        std::vector<IbvQp> qps;
        qps.reserve(kTotalQps);

        // First half of kTotalQps created from pds[0] using
        // cq.getPhysicalCqsRef().at(0) Second half of kTotalQps created from
        // pds[1] using cq.getPhysicalCqsRef().at(1)
        size_t numSendCqs = cq.getPhysicalCqsRef().size();
        size_t numPds = pds.size();

        for (int i = 0; i < kTotalQps; i++) {
          // Distribute QPs evenly across PDs and physical CQs
          size_t pdIdx = (i * numPds) / kTotalQps;
          size_t cqIdx = (i * numSendCqs) / kTotalQps;
          auto initAttr =
              makeIbvQpInitAttr(cq.getPhysicalCqsRef().at(cqIdx).cq());

          auto maybeQp = pds.at(pdIdx).createQp(&initAttr);
          if (maybeQp.hasError()) {
            throw std::runtime_error("Failed to create queue pair");
          }
          qps.emplace_back(std::move(*maybeQp));
        }

        // Create notify QP using the first PD and first physical CQ
        auto notifyInitAttr =
            makeIbvQpInitAttr(cq.getPhysicalCqsRef().at(0).cq());

        auto maybeNotifyQp = pds.at(0).createQp(&notifyInitAttr);
        if (maybeNotifyQp.hasError()) {
          throw std::runtime_error("Failed to create notify queue pair");
        }

        // Create the IbvVirtualQp instance
        return IbvVirtualQp(
            std::move(qps),
            &cq,
            kMaxMsgCntPerQp,
            kMaxMsgSize,
            loadBalancingScheme,
            std::move(*maybeNotifyQp));
      }()) {}

IbvEndPoint::~IbvEndPoint() = default;

// helper functions
ibv_qp_init_attr IbvEndPoint::makeIbvQpInitAttr(ibv_cq* cq) {
  ibv_qp_init_attr initAttr{};
  memset(&initAttr, 0, sizeof(ibv_qp_init_attr));
  initAttr.send_cq = cq;
  initAttr.recv_cq = cq;
  initAttr.qp_type = IBV_QPT_RC; // Reliable Connection
  initAttr.sq_sig_all = 0;
  initAttr.cap.max_send_wr = kMaxOutstandingWrs; // maximum outstanding send WRs
  initAttr.cap.max_recv_wr = kMaxOutstandingWrs; // maximum outstanding recv WRs
  initAttr.cap.max_send_sge = kMaxSge;
  initAttr.cap.max_recv_sge = kMaxSge;
  initAttr.cap.max_inline_data = 0;
  return initAttr;
}

ibv_qp_attr IbvEndPoint::makeQpAttrInit() {
  ibv_qp_attr qpAttr = {
      .qp_state = IBV_QPS_INIT,
      .qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
          IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_REMOTE_WRITE,
      .pkey_index = 0,
      .port_num = kPortNum,
  };
  return qpAttr;
}

ibv_qp_attr IbvEndPoint::makeQpAttrRtr(ibv_gid remoteGid) {
  uint8_t kServiceLevel = 0;
  int kTrafficClass = 0;

  ibv_qp_attr qpAttr{};
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));

  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = IBV_MTU_4096;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;

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
  const uint8_t kTimeout = 10;
  const uint8_t kRetryCnt = 7;

  struct ibv_qp_attr qpAttr{};
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));

  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = kTimeout;
  qpAttr.retry_cnt = kRetryCnt;

  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  return qpAttr;
}

void IbvEndPoint::changeVirtualQpStateToRts(
    const std::vector<ibv_gid>& remoteGids,
    const IbvVirtualQpBusinessCard& remoteVirtualQpBusinessCard) {
  size_t totalQps = qp.getTotalQps();
  size_t numRemoteGids = remoteGids.size();

  if (numRemoteGids == 0) {
    throw std::runtime_error("Remote GIDs cannot be empty");
  }

  // Get references to physical QPs and notify QP
  auto& physicalQps = qp.getQpsRef();
  auto& notifyQp = qp.getNotifyQpRef();

  // Modify all QPs one by one
  for (size_t i = 0; i < totalQps; i++) {
    // Distribute QPs evenly across remote GIDs
    size_t remoteGidIdx = (i * numRemoteGids) / totalQps;

    const auto& remoteGid = remoteGids.at(remoteGidIdx);

    // Change QP state to INIT
    {
      auto qpAttr = makeQpAttrInit();
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

  // Modify notify QP (using the first remote GID)
  const auto& notifyRemoteGid = remoteGids.at(0);

  // Change notify QP state to INIT
  {
    auto qpAttr = makeQpAttrInit();
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

struct BenchmarkSetup {
  std::unique_ptr<IbvEndPoint> sender;
  std::unique_ptr<IbvEndPoint> receiver;
  void* sendBuffer{};
  void* recvBuffer{};
  std::vector<std::optional<IbvMr>> sendMrs; // One MR per PD
  std::vector<std::optional<IbvMr>> recvMrs; // One MR per PD
  std::unordered_map<int32_t, MemoryRegionKeys> senderDeviceIdToKeys;
  std::unordered_map<int32_t, MemoryRegionKeys> receiverDeviceIdToKeys;

  // Disable copy and move operations for RAII resource management
  BenchmarkSetup(const BenchmarkSetup&) = delete;
  BenchmarkSetup& operator=(const BenchmarkSetup&) = delete;
  BenchmarkSetup(BenchmarkSetup&&) = delete;
  BenchmarkSetup& operator=(BenchmarkSetup&&) = delete;

  BenchmarkSetup(
      size_t bufferSize,
      int cudaDev0,
      int cudaDev1,
      int nicDev0,
      int nicDev1,
      LoadBalancingScheme loadBalancingScheme = LoadBalancingScheme::SPRAY) {
    // Setup IbvEndPoint
    sender = std::make_unique<IbvEndPoint>(nicDev0, loadBalancingScheme);
    receiver = std::make_unique<IbvEndPoint>(nicDev1, loadBalancingScheme);
    CHECK_NOTNULL(sender.get());
    CHECK_NOTNULL(receiver.get());

    // Change sender and receiver QP state
    // Query GIDs from both receiver devices
    std::vector<ibv_gid> receiverGids;
    receiverGids.reserve(receiver->devices.size());
    for (const auto& device : receiver->devices) {
      auto gid = device.queryGid(kPortNum, kGidIndex);
      if (!gid) {
        throw std::runtime_error("Failed to query receiver GID");
      }
      receiverGids.push_back(*gid);
    }

    // Query GIDs from both sender devices
    std::vector<ibv_gid> senderGids;
    senderGids.reserve(sender->devices.size());
    for (const auto& device : sender->devices) {
      auto gid = device.queryGid(kPortNum, kGidIndex);
      if (!gid) {
        throw std::runtime_error("Failed to query sender GID");
      }
      senderGids.push_back(*gid);
    }

    // Get business cards
    auto receiverVirtualQpBusinessCard =
        receiver->qp.getVirtualQpBusinessCard();
    auto senderVirtualQpBusinessCard = sender->qp.getVirtualQpBusinessCard();

    // Change QP states
    sender->changeVirtualQpStateToRts(
        receiverGids, receiverVirtualQpBusinessCard);
    receiver->changeVirtualQpStateToRts(
        senderGids, senderVirtualQpBusinessCard);

    // Allocate memory on the sender and receiver side
    ibv_access_flags access = static_cast<ibv_access_flags>(
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ);

    // Check CUDA device availability first
    int deviceCount;
    CHECK_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);

    if (cudaDev0 >= deviceCount || cudaDev1 >= deviceCount) {
      throw std::runtime_error("Required CUDA devices not available");
    }

    CHECK_EQ(cudaSetDevice(cudaDev0), cudaSuccess);
    CHECK_EQ(cudaMalloc(&sendBuffer, bufferSize), cudaSuccess);
    CHECK_NOTNULL(sendBuffer);

    // Register send buffer with all sender PDs
    sendMrs.reserve(sender->pds.size());
    for (size_t i = 0; i < sender->pds.size(); i++) {
      auto sendMrExpected =
          sender->pds.at(i).regMr(sendBuffer, bufferSize, access);
      if (!sendMrExpected) {
        throw std::runtime_error(
            fmt::format("Failed to register send memory region with PD {}", i));
      }
      sendMrs.push_back(std::move(*sendMrExpected));
    }

    CHECK_EQ(cudaSetDevice(cudaDev1), cudaSuccess);
    CHECK_EQ(cudaMalloc(&recvBuffer, bufferSize), cudaSuccess);
    CHECK_NOTNULL(recvBuffer);

    // Register recv buffer with all receiver PDs
    recvMrs.reserve(receiver->pds.size());
    for (size_t i = 0; i < receiver->pds.size(); i++) {
      auto recvMrExpected =
          receiver->pds.at(i).regMr(recvBuffer, bufferSize, access);
      if (!recvMrExpected) {
        throw std::runtime_error(
            fmt::format(
                "Failed to register receive memory region with PD {}", i));
      }
      recvMrs.push_back(std::move(*recvMrExpected));
    }

    // Construct deviceIdToKeys maps for sender and receiver
    for (size_t i = 0; i < sender->devices.size(); i++) {
      int32_t deviceId = sender->devices.at(i).getDeviceId();
      senderDeviceIdToKeys[deviceId] = MemoryRegionKeys{
          .lkey = sendMrs.at(i)->mr()->lkey, .rkey = recvMrs.at(i)->mr()->rkey};
    }

    for (size_t i = 0; i < receiver->devices.size(); i++) {
      int32_t deviceId = receiver->devices.at(i).getDeviceId();
      receiverDeviceIdToKeys[deviceId] = MemoryRegionKeys{
          .lkey = recvMrs.at(i)->mr()->lkey, .rkey = sendMrs.at(i)->mr()->rkey};
    }
  }

  ~BenchmarkSetup() {
    CHECK_EQ(cudaFree(sendBuffer), cudaSuccess);
    CHECK_EQ(cudaFree(recvBuffer), cudaSuccess);
  }

  // Utility function to poll completion queue and wait for completion
  static void pollCqUntilCompletion(
      IbvVirtualCq& cq,
      const std::string& cqName) {
    bool stop = false;
    while (!stop) {
      auto maybeWcsVector = cq.pollCq_v2();
      auto numWc = maybeWcsVector->size();
      if (numWc == 0) {
        continue;
      } else if (numWc == 1) {
        const auto& wc = maybeWcsVector->at(0);
        if (wc.status != IBV_WC_SUCCESS) {
          XLOGF(FATAL, "{} WC failed with status {}", cqName, wc.status);
          return;
        }
        stop = true;
      } else {
        XLOGF(FATAL, "{} got {} wc", cqName, numWc);
      }
    }
  }
};

} // namespace

//------------------------------------------------------------------------------
// Ibverbx Benchmarks
//------------------------------------------------------------------------------

/**
 * Ibverbx virtualQp RdmaWrite benchmark latency
 */

static void BM_Ibverbx_VirtualQp_RdmaRead(benchmark::State& state) {
  const size_t bufferSize = state.range(0);
  const int cudaDev0 = 0;
  const int cudaDev1 = 1;
  const int nicDev0 = 0;
  const int nicDev1 = 1;

  try {
    BenchmarkSetup setup(bufferSize, cudaDev0, cudaDev1, nicDev0, nicDev1);

    // Construct send WRs
    IbvVirtualSendWr sendWr;
    sendWr.wrId = 0;
    sendWr.localAddr = setup.recvBuffer;
    sendWr.length = static_cast<uint32_t>(bufferSize);
    sendWr.remoteAddr = (uint64_t)setup.sendBuffer;
    sendWr.opcode = IBV_WR_RDMA_READ;
    sendWr.sendFlags = IBV_SEND_SIGNALED;
    for (const auto& [deviceId, keys] : setup.receiverDeviceIdToKeys) {
      sendWr.deviceKeys[deviceId] = keys;
    }

    // Benchmark the postSend operation
    for (auto _ : state) {
      setup.receiver->qp.postSend_v2(sendWr);

      // Poll receiver cq until completion
      BenchmarkSetup::pollCqUntilCompletion(setup.receiver->cq, "Receiver");
    }

    // Calculate and report bandwidth using custom counters
    double totalBytes = static_cast<double>(state.iterations()) * bufferSize;
    state.counters["BW_GBps"] =
        benchmark::Counter(totalBytes / 1e9, benchmark::Counter::kIsRate);
  } catch (const std::exception& e) {
    XLOGF(FATAL, "Benchmark setup failed: {}", e.what());
    return;
  }
}

static void BM_Ibverbx_VirtualQp_RdmaWrite(benchmark::State& state) {
  const size_t bufferSize = state.range(0);
  const int cudaDev0 = 0;
  const int cudaDev1 = 1;
  const int nicDev0 = 0;
  const int nicDev1 = 1;

  try {
    BenchmarkSetup setup(bufferSize, cudaDev0, cudaDev1, nicDev0, nicDev1);

    // Construct send WRs
    IbvVirtualSendWr sendWr;
    sendWr.wrId = 0;
    sendWr.localAddr = setup.sendBuffer;
    sendWr.length = static_cast<uint32_t>(bufferSize);
    sendWr.remoteAddr = (uint64_t)setup.recvBuffer;
    sendWr.opcode = IBV_WR_RDMA_WRITE;
    sendWr.sendFlags = IBV_SEND_SIGNALED;
    for (const auto& [deviceId, keys] : setup.senderDeviceIdToKeys) {
      sendWr.deviceKeys[deviceId] = keys;
    }

    // Benchmark the postSend operation
    for (auto _ : state) {
      auto postSendResult = setup.sender->qp.postSend_v2(sendWr);
      BenchmarkSetup::pollCqUntilCompletion(setup.sender->cq, "Sender");
    }

    // Calculate and report bandwidth using custom counters
    double totalBytes = static_cast<double>(state.iterations()) * bufferSize;
    state.counters["BW_GBps"] =
        benchmark::Counter(totalBytes / 1e9, benchmark::Counter::kIsRate);
  } catch (const std::exception& e) {
    XLOGF(FATAL, "[BM_RdmaWrite] Benchmark setup failed: {}", e.what());
    return;
  }
}

static void BM_Ibverbx_VirtualQp_RdmaWriteWithImm(
    benchmark::State& state,
    LoadBalancingScheme loadBalancingScheme) {
  const size_t bufferSize = state.range(0);
  const int cudaDev0 = 0;
  const int cudaDev1 = 1;
  const int nicDev0 = 0;
  const int nicDev1 = 1;

  try {
    BenchmarkSetup setup(
        bufferSize, cudaDev0, cudaDev1, nicDev0, nicDev1, loadBalancingScheme);

    // Construct send WRs
    uint32_t imm_data = static_cast<uint32_t>(bufferSize);
    IbvVirtualSendWr sendWr;
    sendWr.wrId = 0;
    sendWr.localAddr = setup.sendBuffer;
    sendWr.length = static_cast<uint32_t>(bufferSize);
    sendWr.remoteAddr = (uint64_t)setup.recvBuffer;
    sendWr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    sendWr.sendFlags = IBV_SEND_SIGNALED;
    sendWr.immData = imm_data;
    for (const auto& [deviceId, keys] : setup.senderDeviceIdToKeys) {
      sendWr.deviceKeys[deviceId] = keys;
    }

    // Construct recv WRs
    IbvVirtualRecvWr recvWr;
    recvWr.wrId = 0;

    // Benchmark the postSend operation
    for (auto _ : state) {
      setup.receiver->qp.postRecv_v2(recvWr);
      setup.sender->qp.postSend_v2(sendWr);

      // Poll sender and receiver cq until completion
      BenchmarkSetup::pollCqUntilCompletion(setup.sender->cq, "Sender");
      BenchmarkSetup::pollCqUntilCompletion(setup.receiver->cq, "Receiver");
    }

    // Calculate and report bandwidth using custom counters
    double totalBytes = static_cast<double>(state.iterations()) * bufferSize;
    state.counters["BW_GBps"] =
        benchmark::Counter(totalBytes / 1e9, benchmark::Counter::kIsRate);
  } catch (const std::exception& e) {
    XLOGF(FATAL, "Benchmark setup failed: {}", e.what());
    return;
  }
}

// Spray mode benchmark for
static void BM_Ibverbx_VirtualQp_RdmaWriteWithImm_Spray(
    benchmark::State& state) {
  BM_Ibverbx_VirtualQp_RdmaWriteWithImm(state, LoadBalancingScheme::SPRAY);
}

// DQPLB mode benchmark
static void BM_Ibverbx_VirtualQp_RdmaWriteWithImm_Dqplb(
    benchmark::State& state) {
  BM_Ibverbx_VirtualQp_RdmaWriteWithImm(state, LoadBalancingScheme::DQPLB);
}

//------------------------------------------------------------------------------
// Benchmarks
//------------------------------------------------------------------------------

const size_t kMinBufferSize = 8 * 1024; // 8 KB
const size_t kMaxBufferSize = 256 * 1024 * 1024; // 256 MB

BENCHMARK(BM_Ibverbx_VirtualQp_RdmaRead)
    ->RangeMultiplier(2)
    ->Range(kMinBufferSize, kMaxBufferSize)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Ibverbx_VirtualQp_RdmaWrite)
    ->RangeMultiplier(2)
    ->Range(kMinBufferSize, kMaxBufferSize)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Ibverbx_VirtualQp_RdmaWriteWithImm_Spray)
    ->RangeMultiplier(2)
    ->Range(kMinBufferSize, kMaxBufferSize)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Ibverbx_VirtualQp_RdmaWriteWithImm_Dqplb)
    ->RangeMultiplier(2)
    ->Range(kMinBufferSize, kMaxBufferSize)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

// Custom main function to handle initialization
int main(int argc, char** argv) {
  ncclCvarInit();

  // Check if we have multiple CUDA devices for transport benchmarks
  int deviceCount;
  if (cudaGetDeviceCount(&deviceCount) == cudaSuccess) {
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    if (deviceCount < 2) {
      std::cout
          << "Warning: Transport benchmarks require at least 2 CUDA devices"
          << std::endl;
    }
  }

  // Initialize and run benchmark
  ::benchmark::Initialize(&argc, argv);
  folly::init(&argc, &argv);
  ::benchmark::RunSpecifiedBenchmarks();

  // Cleanup
  CHECK_EQ(cudaDeviceReset(), cudaSuccess);

  return 0;
}
