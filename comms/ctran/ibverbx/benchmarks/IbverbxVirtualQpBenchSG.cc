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
constexpr uint32_t kMaxSge = 16; // Increased for scatter-gather support
constexpr uint32_t kNumScatterGatherBuffers = 4; // Number of SG buffers

namespace {

class IbvEndPoint {
 public:
  IbvEndPoint(
      int nicDevId,
      LoadBalancingScheme loadBalancingScheme = LoadBalancingScheme::SPRAY);
  ~IbvEndPoint();
  ibv_qp_init_attr makeIbvQpInitAttr();
  ibv_qp_attr makeQpAttrInit();
  ibv_qp_attr makeQpAttrRtr(ibv_gid remoteGid);
  void changeVirtualQpStateToRts(
      ibv_gid remoteGid,
      const IbvVirtualQpBusinessCard& remoteVirtualQpBusinessCard);

  IbvDevice device;
  IbvPd pd;
  IbvVirtualCq cq;
  IbvVirtualQp qp;
};

IbvEndPoint::IbvEndPoint(int nicDevId, LoadBalancingScheme loadBalancingScheme)
    : device(([nicDevId]() {
        // Initialize ibverbx first
        auto initResult = ibvInit();
        if (!initResult) {
          throw std::runtime_error("ibvInit() failed");
        }

        // TODO: Currently, we use NCCL_IB_HCA to obtain the list of InfiniBand
        // devices. In the future, since Ibverbx is a standalone IB library, it
        // should provide its own interface to enumerate available IB devices
        // and return the device list.
        auto devices =
            IbvDevice::ibvGetDeviceList(NCCL_IB_HCA, NCCL_IB_HCA_PREFIX);
        if (!devices) {
          throw std::runtime_error("Failed to get device list");
        }

        if (devices->empty()) {
          throw std::runtime_error("No InfiniBand devices available");
        }

        if (nicDevId >= static_cast<int>(devices->size())) {
          throw std::out_of_range("nicDevId out of range");
        }
        auto selectedDevice = std::move(devices->at(nicDevId));
        return selectedDevice;
      })()),
      pd(([this]() {
        auto maybePd = device.allocPd();
        if (!maybePd) {
          throw std::runtime_error("Failed to allocate protection domain");
        }
        return std::move(*maybePd);
      })()),
      cq(([this]() {
        auto maybeVirtualCq =
            device.createVirtualCq(32768, nullptr, nullptr, 0);
        if (!maybeVirtualCq) {
          throw std::runtime_error("Failed to create virtual completion queue");
        }
        return std::move(*maybeVirtualCq);
      })()),
      qp([this, loadBalancingScheme]() {
        auto initAttr = makeIbvQpInitAttr();

        auto maybeVirtualQp = pd.createVirtualQp(
            kTotalQps,
            &initAttr,
            &cq,
            &cq,
            kMaxMsgCntPerQp,
            kMaxMsgSize,
            loadBalancingScheme);
        if (!maybeVirtualQp) {
          throw std::runtime_error("Failed to create virtual queue pair");
        }
        return std::move(*maybeVirtualQp);
      }()) {}

IbvEndPoint::~IbvEndPoint() = default;

// helper functions
ibv_qp_init_attr IbvEndPoint::makeIbvQpInitAttr() {
  ibv_qp_init_attr initAttr{};
  memset(&initAttr, 0, sizeof(ibv_qp_init_attr));
  initAttr.send_cq = cq.getPhysicalCqsRef().at(0).cq();
  initAttr.recv_cq = cq.getPhysicalCqsRef().at(0).cq();
  initAttr.qp_type = IBV_QPT_RC; // Reliable Connection
  initAttr.sq_sig_all = 0;
  initAttr.cap.max_send_wr = kMaxOutstandingWrs; // maximum outstanding send WRs
  initAttr.cap.max_recv_wr = kMaxOutstandingWrs; // maximum outstanding recv WRs
  initAttr.cap.max_send_sge = kMaxSge; // Increased for scatter-gather
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
    ibv_gid remoteGid,
    const IbvVirtualQpBusinessCard& remoteVirtualQpBusinessCard) {
  {
    // change QP group state to INIT
    auto qpAttr = makeQpAttrInit();
    auto result = qp.modifyVirtualQp(
        &qpAttr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (!result) {
      throw std::runtime_error("Failed to modify virtual QP to INIT state");
    }
  }
  {
    // change QP group state to RTR
    auto qpAttr = makeQpAttrRtr(remoteGid);
    auto result = qp.modifyVirtualQp(
        &qpAttr,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER,
        remoteVirtualQpBusinessCard);
    if (!result) {
      throw std::runtime_error("Failed to modify virtual QP to RTR state");
    }
  }
  {
    // change QP group state to RTS
    auto qpAttr = makeQpAttrRts();
    auto result = qp.modifyVirtualQp(
        &qpAttr,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
            IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (!result) {
      throw std::runtime_error("Failed to modify virtual QP to RTS state");
    }
  }
}

struct ScatterGatherBenchmarkSetup {
  std::unique_ptr<IbvEndPoint> sender;
  std::unique_ptr<IbvEndPoint> receiver;

  // Scatter-gather source buffers (4 buffers)
  std::vector<void*> sendBuffers;
  std::vector<std::optional<IbvMr>> sendMrs;

  // Contiguous destination buffer
  void* recvBuffer{};
  std::optional<IbvMr> recvMr;

  // Per-buffer size (each of the 4 buffers has this size)
  size_t perBufferSize{0};
  // Total size across all scatter-gather buffers
  size_t totalSize{0};

  // Disable copy and move operations for RAII resource management
  ScatterGatherBenchmarkSetup(const ScatterGatherBenchmarkSetup&) = delete;
  ScatterGatherBenchmarkSetup& operator=(const ScatterGatherBenchmarkSetup&) =
      delete;
  ScatterGatherBenchmarkSetup(ScatterGatherBenchmarkSetup&&) = delete;
  ScatterGatherBenchmarkSetup& operator=(ScatterGatherBenchmarkSetup&&) =
      delete;

  ScatterGatherBenchmarkSetup(
      size_t bufferSizePerSg,
      int cudaDev0,
      int cudaDev1,
      int nicDev0,
      int nicDev1,
      LoadBalancingScheme loadBalancingScheme = LoadBalancingScheme::SPRAY)
      : perBufferSize(bufferSizePerSg),
        totalSize(bufferSizePerSg * kNumScatterGatherBuffers) {
    // Setup IbvEndPoint
    sender = std::make_unique<IbvEndPoint>(nicDev0, loadBalancingScheme);
    receiver = std::make_unique<IbvEndPoint>(nicDev1, loadBalancingScheme);
    CHECK_NOTNULL(sender.get());
    CHECK_NOTNULL(receiver.get());

    // Change sender and receiver QP state
    auto receiverGid = receiver->device.queryGid(kPortNum, kGidIndex);
    if (!receiverGid) {
      throw std::runtime_error("Failed to query receiver GID");
    }
    auto receiverVirtualQpBusinessCard =
        receiver->qp.getVirtualQpBusinessCard();
    sender->changeVirtualQpStateToRts(
        *receiverGid, receiverVirtualQpBusinessCard);

    auto senderGid = sender->device.queryGid(kPortNum, kGidIndex);
    if (!senderGid) {
      throw std::runtime_error("Failed to query sender GID");
    }
    auto senderVirtualQpBusinessCard = sender->qp.getVirtualQpBusinessCard();
    receiver->changeVirtualQpStateToRts(
        *senderGid, senderVirtualQpBusinessCard);

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

    // Allocate scatter-gather source buffers on sender GPU
    CHECK_EQ(cudaSetDevice(cudaDev0), cudaSuccess);
    sendBuffers.resize(kNumScatterGatherBuffers);
    sendMrs.resize(kNumScatterGatherBuffers);

    for (uint32_t i = 0; i < kNumScatterGatherBuffers; ++i) {
      CHECK_EQ(cudaMalloc(&sendBuffers[i], perBufferSize), cudaSuccess);
      CHECK_NOTNULL(sendBuffers[i]);
      auto sendMrExpected =
          sender->pd.regMr(sendBuffers[i], perBufferSize, access);
      if (!sendMrExpected) {
        throw std::runtime_error(
            "Failed to register send memory region for buffer " +
            std::to_string(i));
      }
      sendMrs[i] = std::move(*sendMrExpected);
    }

    // Allocate contiguous destination buffer on receiver GPU
    CHECK_EQ(cudaSetDevice(cudaDev1), cudaSuccess);
    CHECK_EQ(cudaMalloc(&recvBuffer, totalSize), cudaSuccess);
    CHECK_NOTNULL(recvBuffer);
    auto recvMrExpected = receiver->pd.regMr(recvBuffer, totalSize, access);
    if (!recvMrExpected) {
      throw std::runtime_error("Failed to register receive memory region");
    }
    recvMr = std::move(*recvMrExpected);
  }

  ~ScatterGatherBenchmarkSetup() {
    for (auto& buf : sendBuffers) {
      if (buf) {
        CHECK_EQ(cudaFree(buf), cudaSuccess);
      }
    }
    if (recvBuffer) {
      CHECK_EQ(cudaFree(recvBuffer), cudaSuccess);
    }
  }

  // Build scatter-gather list for the send buffers
  std::vector<ibv_sge> buildSgList() const {
    std::vector<ibv_sge> sgList;
    sgList.reserve(kNumScatterGatherBuffers);
    for (uint32_t i = 0; i < kNumScatterGatherBuffers; ++i) {
      ibv_sge sge = {
          .addr = reinterpret_cast<uint64_t>(sendBuffers[i]),
          .length = static_cast<uint32_t>(perBufferSize),
          .lkey = sendMrs[i]->mr()->lkey};
      sgList.push_back(sge);
    }
    return sgList;
  }

  // Build per-buffer device keys for scatter-gather operations
  std::vector<ScatterGatherBufferKeys> buildPerBufferKeys() const {
    std::vector<ScatterGatherBufferKeys> perBufferKeys;
    perBufferKeys.reserve(kNumScatterGatherBuffers);

    // Get the device ID from the sender's QP
    int32_t deviceId = sender->qp.getQpsRef().at(0).getDeviceId();

    for (uint32_t i = 0; i < kNumScatterGatherBuffers; ++i) {
      ScatterGatherBufferKeys bufferKeys;
      MemoryRegionKeys keys;
      keys.lkey = sendMrs[i]->mr()->lkey;
      keys.rkey = sendMrs[i]->mr()->rkey;
      bufferKeys.deviceIdToKeys[deviceId] = keys;
      perBufferKeys.push_back(std::move(bufferKeys));
    }
    return perBufferKeys;
  }

  // Utility function to poll completion queue and wait for completion
  static void pollCqUntilCompletion(
      IbvVirtualCq& cq,
      const std::string& cqName) {
    int numEntries = 1;
    bool stop = false;
    while (!stop) {
      auto maybeWcsVector = cq.pollCq(numEntries);
      auto numWc = maybeWcsVector->size();
      if (numWc == 0) {
        // CQ empty, retry
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
// Scatter-Gather Ibverbx Benchmarks
//------------------------------------------------------------------------------

/**
 * Ibverbx virtualQp Scatter-Gather RDMA Write benchmark
 * Uses 4 non-contiguous source buffers scattered across memory
 * that are gathered into a contiguous destination buffer
 */

static void BM_Ibverbx_VirtualQp_ScatterGather_RdmaWrite(
    benchmark::State& state) {
  // state.range(0) is the total size across all scatter-gather buffers
  const size_t totalSize = state.range(0);
  const size_t bufferSizePerSg = totalSize / kNumScatterGatherBuffers;
  const int cudaDev0 = 0;
  const int cudaDev1 = 1;
  const int nicDev0 = 0;
  const int nicDev1 = 1;

  try {
    ScatterGatherBenchmarkSetup setup(
        bufferSizePerSg, cudaDev0, cudaDev1, nicDev0, nicDev1);

    // Build scatter-gather list
    auto sgList = setup.buildSgList();
    auto perBufferKeys = setup.buildPerBufferKeys();

    // Construct send WR with scatter-gather list
    int wr_id = 0;
    ibv_send_wr sendWr = {
        .wr_id = static_cast<uint64_t>(wr_id),
        .next = nullptr,
        .sg_list = sgList.data(),
        .num_sge = static_cast<int>(sgList.size()),
        .opcode = IBV_WR_RDMA_WRITE,
        .send_flags = IBV_SEND_SIGNALED};
    sendWr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(setup.recvBuffer);
    sendWr.wr.rdma.rkey = setup.recvMr->mr()->rkey;
    ibv_send_wr sendWrBad{};

    // Benchmark the postSendScatterGather operation
    for (auto _ : state) {
      setup.sender->qp.postSendScatterGather(
          &sendWr, &sendWrBad, perBufferKeys);

      // Poll sender cq until completion
      ScatterGatherBenchmarkSetup::pollCqUntilCompletion(
          setup.sender->cq, "Sender");
    }

    // Calculate and report bandwidth using custom counters
    double totalBytes = static_cast<double>(state.iterations()) * totalSize;
    state.counters["BW_GBps"] =
        benchmark::Counter(totalBytes / 1e9, benchmark::Counter::kIsRate);
    state.counters["PerBufSize"] = bufferSizePerSg;
    state.counters["NumBufs"] = kNumScatterGatherBuffers;
  } catch (const std::exception& e) {
    XLOGF(FATAL, "Benchmark setup failed: {}", e.what());
    return;
  }
}

static void BM_Ibverbx_VirtualQp_ScatterGather_RdmaWriteWithImm(
    benchmark::State& state,
    LoadBalancingScheme loadBalancingScheme) {
  // state.range(0) is the total size across all scatter-gather buffers
  const size_t totalSize = state.range(0);
  const size_t bufferSizePerSg = totalSize / kNumScatterGatherBuffers;
  const int cudaDev0 = 0;
  const int cudaDev1 = 1;
  const int nicDev0 = 0;
  const int nicDev1 = 1;

  try {
    ScatterGatherBenchmarkSetup setup(
        bufferSizePerSg,
        cudaDev0,
        cudaDev1,
        nicDev0,
        nicDev1,
        loadBalancingScheme);

    // Build scatter-gather list
    auto sgList = setup.buildSgList();
    auto perBufferKeys = setup.buildPerBufferKeys();

    // Construct send WR with scatter-gather list
    int wr_id = 0;
    uint32_t imm_data = static_cast<uint32_t>(totalSize);
    ibv_send_wr sendWr = {
        .wr_id = static_cast<uint64_t>(wr_id),
        .next = nullptr,
        .sg_list = sgList.data(),
        .num_sge = static_cast<int>(sgList.size()),
        .opcode = IBV_WR_RDMA_WRITE_WITH_IMM,
        .send_flags = IBV_SEND_SIGNALED};
    sendWr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(setup.recvBuffer);
    sendWr.wr.rdma.rkey = setup.recvMr->mr()->rkey;
    sendWr.imm_data = imm_data;
    ibv_send_wr sendWrBad{};

    // Construct recv WR
    ibv_sge recvSgList = {};
    ibv_recv_wr recvWr = {
        .wr_id = static_cast<uint64_t>(wr_id),
        .sg_list = &recvSgList,
        .num_sge = 0};
    ibv_recv_wr recvWrBad{};

    // Benchmark the postSendScatterGather operation
    for (auto _ : state) {
      setup.receiver->qp.postRecv(&recvWr, &recvWrBad);
      setup.sender->qp.postSendScatterGather(
          &sendWr, &sendWrBad, perBufferKeys);

      // Poll sender and receiver cq until completion
      ScatterGatherBenchmarkSetup::pollCqUntilCompletion(
          setup.sender->cq, "Sender");
      ScatterGatherBenchmarkSetup::pollCqUntilCompletion(
          setup.receiver->cq, "Receiver");
    }

    // Calculate and report bandwidth using custom counters
    double totalBytes = static_cast<double>(state.iterations()) * totalSize;
    state.counters["BW_GBps"] =
        benchmark::Counter(totalBytes / 1e9, benchmark::Counter::kIsRate);
    state.counters["PerBufSize"] = bufferSizePerSg;
    state.counters["NumBufs"] = kNumScatterGatherBuffers;
  } catch (const std::exception& e) {
    XLOGF(FATAL, "Benchmark setup failed: {}", e.what());
    return;
  }
}

// Spray mode benchmark for scatter-gather
static void BM_Ibverbx_VirtualQp_ScatterGather_RdmaWriteWithImm_Spray(
    benchmark::State& state) {
  BM_Ibverbx_VirtualQp_ScatterGather_RdmaWriteWithImm(
      state, LoadBalancingScheme::SPRAY);
}

// DQPLB mode benchmark for scatter-gather
static void BM_Ibverbx_VirtualQp_ScatterGather_RdmaWriteWithImm_Dqplb(
    benchmark::State& state) {
  BM_Ibverbx_VirtualQp_ScatterGather_RdmaWriteWithImm(
      state, LoadBalancingScheme::DQPLB);
}

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

// Total transfer sizes: min 32KB (4 * 8KB), max 1024MB (4 * 256MB)
// Note: Per-buffer size = total size / 4 (number of SG buffers)
const size_t kMinTotalSize = 32 * 1024; // 32 KB total (8 KB per buffer)
const size_t kMaxTotalSize =
    1024 * 1024 * 1024; // 1024 MB total (256 MB per buffer)

BENCHMARK(BM_Ibverbx_VirtualQp_ScatterGather_RdmaWrite)
    ->RangeMultiplier(2)
    ->Range(kMinTotalSize, kMaxTotalSize)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Ibverbx_VirtualQp_ScatterGather_RdmaWriteWithImm_Spray)
    ->RangeMultiplier(2)
    ->Range(kMinTotalSize, kMaxTotalSize)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Ibverbx_VirtualQp_ScatterGather_RdmaWriteWithImm_Dqplb)
    ->RangeMultiplier(2)
    ->Range(kMinTotalSize, kMaxTotalSize)
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

  std::cout << "Running Scatter-Gather benchmarks with "
            << kNumScatterGatherBuffers << " source buffers" << std::endl;
  std::cout << "Per-buffer size range: "
            << kMinTotalSize / kNumScatterGatherBuffers / 1024 << " KB to "
            << kMaxTotalSize / kNumScatterGatherBuffers / (1024 * 1024) << " MB"
            << std::endl;
  std::cout << "Total transfer size range: " << kMinTotalSize / 1024
            << " KB to " << kMaxTotalSize / (1024 * 1024) << " MB" << std::endl;

  // Initialize and run benchmark
  ::benchmark::Initialize(&argc, argv);
  folly::init(&argc, &argv);
  ::benchmark::RunSpecifiedBenchmarks();

  // Cleanup
  CHECK_EQ(cudaDeviceReset(), cudaSuccess);

  return 0;
}
