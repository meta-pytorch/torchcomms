// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <dirent.h>
#include <mpi.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ibverbx;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

//------------------------------------------------------------------------------
// Real-time Priority Helper
//------------------------------------------------------------------------------

// Set real-time priority for ALL threads in the current process
static void setRealtimePriorityForAllThreads([[maybe_unused]] int rank) {
  // Read all thread IDs from /proc/self/task/
  std::vector<pid_t> tids;
  DIR* dir = opendir("/proc/self/task");
  if (dir) {
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
      if (entry->d_name[0] != '.') {
        pid_t tid = static_cast<pid_t>(std::stoi(entry->d_name));
        tids.push_back(tid);
      }
    }
    closedir(dir);
  }

  // Set SCHED_FIFO priority 99 for all threads
  struct sched_param param;
  param.sched_priority = 99;

  for (pid_t tid : tids) {
    sched_setscheduler(tid, SCHED_FIFO, &param);
  }
}

static void setRealtimePriority(int rank) {
  struct sched_param param;
  param.sched_priority = 99; // Highest FIFO priority

  sched_setscheduler(0, SCHED_FIFO, &param);
  setRealtimePriorityForAllThreads(rank);
}

//------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------

constexpr uint8_t kPortNum = 1;
constexpr int kGidIndex = 3;
constexpr uint32_t kTotalQps = 1;
constexpr uint32_t kMaxMsgCntPerQp = 128;
constexpr uint32_t kMaxMsgSize = 524288;
constexpr uint32_t kMaxOutstandingWrs = 128;
constexpr uint32_t kMaxSge = 1;
constexpr int kDefaultIterations = 10000;
constexpr int kWarmupIterations = 100;
constexpr uint32_t kVirtualCqSize = 32768;
constexpr uint32_t kMaxInlineData = 220;

// Large message test constants (for messages >= 1MB)
constexpr size_t kLargeMsgThreshold = 1024 * 1024; // 1MB
constexpr int kLargeMsgIterations = 2000;

// Extra large message test constants (for messages >= 64MB)
constexpr size_t kExtraLargeMsgThreshold = 64 * 1024 * 1024; // 64MB
constexpr int kExtraLargeMsgIterations = 500;

// Bandwidth test constants (following perftest patterns)
constexpr int kBwTxDepth = 128; // Max outstanding WRs for pipelining
constexpr int kBwIterations = 5000; // Number of iterations for BW test

// Message size range for benchmarks
constexpr size_t kMinBenchmarkMsgSize = 1; // 1 byte
constexpr size_t kMaxBenchmarkMsgSize = 256 * 1024 * 1024; // 256 MB

//------------------------------------------------------------------------------
// Helper Functions
//------------------------------------------------------------------------------

// Get iteration count based on message size
static int getIterationsForMsgSize(size_t msgSize) {
  if (msgSize >= kExtraLargeMsgThreshold) {
    return kExtraLargeMsgIterations;
  }
  if (msgSize >= kLargeMsgThreshold) {
    return kLargeMsgIterations;
  }
  return kDefaultIterations;
}

// Generate standard message sizes for benchmarking (powers of 2)
static std::vector<size_t> generateMessageSizes(
    size_t minSize = kMinBenchmarkMsgSize,
    size_t maxSize = kMaxBenchmarkMsgSize) {
  std::vector<size_t> sizes;
  for (size_t size = minSize; size <= maxSize; size *= 2) {
    sizes.push_back(size);
  }
  return sizes;
}

//------------------------------------------------------------------------------
// Exchange Info for cross-host connection
//------------------------------------------------------------------------------

struct ExchangeInfo {
  // GID for address handle
  uint8_t gid[16];
  // Serialized IbvVirtualQpBusinessCard
  char businessCardJson[1024];
  size_t businessCardJsonLen;
  // Memory region info for RDMA
  uint32_t rkey;
  uint64_t addr; // Remote buffer address (recv area)
};

//------------------------------------------------------------------------------
// IbvEndPoint for VirtualQp
//------------------------------------------------------------------------------

class IbvEndPoint {
 public:
  explicit IbvEndPoint(
      int nicDevId,
      LoadBalancingScheme loadBalancingScheme = LoadBalancingScheme::SPRAY);
  ~IbvEndPoint();

  ibv_qp_init_attr makeIbvQpInitAttr();
  ibv_qp_attr makeQpAttrInit();
  ibv_qp_attr makeQpAttrRtr(ibv_gid remoteGid);
  static ibv_qp_attr makeQpAttrRts();

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
        auto initResult = ibvInit();
        if (!initResult) {
          throw std::runtime_error("ibvInit() failed");
        }

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
            device.createVirtualCq(kVirtualCqSize, nullptr, nullptr, 0);
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

ibv_qp_init_attr IbvEndPoint::makeIbvQpInitAttr() {
  ibv_qp_init_attr initAttr{};
  initAttr.send_cq = cq.getPhysicalCqsRef().at(0).cq();
  initAttr.recv_cq = cq.getPhysicalCqsRef().at(0).cq();
  initAttr.qp_type = IBV_QPT_RC;
  initAttr.sq_sig_all = 0;
  initAttr.cap.max_send_wr = kMaxOutstandingWrs;
  initAttr.cap.max_recv_wr = kMaxOutstandingWrs;
  initAttr.cap.max_send_sge = kMaxSge;
  initAttr.cap.max_recv_sge = kMaxSge;
  initAttr.cap.max_inline_data = kMaxInlineData;
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

ibv_qp_attr IbvEndPoint::makeQpAttrRts() {
  const uint8_t kTimeout = 14;
  const uint8_t kRetryCnt = 7;

  struct ibv_qp_attr qpAttr{};

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
    auto qpAttr = makeQpAttrInit();
    auto result = qp.modifyVirtualQp(
        &qpAttr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (!result) {
      throw std::runtime_error("Failed to modify virtual QP to INIT state");
    }
  }
  {
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

//------------------------------------------------------------------------------
// Benchmark Context
//------------------------------------------------------------------------------

struct BenchmarkContext {
  std::unique_ptr<IbvEndPoint> endpoint;
  void* buffer{nullptr};
  size_t bufferSize{0};
  size_t msgSize{0};
  std::optional<IbvMr> mr;

  // Buffer layout: [send_area: msgSize][recv_area: msgSize]
  char* sendBuf{nullptr};
  volatile char* recvBuf{nullptr};

  // Pre-constructed work requests
  ibv_sge sge{};
  ibv_send_wr sendWr{};
  ibv_send_wr badSendWr{};

  ~BenchmarkContext() {
    if (buffer) {
      free(buffer);
    }
  }
};

static std::unique_ptr<BenchmarkContext> createBenchmarkContext(
    int nicDevId,
    size_t msgSize,
    LoadBalancingScheme loadBalancingScheme = LoadBalancingScheme::SPRAY) {
  auto ctx = std::make_unique<BenchmarkContext>();

  size_t totalSize = 2 * msgSize;
  ctx->bufferSize = totalSize;
  ctx->msgSize = msgSize;

  ctx->endpoint = std::make_unique<IbvEndPoint>(nicDevId, loadBalancingScheme);

  if (posix_memalign(&ctx->buffer, 4096, totalSize) != 0) {
    throw std::runtime_error("Failed to allocate host memory");
  }
  memset(ctx->buffer, 0, totalSize);

  ctx->sendBuf = static_cast<char*>(ctx->buffer);
  ctx->recvBuf = static_cast<char*>(ctx->buffer) + msgSize;

  ibv_access_flags access = static_cast<ibv_access_flags>(
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
      IBV_ACCESS_REMOTE_READ);

  auto mrExpected = ctx->endpoint->pd.regMr(ctx->buffer, totalSize, access);
  if (!mrExpected) {
    throw std::runtime_error("Failed to register memory region");
  }
  ctx->mr = std::move(*mrExpected);

  return ctx;
}

static ExchangeInfo getLocalExchangeInfo(BenchmarkContext* ctx) {
  ExchangeInfo info{};

  auto gidResult = ctx->endpoint->device.queryGid(kPortNum, kGidIndex);
  if (!gidResult) {
    throw std::runtime_error("Failed to query GID");
  }
  memcpy(info.gid, &(*gidResult), 16);

  auto businessCard = ctx->endpoint->qp.getVirtualQpBusinessCard();
  std::string jsonStr = businessCard.serialize();
  if (jsonStr.size() >= sizeof(info.businessCardJson)) {
    throw std::runtime_error("Business card JSON too large");
  }
  memcpy(info.businessCardJson, jsonStr.c_str(), jsonStr.size());
  info.businessCardJsonLen = jsonStr.size();

  info.rkey = ctx->mr->mr()->rkey;
  info.addr = reinterpret_cast<uint64_t>(ctx->recvBuf);

  return info;
}

static void initSendWr(
    BenchmarkContext* ctx,
    size_t msgSize,
    const ExchangeInfo& remoteInfo) {
  ctx->sge.addr = reinterpret_cast<uint64_t>(ctx->sendBuf);
  ctx->sge.length = static_cast<uint32_t>(msgSize);
  ctx->sge.lkey = ctx->mr->mr()->lkey;

  ctx->sendWr.wr_id = 0;
  ctx->sendWr.next = nullptr;
  ctx->sendWr.sg_list = &ctx->sge;
  ctx->sendWr.num_sge = 1;
  ctx->sendWr.opcode = IBV_WR_RDMA_WRITE;
  ctx->sendWr.send_flags = IBV_SEND_SIGNALED;
  if (msgSize <= kMaxInlineData) {
    ctx->sendWr.send_flags |= IBV_SEND_INLINE;
  }
  ctx->sendWr.wr.rdma.remote_addr = remoteInfo.addr;
  ctx->sendWr.wr.rdma.rkey = remoteInfo.rkey;
}

// Exchange connection info with peer and establish QP connection
static ExchangeInfo exchangeAndConnect(BenchmarkContext* ctx, int peerRank) {
  auto localInfo = getLocalExchangeInfo(ctx);

  ExchangeInfo remoteInfo{};
  MPI_CHECK(MPI_Sendrecv(
      &localInfo,
      sizeof(ExchangeInfo),
      MPI_BYTE,
      peerRank,
      0,
      &remoteInfo,
      sizeof(ExchangeInfo),
      MPI_BYTE,
      peerRank,
      0,
      MPI_COMM_WORLD,
      MPI_STATUS_IGNORE));

  std::string remoteJsonStr(
      remoteInfo.businessCardJson, remoteInfo.businessCardJsonLen);
  auto remoteBusinessCardResult =
      IbvVirtualQpBusinessCard::deserialize(remoteJsonStr);
  if (!remoteBusinessCardResult) {
    throw std::runtime_error("Failed to deserialize remote business card");
  }

  ibv_gid remoteGid{};
  memcpy(&remoteGid, remoteInfo.gid, 16);
  ctx->endpoint->changeVirtualQpStateToRts(
      remoteGid, *remoteBusinessCardResult);

  return remoteInfo;
}

//------------------------------------------------------------------------------
// Poll completion queue helper
//------------------------------------------------------------------------------

static void pollCqUntilCompletion(IbvVirtualCq& cq) {
  while (true) {
    auto maybeWcs = cq.pollCq(1);
    if (maybeWcs && !maybeWcs->empty()) {
      const auto& wc = maybeWcs->at(0);
      if (wc.status != IBV_WC_SUCCESS) {
        XLOGF(ERR, "WC failed with status {}", wc.status);
        throw std::runtime_error("Work completion failed");
      }
      break;
    }
  }
}

//------------------------------------------------------------------------------
// Benchmark Result
//------------------------------------------------------------------------------

struct BenchmarkResult {
  std::string testName;
  size_t messageSize{};
  double medianUs{};
  double avgUs{};
  double minUs{};
  double maxUs{};
  double p99Us{};
  double bandwidthGBps{};
  double msgRateMpps{}; // Message rate in Million packets per second
};

//------------------------------------------------------------------------------
// Test Fixture
//------------------------------------------------------------------------------

class IbverbxVirtualQpBenchmarkFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    ncclCvarInit();
  }

  void TearDown() override {
    MpiBaseTestFixture::TearDown();
  }

  // Run ping-pong benchmark for sender (rank 0)
  BenchmarkResult runSenderBenchmark(
      BenchmarkContext* ctx,
      int iterations,
      const std::string& testName) {
    char* postBuf = ctx->sendBuf + ctx->msgSize - 1;
    volatile char* pollBuf = ctx->recvBuf + ctx->msgSize - 1;

    XLOGF(INFO, "[Sender] Running ping-pong benchmark: {}", testName);

    *postBuf = 0;
    *pollBuf = 0;

    uint64_t scnt = 0;
    uint64_t rcnt = 0;

    // Warmup
    for (int i = 0; i < kWarmupIterations; ++i) {
      *postBuf = static_cast<char>(++scnt);
      ctx->endpoint->qp.postSend(&ctx->sendWr, &ctx->badSendWr);
      pollCqUntilCompletion(ctx->endpoint->cq);

      ++rcnt;
      while (*pollBuf != static_cast<char>(rcnt)) {
      }
    }

    XLOG(INFO) << "[Sender] Warmup complete, starting measurement...";

    // Benchmark
    using Clock = std::chrono::high_resolution_clock;
    std::vector<double> deltasUs(iterations);

    for (int i = 0; i < iterations; ++i) {
      auto start = Clock::now();

      *postBuf = static_cast<char>(++scnt);
      ctx->endpoint->qp.postSend(&ctx->sendWr, &ctx->badSendWr);
      pollCqUntilCompletion(ctx->endpoint->cq);

      ++rcnt;
      while (*pollBuf != static_cast<char>(rcnt)) {
      }

      auto end = Clock::now();
      deltasUs[i] =
          std::chrono::duration<double, std::micro>(end - start).count();
    }

    std::sort(deltasUs.begin(), deltasUs.end());

    double sum = 0;
    for (auto d : deltasUs) {
      sum += d;
    }

    // Report HALF RTT (same as perftest)
    BenchmarkResult result;
    result.testName = testName;
    result.messageSize = ctx->msgSize;
    result.medianUs = deltasUs[iterations / 2] / 2.0;
    result.avgUs = (sum / static_cast<double>(iterations)) / 2.0;
    result.minUs = deltasUs[0] / 2.0;
    result.maxUs = deltasUs[iterations - 1] / 2.0;
    result.p99Us = deltasUs[static_cast<int>(iterations * 0.99)] / 2.0;

    return result;
  }

  // Run ping-pong benchmark for receiver (rank 1)
  void runReceiverBenchmark(
      BenchmarkContext* ctx,
      int iterations,
      const std::string& testName) {
    char* postBuf = ctx->sendBuf + ctx->msgSize - 1;
    volatile char* pollBuf = ctx->recvBuf + ctx->msgSize - 1;

    XLOGF(INFO, "[Receiver] Running ping-pong benchmark: {}", testName);

    *postBuf = 0;
    *pollBuf = 0;

    uint64_t scnt = 0;
    uint64_t rcnt = 0;

    int totalIterations = kWarmupIterations + iterations;

    for (int i = 0; i < totalIterations; ++i) {
      ++rcnt;
      while (*pollBuf != static_cast<char>(rcnt)) {
      }

      *postBuf = static_cast<char>(++scnt);
      ctx->endpoint->qp.postSend(&ctx->sendWr, &ctx->badSendWr);
      pollCqUntilCompletion(ctx->endpoint->cq);
    }

    XLOGF(INFO, "[Receiver] Completed {} iterations", iterations);
  }

  void printResultsTable(
      const std::vector<BenchmarkResult>& results,
      const std::string& opType = "RDMA_WRITE") {
    if (globalRank != 0) {
      return;
    }

    std::stringstream ss;
    ss << "\n";
    ss << "==========================================================================\n";
    ss << "    IbverbxVirtualQp Cross-Host Latency Benchmark Results ("
       << opType << ")\n";
    ss << "==========================================================================\n";
    ss << std::left << std::setw(15) << "Test Name" << std::right
       << std::setw(12) << "Msg Size" << std::right << std::setw(12) << "Median"
       << std::right << std::setw(12) << "Average" << std::right
       << std::setw(12) << "Min" << std::right << std::setw(12) << "Max"
       << std::right << std::setw(12) << "P99\n";
    ss << std::left << std::setw(15) << "" << std::right << std::setw(12) << ""
       << std::right << std::setw(12) << "(us)" << std::right << std::setw(12)
       << "(us)" << std::right << std::setw(12) << "(us)" << std::right
       << std::setw(12) << "(us)" << std::right << std::setw(12) << "(us)\n";
    ss << "--------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      std::string msgSize = formatSize(r.messageSize);

      ss << std::left << std::setw(15) << r.testName << std::right
         << std::setw(12) << msgSize << std::right << std::setw(12)
         << std::fixed << std::setprecision(2) << r.medianUs << std::right
         << std::setw(12) << std::fixed << std::setprecision(2) << r.avgUs
         << std::right << std::setw(12) << std::fixed << std::setprecision(2)
         << r.minUs << std::right << std::setw(12) << std::fixed
         << std::setprecision(2) << r.maxUs << std::right << std::setw(12)
         << std::fixed << std::setprecision(2) << r.p99Us << "\n";
    }
    ss << "==========================================================================\n\n";

    std::cout << ss.str() << std::flush;
  }

  void printBandwidthResultsTable(
      const std::vector<BenchmarkResult>& results,
      const std::string& opType = "RDMA_WRITE") {
    if (globalRank != 0) {
      return;
    }

    std::stringstream ss;
    ss << "\n";
    ss << "==========================================================================\n";
    ss << "    IbverbxVirtualQp Cross-Host Bandwidth Benchmark (" << opType
       << ")\n";
    ss << "==========================================================================\n";
    ss << std::left << std::setw(15) << "Test Name" << std::right
       << std::setw(12) << "Msg Size" << std::right << std::setw(18)
       << "Bandwidth" << std::right << std::setw(15) << "Msg Rate\n";
    ss << "--------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      std::string msgSize = formatSize(r.messageSize);
      std::string bw = formatBandwidth(r.bandwidthGBps);
      std::string msgRate = formatMsgRate(r.msgRateMpps);

      ss << std::left << std::setw(15) << r.testName << std::right
         << std::setw(12) << msgSize << std::right << std::setw(18) << bw
         << std::right << std::setw(15) << msgRate << "\n";
    }
    ss << "==========================================================================\n\n";

    std::cout << ss.str() << std::flush;
  }

  std::string formatSize(size_t bytes) {
    std::stringstream ss;
    if (bytes >= 1024 * 1024 * 1024) {
      ss << std::fixed << std::setprecision(0)
         << (bytes / (1024.0 * 1024.0 * 1024.0)) << "GB";
    } else if (bytes >= 1024 * 1024) {
      ss << std::fixed << std::setprecision(0) << (bytes / (1024.0 * 1024.0))
         << "MB";
    } else if (bytes >= 1024) {
      ss << std::fixed << std::setprecision(0) << (bytes / 1024.0) << "KB";
    } else {
      ss << bytes << "B";
    }
    return ss.str();
  }

  std::string formatBandwidth(double gbps) {
    std::stringstream ss;
    if (gbps >= 1.0) {
      ss << std::fixed << std::setprecision(2) << gbps << " GBps";
    } else if (gbps >= 0.001) {
      ss << std::fixed << std::setprecision(2) << (gbps * 1000.0) << " MBps";
    } else if (gbps >= 0.000001) {
      ss << std::fixed << std::setprecision(2) << (gbps * 1000000.0) << " KBps";
    } else {
      ss << std::fixed << std::setprecision(2) << (gbps * 1000000000.0)
         << " Bps";
    }
    return ss.str();
  }

  std::string formatMsgRate(double mpps) {
    std::stringstream ss;
    if (mpps >= 1.0) {
      ss << std::fixed << std::setprecision(2) << mpps << " Mpps";
    } else if (mpps >= 0.001) {
      ss << std::fixed << std::setprecision(2) << (mpps * 1000.0) << " Kpps";
    } else {
      ss << std::fixed << std::setprecision(2) << (mpps * 1000000.0) << " pps";
    }
    return ss.str();
  }
};

//------------------------------------------------------------------------------
// Test Cases
//------------------------------------------------------------------------------

TEST_F(
    IbverbxVirtualQpBenchmarkFixture,
    CrossHostPingPongHalfRttLatency_RdmaWrite) {
  // Only test with 2 ranks (one per host)
  if (numRanks != 2) {
    XLOGF(INFO, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;
  int nicDevId = globalRank; // Rank 0 uses NIC 0, Rank 1 uses NIC 1

  setRealtimePriority(globalRank);

  if (globalRank == 0) {
    std::cout
        << "\n================================================================================\n";
    std::cout << "IbverbxVirtualQp Cross-Host RDMA Write Latency Benchmark\n";
    std::cout
        << "================================================================================\n";
    std::cout << "Number of ranks: " << numRanks << std::endl;
    std::cout << "Iterations: " << kDefaultIterations << " (<1MB) / "
              << kLargeMsgIterations << " (1-64MB) / "
              << kExtraLargeMsgIterations << " (>=64MB)" << std::endl;
    std::cout << "Warmup: " << kWarmupIterations << std::endl;
    std::cout
        << "================================================================================\n\n";
  }

  auto messageSizes = generateMessageSizes();
  std::vector<BenchmarkResult> results;

  for (size_t msgSize : messageSizes) {
    std::string testName = formatSize(msgSize);
    int iterations = getIterationsForMsgSize(msgSize);

    if (globalRank == 0) {
      std::cout << "Testing message size: " << testName << " ..." << std::flush;
    }

    auto ctx = createBenchmarkContext(nicDevId, msgSize);
    auto remoteInfo = exchangeAndConnect(ctx.get(), peerRank);
    initSendWr(ctx.get(), msgSize, remoteInfo);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (globalRank == 0) {
      auto result = runSenderBenchmark(ctx.get(), iterations, testName);
      results.push_back(result);
    } else {
      runReceiverBenchmark(ctx.get(), iterations, testName);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (globalRank == 0) {
      std::cout << " done" << std::endl;
    }
  }

  // Print results table
  printResultsTable(results);
}

//------------------------------------------------------------------------------
// Helper: Compute statistics from latency samples
//------------------------------------------------------------------------------

static BenchmarkResult computeLatencyStats(
    std::vector<double>& deltasUs,
    const std::string& testName,
    size_t msgSize) {
  BenchmarkResult result;
  if (deltasUs.empty()) {
    return result;
  }

  std::sort(deltasUs.begin(), deltasUs.end());
  int iterations = static_cast<int>(deltasUs.size());

  double sum = 0;
  for (auto d : deltasUs) {
    sum += d;
  }

  result.testName = testName;
  result.messageSize = msgSize;
  result.medianUs = deltasUs[iterations / 2];
  result.avgUs = sum / static_cast<double>(iterations);
  result.minUs = deltasUs[0];
  result.maxUs = deltasUs[iterations - 1];
  result.p99Us = deltasUs[static_cast<int>(iterations * 0.99)];
  // Bandwidth in GBps: bytes / (us * 1e6) = bytes/s, then / 1e9 = GB/s
  result.bandwidthGBps = (result.medianUs > 0)
      ? static_cast<double>(msgSize) / (result.medianUs * 1000.0)
      : 0.0;

  return result;
}

//------------------------------------------------------------------------------
// RDMA_WRITE Latency Benchmark
//------------------------------------------------------------------------------

// Sender (rank 0) performs RDMA_WRITE, receiver is passive (one-sided)
static BenchmarkResult runRdmaWriteLatencyBenchmark(
    BenchmarkContext* ctx,
    const ExchangeInfo& remoteInfo,
    size_t msgSize,
    int iterations,
    int globalRank) {
  bool isSender = (globalRank == 0);

  // Prepare send WR (only sender needs it)
  ibv_sge sge{};
  ibv_send_wr sendWr{};
  ibv_send_wr badSendWr{};

  if (isSender) {
    sge.addr = reinterpret_cast<uint64_t>(ctx->sendBuf);
    sge.length = static_cast<uint32_t>(msgSize);
    sge.lkey = ctx->mr->mr()->lkey;

    sendWr.wr_id = 0;
    sendWr.next = nullptr;
    sendWr.sg_list = &sge;
    sendWr.num_sge = 1;
    sendWr.opcode = IBV_WR_RDMA_WRITE;
    sendWr.send_flags = IBV_SEND_SIGNALED;
    sendWr.wr.rdma.remote_addr = remoteInfo.addr;
    sendWr.wr.rdma.rkey = remoteInfo.rkey;
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  std::vector<double> deltasUs;
  using Clock = std::chrono::high_resolution_clock;

  if (isSender) {
    // Warmup
    for (int i = 0; i < kWarmupIterations; ++i) {
      ctx->endpoint->qp.postSend(&sendWr, &badSendWr);
      pollCqUntilCompletion(ctx->endpoint->cq);
    }

    // Measurement
    deltasUs.resize(iterations);
    for (int i = 0; i < iterations; ++i) {
      auto start = Clock::now();
      ctx->endpoint->qp.postSend(&sendWr, &badSendWr);
      pollCqUntilCompletion(ctx->endpoint->cq);
      auto end = Clock::now();
      deltasUs[i] =
          std::chrono::duration<double, std::micro>(end - start).count();
    }
  }
  // Receiver does nothing - RDMA_WRITE is one-sided

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  return computeLatencyStats(deltasUs, "RDMA_WRITE", msgSize);
}

//------------------------------------------------------------------------------
// RDMA_READ Latency Benchmark
//------------------------------------------------------------------------------

// Receiver (rank 1) performs RDMA_READ from sender's buffer, sender is passive
static BenchmarkResult runRdmaReadLatencyBenchmark(
    BenchmarkContext* ctx,
    const ExchangeInfo& remoteInfo,
    size_t msgSize,
    int iterations,
    int globalRank) {
  bool isSender = (globalRank == 0);

  // Prepare send WR for RDMA_READ (only receiver needs it)
  ibv_sge sge{};
  ibv_send_wr sendWr{};
  ibv_send_wr badSendWr{};

  if (!isSender) {
    // Receiver reads FROM sender's buffer INTO its own recvBuf
    sge.addr = reinterpret_cast<uint64_t>(ctx->recvBuf);
    sge.length = static_cast<uint32_t>(msgSize);
    sge.lkey = ctx->mr->mr()->lkey;

    sendWr.wr_id = 0;
    sendWr.next = nullptr;
    sendWr.sg_list = &sge;
    sendWr.num_sge = 1;
    sendWr.opcode = IBV_WR_RDMA_READ;
    sendWr.send_flags = IBV_SEND_SIGNALED;
    sendWr.wr.rdma.remote_addr = remoteInfo.addr;
    sendWr.wr.rdma.rkey = remoteInfo.rkey;
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  std::vector<double> deltasUs;
  using Clock = std::chrono::high_resolution_clock;

  if (!isSender) {
    // Warmup
    for (int i = 0; i < kWarmupIterations; ++i) {
      ctx->endpoint->qp.postSend(&sendWr, &badSendWr);
      pollCqUntilCompletion(ctx->endpoint->cq);
    }

    // Measurement
    deltasUs.resize(iterations);
    for (int i = 0; i < iterations; ++i) {
      auto start = Clock::now();
      ctx->endpoint->qp.postSend(&sendWr, &badSendWr);
      pollCqUntilCompletion(ctx->endpoint->cq);
      auto end = Clock::now();
      deltasUs[i] =
          std::chrono::duration<double, std::micro>(end - start).count();
    }
  }
  // Sender does nothing - RDMA_READ is one-sided

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  return computeLatencyStats(deltasUs, "RDMA_READ", msgSize);
}

//------------------------------------------------------------------------------
// RDMA_WRITE_WITH_IMM Latency Benchmark
//------------------------------------------------------------------------------

// Sender (rank 0) performs RDMA_WRITE_WITH_IMM, receiver posts recv and polls
static BenchmarkResult runRdmaWriteWithImmLatencyBenchmark(
    BenchmarkContext* ctx,
    const ExchangeInfo& remoteInfo,
    size_t msgSize,
    int iterations,
    int globalRank) {
  bool isSender = (globalRank == 0);

  // Prepare WRs
  ibv_sge sge{};
  ibv_send_wr sendWr{};
  ibv_send_wr badSendWr{};
  ibv_sge recvSge{};
  ibv_recv_wr recvWr{};
  ibv_recv_wr badRecvWr{};

  if (isSender) {
    sge.addr = reinterpret_cast<uint64_t>(ctx->sendBuf);
    sge.length = static_cast<uint32_t>(msgSize);
    sge.lkey = ctx->mr->mr()->lkey;

    sendWr.wr_id = 0;
    sendWr.next = nullptr;
    sendWr.sg_list = &sge;
    sendWr.num_sge = 1;
    sendWr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    sendWr.send_flags = IBV_SEND_SIGNALED;
    sendWr.wr.rdma.remote_addr = remoteInfo.addr;
    sendWr.wr.rdma.rkey = remoteInfo.rkey;
    sendWr.imm_data = static_cast<uint32_t>(msgSize);
  } else {
    recvWr.wr_id = 0;
    recvWr.next = nullptr;
    recvWr.sg_list = &recvSge;
    recvWr.num_sge = 0;
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  std::vector<double> deltasUs;
  using Clock = std::chrono::high_resolution_clock;

  if (isSender) {
    // Warmup
    for (int i = 0; i < kWarmupIterations; ++i) {
      ctx->endpoint->qp.postSend(&sendWr, &badSendWr);
      pollCqUntilCompletion(ctx->endpoint->cq);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Measurement
    deltasUs.resize(iterations);
    for (int i = 0; i < iterations; ++i) {
      auto start = Clock::now();
      ctx->endpoint->qp.postSend(&sendWr, &badSendWr);
      pollCqUntilCompletion(ctx->endpoint->cq);
      auto end = Clock::now();
      deltasUs[i] =
          std::chrono::duration<double, std::micro>(end - start).count();
    }
  } else {
    // Receiver: post recv and poll completion
    // Warmup
    for (int i = 0; i < kWarmupIterations; ++i) {
      ctx->endpoint->qp.postRecv(&recvWr, &badRecvWr);
      pollCqUntilCompletion(ctx->endpoint->cq);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Measurement
    deltasUs.resize(iterations);
    for (int i = 0; i < iterations; ++i) {
      auto start = Clock::now();
      ctx->endpoint->qp.postRecv(&recvWr, &badRecvWr);
      pollCqUntilCompletion(ctx->endpoint->cq);
      auto end = Clock::now();
      deltasUs[i] =
          std::chrono::duration<double, std::micro>(end - start).count();
    }
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  return computeLatencyStats(deltasUs, "RDMA_WRITE_WITH_IMM", msgSize);
}

// Helper function for bandwidth measurement using batched posting
// Signals on every WR but uses pipelining (tx_depth outstanding WRs)
// Only rank 0 sends, rank 1 is passive (half-duplex)
static BenchmarkResult runBandwidthBenchmark(
    BenchmarkContext* ctx,
    const ExchangeInfo& remoteInfo,
    size_t msgSize,
    int iterations,
    int globalRank) {
  bool isSender = (globalRank == 0);

  // Prepare SGE and WR for RDMA_WRITE
  ibv_sge sge{};
  ibv_send_wr sendWr{};
  ibv_send_wr badSendWr{};

  if (isSender) {
    sge.addr = reinterpret_cast<uint64_t>(ctx->sendBuf);
    sge.length = static_cast<uint32_t>(msgSize);
    sge.lkey = ctx->mr->mr()->lkey;

    sendWr.wr_id = 0;
    sendWr.next = nullptr;
    sendWr.sg_list = &sge;
    sendWr.num_sge = 1;
    sendWr.opcode = IBV_WR_RDMA_WRITE;
    sendWr.send_flags = IBV_SEND_SIGNALED;
    sendWr.wr.rdma.remote_addr = remoteInfo.addr;
    sendWr.wr.rdma.rkey = remoteInfo.rkey;
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  using Clock = std::chrono::high_resolution_clock;
  BenchmarkResult result;

  if (isSender) {
    uint64_t wrIdCounter = 0;

    // Warmup phase
    for (int i = 0; i < kWarmupIterations; ++i) {
      sendWr.wr_id = wrIdCounter++;
      ctx->endpoint->qp.postSend(&sendWr, &badSendWr);
      pollCqUntilCompletion(ctx->endpoint->cq);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Measurement phase - pipelined posting with tx_depth outstanding WRs
    uint64_t scnt = 0;
    uint64_t ccnt = 0;
    constexpr int kPollBatch = 16;

    auto start = Clock::now();

    while (scnt < static_cast<uint64_t>(iterations) ||
           ccnt < static_cast<uint64_t>(iterations)) {
      // Post while pipeline has room
      while (scnt < static_cast<uint64_t>(iterations) &&
             (scnt - ccnt + 1) <= static_cast<uint64_t>(kBwTxDepth)) {
        sendWr.wr_id = wrIdCounter++;
        ctx->endpoint->qp.postSend(&sendWr, &badSendWr);
        ++scnt;
      }

      // Poll completions
      if (ccnt < static_cast<uint64_t>(iterations)) {
        auto maybeWcs = ctx->endpoint->cq.pollCq(kPollBatch);
        if (maybeWcs && !maybeWcs->empty()) {
          for (const auto& wc : *maybeWcs) {
            if (wc.status != IBV_WC_SUCCESS) {
              throw std::runtime_error("CQ poll returned error status");
            }
            ++ccnt;
          }
        }
      }
    }

    // Ensure remote has received all data before stopping timer
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    auto end = Clock::now();
    double totalTimeUs =
        std::chrono::duration<double, std::micro>(end - start).count();

    // Calculate bandwidth
    uint64_t totalBytes = static_cast<uint64_t>(iterations) * msgSize;
    double bandwidthGBps = (totalTimeUs > 0)
        ? static_cast<double>(totalBytes) / (totalTimeUs * 1000.0)
        : 0.0;

    // Calculate message rate in Mpps
    double totalTimeSec = totalTimeUs / 1000000.0;
    double msgRateMpps = (totalTimeSec > 0)
        ? static_cast<double>(iterations) / totalTimeSec / 1000000.0
        : 0.0;

    result.testName = "RDMA_WRITE";
    result.messageSize = msgSize;
    result.avgUs = totalTimeUs / static_cast<double>(iterations);
    result.medianUs = result.avgUs;
    result.minUs = result.avgUs;
    result.maxUs = result.avgUs;
    result.p99Us = result.avgUs;
    result.bandwidthGBps = bandwidthGBps;
    result.msgRateMpps = msgRateMpps;
  } else {
    // Receiver waits at barriers to synchronize with sender
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  return result;
}

TEST_F(IbverbxVirtualQpBenchmarkFixture, CrossHostFullRttLatency_RdmaWrite) {
  if (numRanks != 2) {
    XLOGF(INFO, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;
  int nicDevId = globalRank;

  setRealtimePriority(globalRank);

  if (globalRank == 0) {
    std::cout
        << "\n================================================================================\n";
    std::cout
        << "IbverbxVirtualQp Cross-Host Full-RTT Latency Benchmark: RDMA_WRITE\n";
    std::cout
        << "================================================================================\n";
    std::cout << "Number of ranks: " << numRanks << std::endl;
    std::cout << "Iterations: " << kDefaultIterations << " (<1MB) / "
              << kLargeMsgIterations << " (1-64MB) / "
              << kExtraLargeMsgIterations << " (>=64MB)" << std::endl;
    std::cout << "Warmup: " << kWarmupIterations << std::endl;
    std::cout
        << "================================================================================\n\n";
  }

  auto messageSizes = generateMessageSizes();
  std::vector<BenchmarkResult> results;

  for (size_t msgSize : messageSizes) {
    std::string testName = formatSize(msgSize);
    int iterations = getIterationsForMsgSize(msgSize);

    if (globalRank == 0) {
      std::cout << "Testing " << testName << " ..." << std::flush;
    }

    auto ctx = createBenchmarkContext(nicDevId, msgSize);
    auto remoteInfo = exchangeAndConnect(ctx.get(), peerRank);

    auto result = runRdmaWriteLatencyBenchmark(
        ctx.get(), remoteInfo, msgSize, iterations, globalRank);

    if (globalRank == 0) {
      result.testName = testName;
      results.push_back(result);
      std::cout << " done" << std::endl;
    }
  }

  if (globalRank == 0) {
    printResultsTable(results, "RDMA_WRITE");
  }
}

TEST_F(IbverbxVirtualQpBenchmarkFixture, CrossHostFullRttLatency_RdmaRead) {
  if (numRanks != 2) {
    XLOGF(INFO, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;
  int nicDevId = globalRank;

  setRealtimePriority(globalRank);

  if (globalRank == 0) {
    std::cout
        << "\n================================================================================\n";
    std::cout
        << "IbverbxVirtualQp Cross-Host Full-RTT Latency Benchmark: RDMA_READ\n";
    std::cout
        << "================================================================================\n";
    std::cout << "Number of ranks: " << numRanks << std::endl;
    std::cout << "Iterations: " << kDefaultIterations << " (<1MB) / "
              << kLargeMsgIterations << " (1-64MB) / "
              << kExtraLargeMsgIterations << " (>=64MB)" << std::endl;
    std::cout << "Warmup: " << kWarmupIterations << std::endl;
    std::cout
        << "================================================================================\n\n";
  }

  auto messageSizes = generateMessageSizes();
  std::vector<BenchmarkResult> results;

  for (size_t msgSize : messageSizes) {
    std::string testName = formatSize(msgSize);
    int iterations = getIterationsForMsgSize(msgSize);

    if (globalRank == 0) {
      std::cout << "Testing " << testName << " ..." << std::flush;
    }

    auto ctx = createBenchmarkContext(nicDevId, msgSize);
    auto remoteInfo = exchangeAndConnect(ctx.get(), peerRank);

    auto result = runRdmaReadLatencyBenchmark(
        ctx.get(), remoteInfo, msgSize, iterations, globalRank);

    // For RDMA_READ, receiver (rank 1) has the results - send to rank 0
    if (globalRank == 1) {
      double resultData[6] = {
          result.medianUs,
          result.avgUs,
          result.minUs,
          result.maxUs,
          result.p99Us,
          result.bandwidthGBps};
      MPI_CHECK(MPI_Send(resultData, 6, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD));
    } else {
      double resultData[6];
      MPI_CHECK(MPI_Recv(
          resultData, 6, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      BenchmarkResult recvResult;
      recvResult.testName = testName;
      recvResult.messageSize = msgSize;
      recvResult.medianUs = resultData[0];
      recvResult.avgUs = resultData[1];
      recvResult.minUs = resultData[2];
      recvResult.maxUs = resultData[3];
      recvResult.p99Us = resultData[4];
      recvResult.bandwidthGBps = resultData[5];
      results.push_back(recvResult);
      std::cout << " done" << std::endl;
    }
  }

  if (globalRank == 0) {
    printResultsTable(results, "RDMA_READ");
  }
}

TEST_F(
    IbverbxVirtualQpBenchmarkFixture,
    CrossHostFullRttLatency_RdmaWriteWithImm) {
  if (numRanks != 2) {
    XLOGF(INFO, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  setRealtimePriority(globalRank);

  if (globalRank == 0) {
    std::cout
        << "\n================================================================================\n";
    std::cout
        << "IbverbxVirtualQp Cross-Host Full-RTT Latency Benchmark: RDMA_WRITE_WITH_IMM\n";
    std::cout
        << "================================================================================\n";
    std::cout << "Number of ranks: " << numRanks << std::endl;
    std::cout << "Iterations: " << kDefaultIterations << " (<1MB) / "
              << kLargeMsgIterations << " (1-64MB) / "
              << kExtraLargeMsgIterations << " (>=64MB)" << std::endl;
    std::cout << "Warmup: " << kWarmupIterations << std::endl;
    std::cout
        << "================================================================================\n\n";
  }

  auto messageSizes = generateMessageSizes();
  std::vector<BenchmarkResult> results;

  for (size_t msgSize : messageSizes) {
    std::string testName = formatSize(msgSize);
    int iterations = getIterationsForMsgSize(msgSize);

    if (globalRank == 0) {
      std::cout << "Testing " << testName << " ..." << std::flush;
    }

    auto ctx = createBenchmarkContext(globalRank, msgSize);
    auto remoteInfo = exchangeAndConnect(ctx.get(), peerRank);

    auto result = runRdmaWriteWithImmLatencyBenchmark(
        ctx.get(), remoteInfo, msgSize, iterations, globalRank);

    if (globalRank == 0) {
      result.testName = testName;
      results.push_back(result);
      std::cout << " done" << std::endl;
    }
  }

  if (globalRank == 0) {
    printResultsTable(results, "RDMA_WRITE_WITH_IMM");
  }
}

// TODO: Fix and re-enable bandwidth test
TEST_F(IbverbxVirtualQpBenchmarkFixture, CrossHostBandwidth_RdmaWrite) {
  if (numRanks != 2) {
    XLOGF(INFO, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;
  int nicDevId = globalRank;

  setRealtimePriority(globalRank);

  if (globalRank == 0) {
    std::cout
        << "\n================================================================================\n";
    std::cout
        << "IbverbxVirtualQp Cross-Host Bandwidth Benchmark: RDMA_WRITE\n";
    std::cout
        << "================================================================================\n";
    std::cout << "Number of ranks: " << numRanks << std::endl;
    std::cout << "Iterations per test: " << kBwIterations << std::endl;
    std::cout << "Warmup iterations: " << kWarmupIterations << std::endl;
    std::cout << "TX depth (pipelining): " << kBwTxDepth << std::endl;
    std::cout
        << "================================================================================\n\n";
  }

  // Test various message sizes from 1B to 4MB
  auto messageSizes = generateMessageSizes(1, 4 * 1024 * 1024);
  std::vector<BenchmarkResult> results;

  for (size_t msgSize : messageSizes) {
    std::string testName = formatSize(msgSize);

    if (globalRank == 0) {
      std::cout << "Testing " << testName << " ..." << std::flush;
    }

    auto ctx = createBenchmarkContext(nicDevId, msgSize);
    auto remoteInfo = exchangeAndConnect(ctx.get(), peerRank);

    auto result = runBandwidthBenchmark(
        ctx.get(), remoteInfo, msgSize, kBwIterations, globalRank);

    if (globalRank == 0) {
      result.testName = testName;
      results.push_back(result);
      std::cout << " done (" << formatBandwidth(result.bandwidthGBps) << ")"
                << std::endl;
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  if (globalRank == 0) {
    printBandwidthResultsTable(results, "RDMA_WRITE");
  }
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
