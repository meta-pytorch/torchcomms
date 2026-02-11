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
#include <string>
#include <vector>

#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>

#include "comms/ctran/ibverbx/IbverbxSymbols.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ibverbx {
extern IbvSymbols ibvSymbols;
} // namespace ibverbx

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
constexpr uint32_t kMaxOutstandingWrs = 128;
constexpr uint32_t kMaxSge = 1;
constexpr int kCqSize = 256;
constexpr int kDefaultIterations = 10000;
constexpr int kWarmupIterations = 100;
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

struct QpExchangeInfo {
  uint32_t qpn;
  uint16_t lid;
  uint8_t gid[16];
  uint32_t rkey;
  uint64_t addr; // Remote buffer address (recv area)
};

//------------------------------------------------------------------------------
// Raw IBVerbs Endpoint
//------------------------------------------------------------------------------

struct IbvEndPoint {
  ibverbx::ibv_context* ctx{nullptr};
  ibverbx::ibv_pd* pd{nullptr};
  ibverbx::ibv_cq* cq{nullptr};
  ibverbx::ibv_qp* qp{nullptr};
  ibverbx::ibv_mr* mr{nullptr};
  uint8_t port{kPortNum};
  int gidIndex{kGidIndex};

  ~IbvEndPoint() {
    if (qp)
      ibverbx::ibvSymbols.ibv_internal_destroy_qp(qp);
    if (mr)
      ibverbx::ibvSymbols.ibv_internal_dereg_mr(mr);
    if (cq)
      ibverbx::ibvSymbols.ibv_internal_destroy_cq(cq);
    if (pd)
      ibverbx::ibvSymbols.ibv_internal_dealloc_pd(pd);
    if (ctx)
      ibverbx::ibvSymbols.ibv_internal_close_device(ctx);
  }
};

//------------------------------------------------------------------------------
// Benchmark Context
//------------------------------------------------------------------------------

struct BenchmarkContext {
  std::unique_ptr<IbvEndPoint> endpoint;
  void* buffer{nullptr};
  size_t bufferSize{0};
  size_t msgSize{0};

  // Buffer layout: [send_area: msgSize][recv_area: msgSize]
  char* sendBuf{nullptr};
  volatile char* recvBuf{nullptr};

  // Pre-constructed work requests
  ibverbx::ibv_sge sge{};
  ibverbx::ibv_send_wr sendWr{};
  ibverbx::ibv_send_wr* badSendWr{nullptr};
  ibverbx::ibv_wc wc{};

  ~BenchmarkContext() {
    if (buffer) {
      free(buffer);
    }
  }
};

// Filter devices based on NCCL_IB_HCA and NCCL_IB_HCA_PREFIX cvars
// This matches the logic in IbvDevice::ibvFilterDeviceList
static std::vector<ibverbx::ibv_device*> filterDeviceList(
    int numDevices,
    ibverbx::ibv_device** deviceList,
    const std::vector<std::string>& hcaList,
    const std::string& hcaPrefix) {
  std::vector<ibverbx::ibv_device*> filteredDevices;

  if (hcaList.empty()) {
    // No filter specified, return all devices
    for (int i = 0; i < numDevices; i++) {
      filteredDevices.push_back(deviceList[i]);
    }
    return filteredDevices;
  }

  // Filter devices based on hcaPrefix mode
  if (hcaPrefix == "=") {
    // Exact match mode
    for (const auto& hca : hcaList) {
      // Extract device name (strip port if present)
      std::string deviceName = hca;
      auto colonPos = hca.find(':');
      if (colonPos != std::string::npos) {
        deviceName = hca.substr(0, colonPos);
      }
      for (int i = 0; i < numDevices; i++) {
        if (deviceName == deviceList[i]->name) {
          filteredDevices.push_back(deviceList[i]);
          break;
        }
      }
    }
  } else if (hcaPrefix == "^") {
    // Exclude match mode
    for (int i = 0; i < numDevices; i++) {
      bool excluded = false;
      for (const auto& hca : hcaList) {
        std::string deviceName = hca;
        auto colonPos = hca.find(':');
        if (colonPos != std::string::npos) {
          deviceName = hca.substr(0, colonPos);
        }
        if (deviceName == deviceList[i]->name) {
          excluded = true;
          break;
        }
      }
      if (!excluded) {
        filteredDevices.push_back(deviceList[i]);
      }
    }
  } else {
    // Prefix match mode (default)
    for (const auto& hca : hcaList) {
      std::string deviceName = hca;
      auto colonPos = hca.find(':');
      if (colonPos != std::string::npos) {
        deviceName = hca.substr(0, colonPos);
      }
      for (int i = 0; i < numDevices; i++) {
        if (strncmp(
                deviceList[i]->name, deviceName.c_str(), deviceName.length()) ==
            0) {
          filteredDevices.push_back(deviceList[i]);
          break;
        }
      }
    }
  }

  return filteredDevices;
}

static ibverbx::ibv_context* openIbDevice(int nicDevId) {
  int numDevices;
  ibverbx::ibv_device** deviceList =
      ibverbx::ibvSymbols.ibv_internal_get_device_list(&numDevices);
  if (!deviceList || numDevices == 0) {
    throw std::runtime_error("No IB devices found");
  }

  // Filter devices using NCCL_IB_HCA and NCCL_IB_HCA_PREFIX cvars
  auto filteredDevices =
      filterDeviceList(numDevices, deviceList, NCCL_IB_HCA, NCCL_IB_HCA_PREFIX);

  if (filteredDevices.empty()) {
    ibverbx::ibvSymbols.ibv_internal_free_device_list(deviceList);
    throw std::runtime_error(
        "No IB devices found after filtering with NCCL_IB_HCA");
  }

  if (nicDevId >= static_cast<int>(filteredDevices.size())) {
    std::cerr << "nicDevId " << nicDevId
              << " out of range (filtered devices: " << filteredDevices.size()
              << ")" << std::endl;
    ibverbx::ibvSymbols.ibv_internal_free_device_list(deviceList);
    throw std::out_of_range("nicDevId out of range for filtered device list");
  }

  ibverbx::ibv_device* device = filteredDevices[nicDevId];

  ibverbx::ibv_context* ctx =
      ibverbx::ibvSymbols.ibv_internal_open_device(device);
  ibverbx::ibvSymbols.ibv_internal_free_device_list(deviceList);

  if (!ctx) {
    throw std::runtime_error("Failed to open device");
  }

  return ctx;
}

static std::unique_ptr<BenchmarkContext> createBenchmarkContext(
    int nicDevId,
    size_t msgSize) {
  auto ctx = std::make_unique<BenchmarkContext>();

  size_t totalSize = 2 * msgSize;
  ctx->bufferSize = totalSize;
  ctx->msgSize = msgSize;

  ctx->endpoint = std::make_unique<IbvEndPoint>();
  ctx->endpoint->ctx = openIbDevice(nicDevId);

  ctx->endpoint->pd =
      ibverbx::ibvSymbols.ibv_internal_alloc_pd(ctx->endpoint->ctx);
  if (!ctx->endpoint->pd) {
    throw std::runtime_error("Failed to allocate PD");
  }

  ctx->endpoint->cq = ibverbx::ibvSymbols.ibv_internal_create_cq(
      ctx->endpoint->ctx, kCqSize, nullptr, nullptr, 0);
  if (!ctx->endpoint->cq) {
    throw std::runtime_error("Failed to create CQ");
  }

  if (posix_memalign(&ctx->buffer, 4096, totalSize) != 0) {
    throw std::runtime_error("Failed to allocate host memory");
  }
  memset(ctx->buffer, 0, totalSize);

  ctx->sendBuf = static_cast<char*>(ctx->buffer);
  ctx->recvBuf = static_cast<char*>(ctx->buffer) + msgSize;

  int accessFlags = ibverbx::IBV_ACCESS_LOCAL_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_WRITE | ibverbx::IBV_ACCESS_REMOTE_READ;
  ctx->endpoint->mr = ibverbx::ibvSymbols.ibv_internal_reg_mr(
      ctx->endpoint->pd, ctx->buffer, totalSize, accessFlags);
  if (!ctx->endpoint->mr) {
    throw std::runtime_error("Failed to register MR");
  }

  // Create QP
  ibverbx::ibv_qp_init_attr qpAttr{};
  qpAttr.send_cq = ctx->endpoint->cq;
  qpAttr.recv_cq = ctx->endpoint->cq;
  qpAttr.cap.max_send_wr = kMaxOutstandingWrs;
  qpAttr.cap.max_recv_wr = kMaxOutstandingWrs;
  qpAttr.cap.max_send_sge = kMaxSge;
  qpAttr.cap.max_recv_sge = kMaxSge;
  qpAttr.cap.max_inline_data = kMaxInlineData;
  qpAttr.qp_type = ibverbx::IBV_QPT_RC;
  qpAttr.sq_sig_all = 0;

  ctx->endpoint->qp =
      ibverbx::ibvSymbols.ibv_internal_create_qp(ctx->endpoint->pd, &qpAttr);
  if (!ctx->endpoint->qp) {
    throw std::runtime_error("Failed to create QP");
  }

  return ctx;
}

static QpExchangeInfo getLocalQpInfo(BenchmarkContext* ctx) {
  QpExchangeInfo info{};
  info.qpn = ctx->endpoint->qp->qp_num;

  ibverbx::ibv_port_attr portAttr{};
  ibverbx::ibvSymbols.ibv_internal_query_port(
      ctx->endpoint->ctx, ctx->endpoint->port, &portAttr);
  info.lid = portAttr.lid;

  ibverbx::ibv_gid gid{};
  ibverbx::ibvSymbols.ibv_internal_query_gid(
      ctx->endpoint->ctx, ctx->endpoint->port, ctx->endpoint->gidIndex, &gid);
  memcpy(info.gid, &gid, 16);

  info.rkey = ctx->endpoint->mr->rkey;
  info.addr = reinterpret_cast<uint64_t>(ctx->recvBuf);

  return info;
}

static void modifyQpToInit(BenchmarkContext* ctx) {
  ibverbx::ibv_qp_attr attr{};
  attr.qp_state = ibverbx::IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = ctx->endpoint->port;
  attr.qp_access_flags = ibverbx::IBV_ACCESS_LOCAL_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_WRITE | ibverbx::IBV_ACCESS_REMOTE_READ;

  int flags = ibverbx::IBV_QP_STATE | ibverbx::IBV_QP_PKEY_INDEX |
      ibverbx::IBV_QP_PORT | ibverbx::IBV_QP_ACCESS_FLAGS;

  if (ibverbx::ibvSymbols.ibv_internal_modify_qp(
          ctx->endpoint->qp, &attr, flags)) {
    throw std::runtime_error("Failed to modify QP to INIT");
  }
}

static void modifyQpToRtr(
    BenchmarkContext* ctx,
    const QpExchangeInfo& remoteInfo) {
  ibverbx::ibv_qp_attr attr{};
  attr.qp_state = ibverbx::IBV_QPS_RTR;
  attr.path_mtu = ibverbx::IBV_MTU_4096;
  attr.dest_qp_num = remoteInfo.qpn;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;

  attr.ah_attr.dlid = remoteInfo.lid;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = ctx->endpoint->port;
  attr.ah_attr.is_global = 1;
  memcpy(&attr.ah_attr.grh.dgid, remoteInfo.gid, 16);
  attr.ah_attr.grh.sgid_index = ctx->endpoint->gidIndex;
  attr.ah_attr.grh.hop_limit = 255;
  attr.ah_attr.grh.traffic_class = 0;

  int flags = ibverbx::IBV_QP_STATE | ibverbx::IBV_QP_AV |
      ibverbx::IBV_QP_PATH_MTU | ibverbx::IBV_QP_DEST_QPN |
      ibverbx::IBV_QP_RQ_PSN | ibverbx::IBV_QP_MAX_DEST_RD_ATOMIC |
      ibverbx::IBV_QP_MIN_RNR_TIMER;

  if (ibverbx::ibvSymbols.ibv_internal_modify_qp(
          ctx->endpoint->qp, &attr, flags)) {
    throw std::runtime_error("Failed to modify QP to RTR");
  }
}

static void modifyQpToRts(BenchmarkContext* ctx) {
  ibverbx::ibv_qp_attr attr{};
  attr.qp_state = ibverbx::IBV_QPS_RTS;
  attr.sq_psn = 0;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.max_rd_atomic = 1;

  int flags = ibverbx::IBV_QP_STATE | ibverbx::IBV_QP_SQ_PSN |
      ibverbx::IBV_QP_TIMEOUT | ibverbx::IBV_QP_RETRY_CNT |
      ibverbx::IBV_QP_RNR_RETRY | ibverbx::IBV_QP_MAX_QP_RD_ATOMIC;

  if (ibverbx::ibvSymbols.ibv_internal_modify_qp(
          ctx->endpoint->qp, &attr, flags)) {
    throw std::runtime_error("Failed to modify QP to RTS");
  }
}

static void connectQp(BenchmarkContext* ctx, const QpExchangeInfo& remoteInfo) {
  modifyQpToInit(ctx);
  modifyQpToRtr(ctx, remoteInfo);
  modifyQpToRts(ctx);
}

static void initSendWr(
    BenchmarkContext* ctx,
    size_t msgSize,
    const QpExchangeInfo& remoteInfo) {
  ctx->sge.addr = reinterpret_cast<uint64_t>(ctx->sendBuf);
  ctx->sge.length = static_cast<uint32_t>(msgSize);
  ctx->sge.lkey = ctx->endpoint->mr->lkey;

  ctx->sendWr.wr_id = 0;
  ctx->sendWr.next = nullptr;
  ctx->sendWr.sg_list = &ctx->sge;
  ctx->sendWr.num_sge = 1;
  ctx->sendWr.opcode = ibverbx::IBV_WR_RDMA_WRITE;
  ctx->sendWr.send_flags = ibverbx::IBV_SEND_SIGNALED;
  if (msgSize <= kMaxInlineData) {
    ctx->sendWr.send_flags |= ibverbx::IBV_SEND_INLINE;
  }
  ctx->sendWr.wr.rdma.remote_addr = remoteInfo.addr;
  ctx->sendWr.wr.rdma.rkey = remoteInfo.rkey;
}

// Exchange connection info with peer and establish QP connection
static QpExchangeInfo exchangeAndConnect(BenchmarkContext* ctx, int peerRank) {
  auto localInfo = getLocalQpInfo(ctx);

  QpExchangeInfo remoteInfo{};
  MPI_CHECK(MPI_Sendrecv(
      &localInfo,
      sizeof(QpExchangeInfo),
      MPI_BYTE,
      peerRank,
      0,
      &remoteInfo,
      sizeof(QpExchangeInfo),
      MPI_BYTE,
      peerRank,
      0,
      MPI_COMM_WORLD,
      MPI_STATUS_IGNORE));

  connectQp(ctx, remoteInfo);
  return remoteInfo;
}

//------------------------------------------------------------------------------
// Poll completion queue helper
//------------------------------------------------------------------------------

static void pollCqUntilCompletion(BenchmarkContext* ctx) {
  auto pollCqFn = ctx->endpoint->cq->context->ops.poll_cq;
  while (true) {
    int ne = pollCqFn(ctx->endpoint->cq, 1, &ctx->wc);
    if (ne > 0) {
      if (ctx->wc.status != ibverbx::IBV_WC_SUCCESS) {
        XLOGF(ERR, "WC failed with status {}", ctx->wc.status);
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

class IbverbsBenchmarkFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    ncclCvarInit();

    // Initialize ibverbs symbols
    if (ibverbx::buildIbvSymbols(ibverbx::ibvSymbols) != 0) {
      throw std::runtime_error("Failed to initialize ibverbs symbols");
    }
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

    auto postSendFn = ctx->endpoint->qp->context->ops.post_send;
    auto pollCqFn = ctx->endpoint->cq->context->ops.poll_cq;

    uint64_t scnt = 0;
    uint64_t rcnt = 0;

    // Warmup
    for (int i = 0; i < kWarmupIterations; ++i) {
      *postBuf = static_cast<char>(++scnt);
      postSendFn(ctx->endpoint->qp, &ctx->sendWr, &ctx->badSendWr);
      while (pollCqFn(ctx->endpoint->cq, 1, &ctx->wc) != 1) {
      }

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
      postSendFn(ctx->endpoint->qp, &ctx->sendWr, &ctx->badSendWr);
      while (pollCqFn(ctx->endpoint->cq, 1, &ctx->wc) != 1) {
      }

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

    auto postSendFn = ctx->endpoint->qp->context->ops.post_send;
    auto pollCqFn = ctx->endpoint->cq->context->ops.poll_cq;

    uint64_t scnt = 0;
    uint64_t rcnt = 0;

    int totalIterations = kWarmupIterations + iterations;

    for (int i = 0; i < totalIterations; ++i) {
      ++rcnt;
      while (*pollBuf != static_cast<char>(rcnt)) {
      }

      *postBuf = static_cast<char>(++scnt);
      postSendFn(ctx->endpoint->qp, &ctx->sendWr, &ctx->badSendWr);
      while (pollCqFn(ctx->endpoint->cq, 1, &ctx->wc) != 1) {
      }
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
    ss << "    Raw Ibverbs Cross-Host Latency Benchmark Results (" << opType
       << ")\n";
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
    ss << "    Raw Ibverbs Cross-Host Bandwidth Benchmark (" << opType << ")\n";
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

TEST_F(IbverbsBenchmarkFixture, CrossHostPingPongHalfRttLatency_RdmaWrite) {
  // Only test with 2 ranks (one per host)
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
    std::cout << "Raw Ibverbs Cross-Host RDMA Write Latency Benchmark\n";
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
    const QpExchangeInfo& remoteInfo,
    size_t msgSize,
    int iterations,
    int globalRank) {
  bool isSender = (globalRank == 0);

  // Prepare send WR (only sender needs it)
  ibverbx::ibv_sge sge{};
  ibverbx::ibv_send_wr sendWr{};
  ibverbx::ibv_send_wr* badSendWr{nullptr};

  if (isSender) {
    sge.addr = reinterpret_cast<uint64_t>(ctx->sendBuf);
    sge.length = static_cast<uint32_t>(msgSize);
    sge.lkey = ctx->endpoint->mr->lkey;

    sendWr.wr_id = 0;
    sendWr.next = nullptr;
    sendWr.sg_list = &sge;
    sendWr.num_sge = 1;
    sendWr.opcode = ibverbx::IBV_WR_RDMA_WRITE;
    sendWr.send_flags = ibverbx::IBV_SEND_SIGNALED;
    sendWr.wr.rdma.remote_addr = remoteInfo.addr;
    sendWr.wr.rdma.rkey = remoteInfo.rkey;
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  std::vector<double> deltasUs;
  using Clock = std::chrono::high_resolution_clock;

  if (isSender) {
    auto pollCqFn = ctx->endpoint->cq->context->ops.poll_cq;
    auto postSendFn = ctx->endpoint->qp->context->ops.post_send;

    // Warmup
    for (int i = 0; i < kWarmupIterations; ++i) {
      postSendFn(ctx->endpoint->qp, &sendWr, &badSendWr);
      while (pollCqFn(ctx->endpoint->cq, 1, &ctx->wc) != 1) {
      }
    }

    // Measurement
    deltasUs.resize(iterations);
    for (int i = 0; i < iterations; ++i) {
      auto start = Clock::now();
      postSendFn(ctx->endpoint->qp, &sendWr, &badSendWr);
      while (pollCqFn(ctx->endpoint->cq, 1, &ctx->wc) != 1) {
      }
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
    const QpExchangeInfo& remoteInfo,
    size_t msgSize,
    int iterations,
    int globalRank) {
  bool isSender = (globalRank == 0);

  // Prepare send WR for RDMA_READ (only receiver needs it)
  ibverbx::ibv_sge sge{};
  ibverbx::ibv_send_wr sendWr{};
  ibverbx::ibv_send_wr* badSendWr{nullptr};

  if (!isSender) {
    // Receiver reads FROM sender's buffer INTO its own recvBuf
    sge.addr = reinterpret_cast<uint64_t>(ctx->recvBuf);
    sge.length = static_cast<uint32_t>(msgSize);
    sge.lkey = ctx->endpoint->mr->lkey;

    sendWr.wr_id = 0;
    sendWr.next = nullptr;
    sendWr.sg_list = &sge;
    sendWr.num_sge = 1;
    sendWr.opcode = ibverbx::IBV_WR_RDMA_READ;
    sendWr.send_flags = ibverbx::IBV_SEND_SIGNALED;
    sendWr.wr.rdma.remote_addr = remoteInfo.addr;
    sendWr.wr.rdma.rkey = remoteInfo.rkey;
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  std::vector<double> deltasUs;
  using Clock = std::chrono::high_resolution_clock;

  if (!isSender) {
    auto pollCqFn = ctx->endpoint->cq->context->ops.poll_cq;
    auto postSendFn = ctx->endpoint->qp->context->ops.post_send;

    // Warmup
    for (int i = 0; i < kWarmupIterations; ++i) {
      postSendFn(ctx->endpoint->qp, &sendWr, &badSendWr);
      while (pollCqFn(ctx->endpoint->cq, 1, &ctx->wc) != 1) {
      }
    }

    // Measurement
    deltasUs.resize(iterations);
    for (int i = 0; i < iterations; ++i) {
      auto start = Clock::now();
      postSendFn(ctx->endpoint->qp, &sendWr, &badSendWr);
      while (pollCqFn(ctx->endpoint->cq, 1, &ctx->wc) != 1) {
      }
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
    const QpExchangeInfo& remoteInfo,
    size_t msgSize,
    int iterations,
    int globalRank) {
  bool isSender = (globalRank == 0);

  // Prepare WRs
  ibverbx::ibv_sge sge{};
  ibverbx::ibv_send_wr sendWr{};
  ibverbx::ibv_send_wr* badSendWr{nullptr};
  ibverbx::ibv_sge recvSge{};
  ibverbx::ibv_recv_wr recvWr{};
  ibverbx::ibv_recv_wr* badRecvWr{nullptr};

  if (isSender) {
    sge.addr = reinterpret_cast<uint64_t>(ctx->sendBuf);
    sge.length = static_cast<uint32_t>(msgSize);
    sge.lkey = ctx->endpoint->mr->lkey;

    sendWr.wr_id = 0;
    sendWr.next = nullptr;
    sendWr.sg_list = &sge;
    sendWr.num_sge = 1;
    sendWr.opcode = ibverbx::IBV_WR_RDMA_WRITE_WITH_IMM;
    sendWr.send_flags = ibverbx::IBV_SEND_SIGNALED;
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
    auto pollCqFn = ctx->endpoint->cq->context->ops.poll_cq;
    auto postSendFn = ctx->endpoint->qp->context->ops.post_send;

    // Warmup
    for (int i = 0; i < kWarmupIterations; ++i) {
      postSendFn(ctx->endpoint->qp, &sendWr, &badSendWr);
      while (pollCqFn(ctx->endpoint->cq, 1, &ctx->wc) != 1) {
      }
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Measurement
    deltasUs.resize(iterations);
    for (int i = 0; i < iterations; ++i) {
      auto start = Clock::now();
      postSendFn(ctx->endpoint->qp, &sendWr, &badSendWr);
      while (pollCqFn(ctx->endpoint->cq, 1, &ctx->wc) != 1) {
      }
      auto end = Clock::now();
      deltasUs[i] =
          std::chrono::duration<double, std::micro>(end - start).count();
    }
  } else {
    auto pollCqFn = ctx->endpoint->cq->context->ops.poll_cq;
    auto postRecvFn = ctx->endpoint->qp->context->ops.post_recv;

    // Receiver: post recv and poll completion
    // Warmup
    for (int i = 0; i < kWarmupIterations; ++i) {
      postRecvFn(ctx->endpoint->qp, &recvWr, &badRecvWr);
      while (pollCqFn(ctx->endpoint->cq, 1, &ctx->wc) != 1) {
      }
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Measurement
    deltasUs.resize(iterations);
    for (int i = 0; i < iterations; ++i) {
      auto start = Clock::now();
      postRecvFn(ctx->endpoint->qp, &recvWr, &badRecvWr);
      while (pollCqFn(ctx->endpoint->cq, 1, &ctx->wc) != 1) {
      }
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
    const QpExchangeInfo& remoteInfo,
    size_t msgSize,
    int iterations,
    int globalRank) {
  bool isSender = (globalRank == 0);

  // Prepare SGE and WR for RDMA_WRITE
  ibverbx::ibv_sge sge{};
  ibverbx::ibv_send_wr sendWr{};
  ibverbx::ibv_send_wr* badSendWr{nullptr};

  if (isSender) {
    sge.addr = reinterpret_cast<uint64_t>(ctx->sendBuf);
    sge.length = static_cast<uint32_t>(msgSize);
    sge.lkey = ctx->endpoint->mr->lkey;

    sendWr.wr_id = 0;
    sendWr.next = nullptr;
    sendWr.sg_list = &sge;
    sendWr.num_sge = 1;
    sendWr.opcode = ibverbx::IBV_WR_RDMA_WRITE;
    sendWr.send_flags = ibverbx::IBV_SEND_SIGNALED;
    sendWr.wr.rdma.remote_addr = remoteInfo.addr;
    sendWr.wr.rdma.rkey = remoteInfo.rkey;
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  using Clock = std::chrono::high_resolution_clock;
  BenchmarkResult result;

  if (isSender) {
    auto pollCqFn = ctx->endpoint->cq->context->ops.poll_cq;
    auto postSendFn = ctx->endpoint->qp->context->ops.post_send;
    uint64_t wrIdCounter = 0;

    // Warmup phase
    for (int i = 0; i < kWarmupIterations; ++i) {
      sendWr.wr_id = wrIdCounter++;
      postSendFn(ctx->endpoint->qp, &sendWr, &badSendWr);
      while (pollCqFn(ctx->endpoint->cq, 1, &ctx->wc) != 1) {
      }
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Measurement phase - pipelined posting with tx_depth outstanding WRs
    uint64_t scnt = 0;
    uint64_t ccnt = 0;
    ibverbx::ibv_wc wc;

    auto start = Clock::now();

    while (scnt < static_cast<uint64_t>(iterations) ||
           ccnt < static_cast<uint64_t>(iterations)) {
      // Post while pipeline has room
      while (scnt < static_cast<uint64_t>(iterations) &&
             (scnt - ccnt + 1) <= static_cast<uint64_t>(kBwTxDepth)) {
        sendWr.wr_id = wrIdCounter++;
        postSendFn(ctx->endpoint->qp, &sendWr, &badSendWr);
        ++scnt;
      }

      // Poll completions
      if (ccnt < static_cast<uint64_t>(iterations)) {
        int ne = pollCqFn(ctx->endpoint->cq, 1, &wc);
        if (ne > 0) {
          if (wc.status != ibverbx::IBV_WC_SUCCESS) {
            throw std::runtime_error("CQ poll returned error status");
          }
          ++ccnt;
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

TEST_F(IbverbsBenchmarkFixture, CrossHostFullRttLatency_RdmaWrite) {
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
        << "Raw Ibverbs Cross-Host Full-RTT Latency Benchmark: RDMA_WRITE\n";
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

TEST_F(IbverbsBenchmarkFixture, CrossHostFullRttLatency_RdmaRead) {
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
        << "Raw Ibverbs Cross-Host Full-RTT Latency Benchmark: RDMA_READ\n";
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

TEST_F(IbverbsBenchmarkFixture, CrossHostFullRttLatency_RdmaWriteWithImm) {
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
        << "Raw Ibverbs Cross-Host Full-RTT Latency Benchmark: RDMA_WRITE_WITH_IMM\n";
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

    if (globalRank == 0 || globalRank == 1) {
      result.testName = testName;
      results.push_back(result);
    }

    if (globalRank == 0) {
      std::cout << " done" << std::endl;
    }
  }

  if (globalRank == 0) {
    printResultsTable(results, "RDMA_WRITE_WITH_IMM");
  }
}

TEST_F(IbverbsBenchmarkFixture, CrossHostBandwidth_RdmaWrite) {
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
    std::cout << "Raw Ibverbs Cross-Host Bandwidth Benchmark: RDMA_WRITE\n";
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
