// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <folly/Synchronized.h>
#include <folly/futures/Future.h>
#include <folly/io/async/EventBase.h>
#include <folly/io/async/EventHandler.h>
#include <folly/io/async/ScopedEventBaseThread.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ibverbx;

// use broadcom nic for AMD platform, use mellanox nic for NV platform
#if defined(__HIP_PLATFORM_AMD__) && !defined(USE_FE_NIC)
const std::string kNicPrefix("bnxt_re");
#else
const std::string kNicPrefix("mlx5_");
#endif

constexpr uint8_t kPortNum = 1;
constexpr int kGidIndex = 3;
constexpr uint32_t kMaxOutstandingWrs = 128;
constexpr uint32_t kMaxSge = 1;

namespace {

class CqHandler : public folly::EventHandler {
 public:
  CqHandler(
      folly::EventBase* evb,
      ibv_comp_channel* ch,
      IbvCq* cq,
      std::function<void(const ibv_wc&)> onWc)
      : folly::EventHandler(evb), ch_(ch), cq_(cq), onWc_(std::move(onWc)) {
    changeHandlerFD(folly::NetworkSocket::fromFd(ch_->fd));
    registerHandler(READ | PERSIST); // keep firing while readable
  }
  void handlerReady(uint16_t) noexcept override {
    ibv_cq* cq;
    void* ctx;
    // drain all channel events
    while (true) {
      // reference:
      // https://man7.org/linux/man-pages/man3/ibv_ack_cq_events.3.html?utm_source=chatgpt.com
      auto result = ibvGetCqEvent(ch_, &cq, &ctx);
      if (result.hasError()) {
        // No more events available: EAGAIN
        break;
      }
      ibvAckCqEvents(cq, 1);
      cq_->reqNotifyCq(0); // re-arm
      // drain CQ in batches
      for (;;) {
        auto maybeWcs = cq_->pollCq(1);
        if (maybeWcs.hasError()) {
          std::cout << "pollCq error: " << maybeWcs.error().errStr << std::endl;
          break;
        }
        std::vector<ibv_wc> wcs = *maybeWcs;
        if (wcs.empty()) {
          break;
        }
        onWc_(wcs.at(0));
      }
    }
  }

 private:
  ibv_comp_channel* ch_;
  IbvCq* cq_;
  std::function<void(const ibv_wc&)> onWc_;
};

class IbvEndPoint {
 public:
  explicit IbvEndPoint(int nicDevId);
  ~IbvEndPoint();
  ibv_qp_init_attr makeIbvQpInitAttr();
  ibv_qp_attr makeQpAttrInit();
  ibv_qp_attr makeQpAttrRtr(ibv_gid remoteGid, const uint32_t remoteQpNum);
  void changeQpStateToRts(ibv_gid remoteGid, const uint32_t remoteQpNum);

  IbvDevice device;
  IbvPd pd;
  IbvCq cq;
  IbvQp qp;
  folly::ScopedEventBaseThread proxyThread{"IbvEndPointProxyThread"};
  folly::EventBase* evb;
  std::unique_ptr<CqHandler> cqHandler;
  ibv_comp_channel* ch;
  ibv_recv_wr recvNotifyWr_, badRecvNotifyWr_;
};

IbvEndPoint::IbvEndPoint(int nicDevId)
    : device(([nicDevId]() {
        // Initialize ibverbx first
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
        // Create a completion channel using the device API
        auto maybeCompChannel = device.createCompChannel();
        if (!maybeCompChannel) {
          throw std::runtime_error("Failed to create completion channel");
        }
        ch = std::move(*maybeCompChannel);
        // change to nonblocking mode to avoid thread parking if no event
        // arrived
        auto flags = fcntl(ch->fd, F_GETFL);
        if (fcntl(ch->fd, F_SETFL, flags | O_NONBLOCK) < 0) {
          throw std::runtime_error("Failed to create completion queue");
        }

        auto maybeCq = device.createCq(32768, nullptr, ch, 0);
        if (!maybeCq) {
          throw std::runtime_error("Failed to create completion queue");
        }
        auto result = (*maybeCq).reqNotifyCq(0);
        if (!result) {
          throw std::runtime_error("Failed to modify QP to INIT state");
        }
        return std::move(*maybeCq);
      })()),
      qp([this]() {
        auto initAttr = makeIbvQpInitAttr();

        auto maybeQp = pd.createQp(&initAttr);
        if (!maybeQp) {
          throw std::runtime_error("Failed to create queue pair");
        }
        return std::move(*maybeQp);
      }()) {
  memset(&recvNotifyWr_, 0, sizeof(recvNotifyWr_));
  recvNotifyWr_.wr_id = 0;
  recvNotifyWr_.next = nullptr;
  recvNotifyWr_.num_sge = 0;
  evb = proxyThread.getEventBase();
}

IbvEndPoint::~IbvEndPoint() = default;

// helper functions
ibv_qp_init_attr IbvEndPoint::makeIbvQpInitAttr() {
  ibv_qp_init_attr initAttr{};
  memset(&initAttr, 0, sizeof(ibv_qp_init_attr));
  initAttr.send_cq = cq.cq();
  initAttr.recv_cq = cq.cq();
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

ibv_qp_attr IbvEndPoint::makeQpAttrRtr(
    ibv_gid remoteGid,
    const uint32_t remoteQpNum) {
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
  qpAttr.dest_qp_num = remoteQpNum;
  return qpAttr;
}

ibv_qp_attr makeQpAttrRts() {
  const uint8_t kTimeout = 10;
  const uint8_t kRetryCnt = 7;

  struct ibv_qp_attr qpAttr {};
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));

  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = kTimeout;
  qpAttr.retry_cnt = kRetryCnt;

  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  return qpAttr;
}

void IbvEndPoint::changeQpStateToRts(
    ibv_gid remoteGid,
    const uint32_t remoteQpNum) {
  {
    // change QP group state to INIT
    auto qpAttr = makeQpAttrInit();
    auto result = qp.modifyQp(
        &qpAttr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (!result) {
      throw std::runtime_error("Failed to modify QP to INIT state");
    }
  }
  {
    // change QP group state to RTR
    auto qpAttr = makeQpAttrRtr(remoteGid, remoteQpNum);
    auto result = qp.modifyQp(
        &qpAttr,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (!result) {
      throw std::runtime_error("Failed to modify QP to RTR state");
    }
  }
  {
    // change QP group state to RTS
    auto qpAttr = makeQpAttrRts();
    auto result = qp.modifyQp(
        &qpAttr,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
            IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (!result) {
      throw std::runtime_error("Failed to modify virtual QP to RTS state");
    }
  }
}

struct BenchmarkSetup {
  std::unique_ptr<IbvEndPoint> sender;
  std::unique_ptr<IbvEndPoint> receiver;
  void* sendBuffer{};
  void* recvBuffer{};
  std::optional<IbvMr> sendMr;
  std::optional<IbvMr> recvMr;

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
      int nicDev1) {
    // Setup IbvEndPoint
    sender = std::make_unique<IbvEndPoint>(nicDev0);
    receiver = std::make_unique<IbvEndPoint>(nicDev1);
    CHECK_NOTNULL(sender.get());
    CHECK_NOTNULL(receiver.get());

    // Change sender and receiver QP state
    auto receiverGid = receiver->device.queryGid(kPortNum, kGidIndex);
    if (!receiverGid) {
      throw std::runtime_error("Failed to query receiver GID");
    }
    sender->changeQpStateToRts(*receiverGid, receiver->qp.getQpNum());

    auto senderGid = sender->device.queryGid(kPortNum, kGidIndex);
    if (!senderGid) {
      throw std::runtime_error("Failed to query sender GID");
    }

    receiver->changeQpStateToRts(*senderGid, sender->qp.getQpNum());

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
    auto sendMrExpected = sender->pd.regMr(sendBuffer, bufferSize, access);
    if (!sendMrExpected) {
      throw std::runtime_error("Failed to register send memory region");
    }
    sendMr = std::move(*sendMrExpected);

    CHECK_EQ(cudaSetDevice(cudaDev1), cudaSuccess);
    CHECK_EQ(cudaMalloc(&recvBuffer, bufferSize), cudaSuccess);
    CHECK_NOTNULL(recvBuffer);
    auto recvMrExpected = receiver->pd.regMr(recvBuffer, bufferSize, access);
    if (!recvMrExpected) {
      throw std::runtime_error("Failed to register receive memory region");
    }
    recvMr = std::move(*recvMrExpected);
  }

  ~BenchmarkSetup() {
    CHECK_EQ(cudaFree(sendBuffer), cudaSuccess);
    CHECK_EQ(cudaFree(recvBuffer), cudaSuccess);
  }
};

} // namespace

//------------------------------------------------------------------------------
// Ibverbx Benchmarks
//------------------------------------------------------------------------------

/**
 * Ibverbx IbEvent RdmaWrite benchmark latency
 */

static void BM_Ibverbx_IbEvent_RdmaWriteWithImm(benchmark::State& state) {
  const size_t bufferSize = state.range(0);
  const int cudaDev0 = 0;
  const int cudaDev1 = 1;
  const int nicDev0 = 0;
  const int nicDev1 = 1;

  try {
    BenchmarkSetup setup(bufferSize, cudaDev0, cudaDev1, nicDev0, nicDev1);

    // Construct send WRs
    int wr_id = 0;
    uint32_t imm_data = static_cast<uint32_t>(bufferSize);
    ibv_sge senderSgList = {
        .addr = (uint64_t)setup.sendBuffer,
        .length = static_cast<uint32_t>(bufferSize),
        .lkey = setup.sendMr->mr()->lkey};
    ibv_send_wr sendWr = {
        .wr_id = static_cast<uint64_t>(wr_id),
        .next = nullptr,
        .sg_list = &senderSgList,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_WRITE_WITH_IMM,
        .send_flags = IBV_SEND_SIGNALED};
    sendWr.wr.rdma.remote_addr = (uint64_t)setup.recvBuffer;
    sendWr.wr.rdma.rkey = setup.recvMr->mr()->rkey;
    sendWr.imm_data = imm_data;
    ibv_send_wr sendWrBad{};

    // Construct recv WRs
    ibv_sge sgList = {};
    ibv_recv_wr recvWr = {
        .wr_id = static_cast<uint64_t>(wr_id),
        .sg_list = &sgList,
        .num_sge = 0};
    ibv_recv_wr recvWrBad{};

    // For event-driven benchmark, we need to track completions
    std::atomic<int> sendCompletions{0};
    std::atomic<int> recvCompletions{0};
    int expectedCompletions = 0;

    // Update the onWc callback to count completions
    setup.sender->cqHandler = std::make_unique<CqHandler>(
        setup.sender->evb,
        setup.sender->ch,
        &setup.sender->cq,
        [&](const ibv_wc& wc) {
          if (wc.status != IBV_WC_SUCCESS) {
            XLOGF(FATAL, "WC failed with status {}", wc.status);
            return;
          }
          if (wc.opcode == IBV_WC_RDMA_WRITE) {
            sendCompletions.fetch_add(1);
          }
        });

    setup.receiver->cqHandler = std::make_unique<CqHandler>(
        setup.receiver->evb,
        setup.receiver->ch,
        &setup.receiver->cq,
        [&](const ibv_wc& wc) {
          if (wc.status != IBV_WC_SUCCESS) {
            XLOGF(FATAL, "WC failed with status {}", wc.status);
            return;
          }
          if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
            setup.receiver->qp.postRecv(&recvWr, &recvWrBad);
            recvCompletions.fetch_add(1);
          }
        });
    for (int i = 0; i < kMaxOutstandingWrs; i++) {
      setup.receiver->qp.postRecv(&recvWr, &recvWrBad);
    }

    // Benchmark the postSend operation
    for (auto _ : state) {
      setup.sender->evb->runInEventBaseThreadAndWait(
          [&]() { setup.sender->qp.postSend(&sendWr, &sendWrBad); });
      expectedCompletions++;

      // Wait for completions for both sender and receiver
      while (
          (sendCompletions.load() < expectedCompletions ||
           recvCompletions.load() < expectedCompletions)) {
      }
    }

    state.SetBytesProcessed(state.iterations() * bufferSize);
  } catch (const std::exception& e) {
    XLOGF(FATAL, "Benchmark setup failed: {}", e.what());
    return;
  }
}

//------------------------------------------------------------------------------
// Benchmarks
//------------------------------------------------------------------------------

const size_t kMinBufferSize = 8 * 1024; // 8 KB
const size_t kMaxBufferSize = 256 * 1024 * 1024; // 256 MB

BENCHMARK(BM_Ibverbx_IbEvent_RdmaWriteWithImm)
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
  ::benchmark::RunSpecifiedBenchmarks();

  // Cleanup
  CHECK_EQ(cudaDeviceReset(), cudaSuccess);

  return 0;
}
