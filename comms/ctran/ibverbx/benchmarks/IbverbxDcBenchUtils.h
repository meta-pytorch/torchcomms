// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <folly/Expected.h>
#include <folly/logging/xlog.h>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"
#include "comms/ctran/ibverbx/tests/dc_utils.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ibverbx {

// Constants
constexpr int kDefaultCqe = 1024;
constexpr int kDefaultSrqMaxWr = 1024;

// BusinessCard for RC connections (simpler than DC - no DCT number needed)
struct RcBusinessCard {
  uint32_t qpNum{0};
  uint8_t port{0};
  uint64_t subnetPrefix{0};
  uint64_t interfaceId{0};
  uint64_t remoteAddr{0};
  uint32_t rkey{0};
  uint32_t psn{0}; // Packet sequence number for RTR
};

// Base endpoint class with common RDMA resources
class EndPointBase {
 public:
  EndPointBase() = default;
  virtual ~EndPointBase() = default;

  // Non-copyable, movable
  EndPointBase(const EndPointBase&) = delete;
  EndPointBase& operator=(const EndPointBase&) = delete;
  EndPointBase(EndPointBase&&) = default;
  EndPointBase& operator=(EndPointBase&&) = default;

  // Initialize common RDMA resources (device, PD, CQ)
  folly::Expected<folly::Unit, Error> init(int deviceIndex) {
    // Get device list
    auto devices = IbvDevice::ibvGetDeviceList(NCCL_IB_HCA, NCCL_IB_HCA_PREFIX);
    if (!devices) {
      return folly::makeUnexpected(Error(ENODEV, "Failed to get device list"));
    }

    XLOGF(DBG, "Found {} RDMA devices", devices->size());
    for (size_t i = 0; i < devices->size(); ++i) {
      XLOGF(DBG, "  Device {}: {}", i, devices->at(i).device()->name);
    }

    if (deviceIndex >= static_cast<int>(devices->size())) {
      return folly::makeUnexpected(Error(
          EINVAL,
          fmt::format(
              "Device index {} out of range (have {} devices)",
              deviceIndex,
              devices->size())));
    }

    device_ = std::make_unique<IbvDevice>(std::move(devices->at(deviceIndex)));

    XLOGF(DBG, "Using device {}: {}", deviceIndex, device_->device()->name);

    // Allocate PD
    auto pd = device_->allocPd();
    if (!pd) {
      return folly::makeUnexpected(pd.error());
    }
    pd_ = std::make_unique<IbvPd>(std::move(*pd));

    // Create CQ
    auto cq = device_->createCq(kDefaultCqe, nullptr, nullptr, 0);
    if (!cq) {
      return folly::makeUnexpected(cq.error());
    }
    cq_ = std::make_unique<IbvCq>(std::move(*cq));

    // Query GID
    auto gid = device_->queryGid(kPortNum, kGidIndex);
    if (!gid) {
      return folly::makeUnexpected(gid.error());
    }
    gid_ = *gid;

    return folly::unit;
  }

  // Register a memory region
  folly::Expected<IbvMr, Error> registerMr(void* buf, size_t size) {
    auto access = static_cast<ibv_access_flags>(
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ);
    return pd_->regMr(buf, size, access);
  }

  // Poll CQ for a single completion (non-blocking)
  folly::Expected<std::vector<ibv_wc>, Error> pollCq(int maxEntries = 1) {
    return cq_->pollCq(maxEntries);
  }

  // Poll CQ with busy-spin for benchmarking (no sleep, no allocation)
  // Returns false on error or timeout.
  bool pollCqBusySpin(int expectedCompletions, int timeoutMs = 5000) {
    int completed = 0;
    ibv_wc wc{};
    auto* rawCq = cq_->cq();
    auto start = std::chrono::steady_clock::now();

    while (completed < expectedCompletions) {
      int n = rawCq->context->ops.poll_cq(rawCq, 1, &wc);
      if (n < 0) {
        XLOGF(ERR, "CQ poll error: returned {}", n);
        return false;
      }
      if (n == 1) {
        if (wc.status != IBV_WC_SUCCESS) {
          XLOGF(
              ERR,
              "WC error: status={}, opcode={}, vendor_err={}",
              wc.status,
              wc.opcode,
              wc.vendor_err);
          return false;
        }
        completed++;
      } else {
        // Check timeout periodically to avoid infinite spin
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                .count() > timeoutMs) {
          XLOGF(
              ERR,
              "CQ busy-spin timeout: got {}/{} completions",
              completed,
              expectedCompletions);
          return false;
        }
      }
    }
    return true;
  }

  // Poll CQ until we get expected completions (blocking with timeout)
  bool pollCqBlocking(int expectedCompletions, int timeoutMs = 5000) {
    int completed = 0;
    auto start = std::chrono::steady_clock::now();

    while (completed < expectedCompletions) {
      auto result = cq_->pollCq(expectedCompletions - completed);
      if (result.hasError()) {
        XLOGF(ERR, "CQ poll error: {}", result.error().errStr);
        return false;
      }

      for (const auto& wc : *result) {
        if (wc.status != IBV_WC_SUCCESS) {
          XLOGF(
              ERR,
              "WC error: status={}, opcode={}, vendor_err={}",
              wc.status,
              wc.opcode,
              wc.vendor_err);
          return false;
        }
        completed++;
      }

      auto elapsed = std::chrono::steady_clock::now() - start;
      if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
              .count() > timeoutMs) {
        XLOGF(
            ERR,
            "CQ poll timeout: got {}/{} completions",
            completed,
            expectedCompletions);
        return false;
      }

      if (completed < expectedCompletions && result->empty()) {
        // Brief sleep to avoid busy-wait
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    }
    return true;
  }

  IbvDevice& device() {
    return *device_;
  }
  IbvPd& pd() {
    return *pd_;
  }
  IbvCq& cq() {
    return *cq_;
  }
  const ibv_gid& gid() const {
    return gid_;
  }

 protected:
  std::unique_ptr<IbvDevice> device_;
  std::unique_ptr<IbvPd> pd_;
  std::unique_ptr<IbvCq> cq_;
  ibv_gid gid_{};
};

// DC Endpoint: DCI (sender) + DCT (receiver) + SRQ
class DcEndPoint : public EndPointBase {
 public:
  DcEndPoint() = default;

  // Initialize DC-specific resources (after base init)
  folly::Expected<folly::Unit, Error> initDc() {
    // Create SRQ
    auto srqResult = createSRQ(*pd_, kDefaultSrqMaxWr);
    if (!srqResult) {
      return folly::makeUnexpected(Error(
          srqResult.error().errNum,
          "SRQ creation failed: " + srqResult.error().errStr));
    }
    srq_ = std::make_unique<IbvSrq>(std::move(*srqResult));

    // Create DCI
    auto dciResult = createDCI(*pd_, *cq_);
    if (!dciResult) {
      return folly::makeUnexpected(Error(
          dciResult.error().errNum,
          "DCI creation failed: " + dciResult.error().errStr));
    }
    dci_ = std::make_unique<IbvQp>(std::move(*dciResult));

    // Create DCT
    auto dctResult = createDCT(*pd_, *cq_, *srq_);
    if (!dctResult) {
      return folly::makeUnexpected(Error(
          dctResult.error().errNum,
          "DCT creation failed: " + dctResult.error().errStr));
    }
    dct_ = std::make_unique<IbvQp>(std::move(*dctResult));

    // Get extended QP interface
    exQp_ = ibvSymbols.ibv_internal_qp_to_qp_ex(dci_->qp());
    dvQp_ = ibvSymbols.mlx5dv_internal_qp_ex_from_ibv_qp_ex(exQp_);

    if (!exQp_ || !dvQp_) {
      return folly::makeUnexpected(Error(
          ENOTSUP,
          fmt::format(
              "Failed to get extended QP interface: exQp_={}, dvQp_={}",
              exQp_ != nullptr,
              dvQp_ != nullptr)));
    }

    // Transition DCI to RTS
    auto dciTransition = transitionDCIToRts(*dci_, kPortNum, IBV_MTU_4096);
    if (!dciTransition) {
      return folly::makeUnexpected(dciTransition.error());
    }

    // Transition DCT to RTR
    auto dctTransition = transitionDCTToRtr(*dct_, kPortNum, IBV_MTU_4096);
    if (!dctTransition) {
      return folly::makeUnexpected(dctTransition.error());
    }

    return folly::unit;
  }

  // Create business card for exchange
  DcBusinessCard createBusinessCard(void* buf, uint32_t rkey) const {
    return DcBusinessCard{
        .mtu = 5, // IBV_MTU_4096
        .dctNum = dct_->qp()->qp_num,
        .port = kPortNum,
        .subnetPrefix = gid_.global.subnet_prefix,
        .interfaceId = gid_.global.interface_id,
        .rank = 0,
        .remoteAddr = reinterpret_cast<uint64_t>(buf),
        .rkey = rkey,
    };
  }

  // Create address handle for remote DCT
  folly::Expected<IbvAh, Error> createAh(const DcBusinessCard& remoteCard) {
    return createAddressHandle(*pd_, remoteCard);
  }

  // Post RDMA write using extended QP API (plain write, one-sided)
  int postRdmaWrite(
      IbvAh& ah,
      const DcBusinessCard& targetCard,
      ibv_sge& sge,
      uint64_t wrId = 0) {
    ibvSymbols.ibv_internal_wr_start(exQp_);
    exQp_->wr_id = wrId;
    exQp_->wr_flags = IBV_SEND_SIGNALED;
    ibvSymbols.ibv_internal_wr_rdma_write(
        exQp_, targetCard.rkey, targetCard.remoteAddr);
    ibvSymbols.ibv_internal_wr_set_sge_list(exQp_, 1, &sge);
    ibvSymbols.mlx5dv_internal_wr_set_dc_addr(
        dvQp_, ah.ah(), targetCard.dctNum, DC_KEY);
    return ibvSymbols.ibv_internal_wr_complete(exQp_);
  }

  IbvQp& dci() {
    return *dci_;
  }
  IbvQp& dct() {
    return *dct_;
  }
  IbvSrq& srq() {
    return *srq_;
  }

 private:
  std::unique_ptr<IbvSrq> srq_;
  std::unique_ptr<IbvQp> dci_;
  std::unique_ptr<IbvQp> dct_;
  ibv_qp_ex* exQp_{nullptr};
  mlx5dv_qp_ex* dvQp_{nullptr};
};

// RC Endpoint: Standard Reliable Connection QP
class RcEndPoint : public EndPointBase {
 public:
  RcEndPoint() = default;

  // Initialize RC QP
  folly::Expected<folly::Unit, Error> initRc() {
    // Create RC QP
    ibv_qp_init_attr initAttr{};
    initAttr.send_cq = cq_->cq();
    initAttr.recv_cq = cq_->cq();
    initAttr.qp_type = IBV_QPT_RC;
    initAttr.cap.max_send_wr = 1024;
    initAttr.cap.max_recv_wr = 1024;
    initAttr.cap.max_send_sge = 1;
    initAttr.cap.max_recv_sge = 1;

    auto qpResult = pd_->createQp(&initAttr);
    if (!qpResult) {
      return folly::makeUnexpected(qpResult.error());
    }
    qp_ = std::make_unique<IbvQp>(std::move(*qpResult));

    return folly::unit;
  }

  // Transition RC QP to INIT state
  folly::Expected<folly::Unit, Error> transitionToInit() {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = kPortNum;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ;

    int mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

    auto result = qp_->modifyQp(&attr, mask);
    if (result.hasError()) {
      return folly::makeUnexpected(
          Error(result.error().errNum, "Failed to transition QP to INIT"));
    }
    return folly::unit;
  }

  // Transition RC QP to RTR state (needs remote info)
  folly::Expected<folly::Unit, Error> transitionToRtr(
      const RcBusinessCard& remoteCard) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = remoteCard.qpNum;
    attr.rq_psn = remoteCard.psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid.global.subnet_prefix = remoteCard.subnetPrefix;
    attr.ah_attr.grh.dgid.global.interface_id = remoteCard.interfaceId;
    attr.ah_attr.grh.sgid_index = kGidIndex;
    attr.ah_attr.grh.hop_limit = 255;
    attr.ah_attr.grh.traffic_class = 0;
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.dlid = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = kPortNum;

    int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

    auto result = qp_->modifyQp(&attr, mask);
    if (result.hasError()) {
      return folly::makeUnexpected(
          Error(result.error().errNum, "Failed to transition QP to RTR"));
    }
    return folly::unit;
  }

  // Transition RC QP to RTS state
  folly::Expected<folly::Unit, Error> transitionToRts() {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = psn_;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 1;

    int mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
        IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;

    auto result = qp_->modifyQp(&attr, mask);
    if (result.hasError()) {
      return folly::makeUnexpected(
          Error(result.error().errNum, "Failed to transition QP to RTS"));
    }
    return folly::unit;
  }

  // Connect to remote RC endpoint
  folly::Expected<folly::Unit, Error> connect(
      const RcBusinessCard& remoteCard) {
    auto initResult = transitionToInit();
    if (!initResult) {
      return initResult;
    }

    auto rtrResult = transitionToRtr(remoteCard);
    if (!rtrResult) {
      return rtrResult;
    }

    auto rtsResult = transitionToRts();
    if (!rtsResult) {
      return rtsResult;
    }

    return folly::unit;
  }

  // Create business card for exchange
  RcBusinessCard createBusinessCard(void* buf, uint32_t rkey) const {
    return RcBusinessCard{
        .qpNum = qp_->qp()->qp_num,
        .port = kPortNum,
        .subnetPrefix = gid_.global.subnet_prefix,
        .interfaceId = gid_.global.interface_id,
        .remoteAddr = reinterpret_cast<uint64_t>(buf),
        .rkey = rkey,
        .psn = psn_,
    };
  }

  // Post RDMA write
  int postRdmaWrite(
      const RcBusinessCard& targetCard,
      ibv_sge& sge,
      uint64_t wrId = 0) {
    ibv_send_wr wr{};
    wr.wr_id = wrId;
    wr.next = nullptr;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = targetCard.remoteAddr;
    wr.wr.rdma.rkey = targetCard.rkey;

    ibv_send_wr* badWr = nullptr;
    auto result = qp_->postSend(&wr, badWr);
    return result.hasError() ? -1 : 0;
  }

  IbvQp& qp() {
    return *qp_;
  }
  uint32_t psn() const {
    return psn_;
  }

 private:
  std::unique_ptr<IbvQp> qp_;
  uint32_t psn_{0}; // Packet sequence number
};

} // namespace ibverbx
