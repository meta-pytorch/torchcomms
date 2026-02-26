// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/tests/dc_utils.h"

#include <chrono>

#include <fmt/format.h>
#include <folly/logging/xlog.h>

namespace ibverbx {

std::ostream& operator<<(std::ostream& out, DcBusinessCard const& card) {
  out << fmt::format(
      "<rank {} qp-num {}, port {}, gid {:x}/{:x} remoteAddr {:x}, rkey {:x}>",
      card.rank,
      card.dctNum,
      card.port,
      card.subnetPrefix,
      card.interfaceId,
      card.remoteAddr,
      card.rkey);
  return out;
}

folly::Expected<IbvSrq, Error> createSRQ(IbvPd& pd, int maxWr, int maxSge) {
  ibv_srq_init_attr srqAttr{};
  srqAttr.attr.max_wr = maxWr;
  srqAttr.attr.max_sge = maxSge;
  return pd.createSrq(&srqAttr);
}

folly::Expected<IbvQp, Error> createDCI(IbvPd& pd, IbvCq& cq) {
  mlx5dv_qp_init_attr dvInitAttr{};
  ibv_qp_init_attr_ex initAttr{};

  initAttr.qp_type = IBV_QPT_DRIVER;
  initAttr.send_cq = cq.cq();
  initAttr.recv_cq = cq.cq();
  initAttr.comp_mask = IBV_QP_INIT_ATTR_PD;

  initAttr.cap.max_send_wr = 1024;
  initAttr.cap.max_send_sge = 1;
  initAttr.comp_mask |= IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
  initAttr.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE;

  dvInitAttr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
  dvInitAttr.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCI;

  return pd.createDcQp(&initAttr, &dvInitAttr);
}

folly::Expected<IbvQp, Error>
createDCT(IbvPd& pd, IbvCq& cq, IbvSrq& srq, uint64_t dcKey) {
  mlx5dv_qp_init_attr dvInitAttr{};
  ibv_qp_init_attr_ex initAttr{};

  initAttr.qp_type = IBV_QPT_DRIVER;
  initAttr.send_cq = cq.cq();
  initAttr.recv_cq = cq.cq();
  initAttr.srq = srq.srq();
  initAttr.comp_mask = IBV_QP_INIT_ATTR_PD;

  dvInitAttr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
  dvInitAttr.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCT;
  dvInitAttr.dc_init_attr.dct_access_key = dcKey;

  return pd.createDcQp(&initAttr, &dvInitAttr);
}

folly::Expected<folly::Unit, Error>
transitionDCIToRts(IbvQp& qp, uint8_t port, ibv_mtu mtu) {
  ibv_qp_attr qpAttr{};

  // RESET -> INIT
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = 0;
  qpAttr.port_num = port;
  auto initResult =
      qp.modifyQp(&qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT);
  if (initResult.hasError()) {
    return folly::makeUnexpected(Error(
        initResult.error().errNum,
        fmt::format(
            "Failed to transition DCI to INIT: {}",
            initResult.error().errStr)));
  }

  // INIT -> RTR
  qpAttr = {};
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = mtu;
  qpAttr.ah_attr.is_global = 1;
  qpAttr.ah_attr.grh.sgid_index = kGidIndex;
  qpAttr.ah_attr.port_num = port;
  auto rtrResult =
      qp.modifyQp(&qpAttr, IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV);
  if (rtrResult.hasError()) {
    return folly::makeUnexpected(Error(
        rtrResult.error().errNum,
        fmt::format(
            "Failed to transition DCI to RTR: {}", rtrResult.error().errStr)));
  }

  // RTR -> RTS
  qpAttr = {};
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = 14;
  qpAttr.retry_cnt = 7;
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  auto rtsResult = qp.modifyQp(
      &qpAttr,
      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
  if (rtsResult.hasError()) {
    return folly::makeUnexpected(Error(
        rtsResult.error().errNum,
        fmt::format(
            "Failed to transition DCI to RTS: {}", rtsResult.error().errStr)));
  }

  return folly::unit;
}

folly::Expected<folly::Unit, Error>
transitionDCTToRtr(IbvQp& qp, uint8_t port, ibv_mtu mtu) {
  ibv_qp_attr qpAttr{};

  // RESET -> INIT (DCT needs access flags for remote operations)
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = 0;
  qpAttr.port_num = port;
  qpAttr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
  auto initResult = qp.modifyQp(
      &qpAttr,
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
  if (initResult.hasError()) {
    return folly::makeUnexpected(Error(
        initResult.error().errNum,
        fmt::format(
            "Failed to transition DCT to INIT: {}",
            initResult.error().errStr)));
  }

  // INIT -> RTR
  qpAttr = {};
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = mtu;
  qpAttr.min_rnr_timer = 12;
  // DCT requires these GRH attributes for RTR transition
  qpAttr.ah_attr.is_global = 1;
  qpAttr.ah_attr.grh.traffic_class = 0;
  qpAttr.ah_attr.grh.flow_label = 0;
  qpAttr.ah_attr.grh.sgid_index = kGidIndex;
  qpAttr.ah_attr.grh.hop_limit = 255;
  qpAttr.ah_attr.port_num = port;
  auto rtrResult = qp.modifyQp(
      &qpAttr,
      IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_MIN_RNR_TIMER | IBV_QP_AV);
  if (rtrResult.hasError()) {
    return folly::makeUnexpected(Error(
        rtrResult.error().errNum,
        fmt::format(
            "Failed to transition DCT to RTR: {}", rtrResult.error().errStr)));
  }

  return folly::unit;
}

folly::Expected<IbvAh, Error> createAddressHandle(
    IbvPd& pd,
    const DcBusinessCard& remoteCard,
    uint8_t sgidIndex) {
  ibv_ah_attr ahAttr{};
  ahAttr.is_global = 1;
  ahAttr.grh.dgid.global.subnet_prefix = remoteCard.subnetPrefix;
  ahAttr.grh.dgid.global.interface_id = remoteCard.interfaceId;
  ahAttr.grh.flow_label = 0;
  ahAttr.grh.sgid_index = sgidIndex;
  ahAttr.grh.hop_limit = 255;
  ahAttr.grh.traffic_class = 0;
  ahAttr.sl = 0;
  ahAttr.src_path_bits = 0;
  ahAttr.port_num = remoteCard.port;

  return pd.createAh(&ahAttr);
}

folly::Expected<folly::Unit, Error> pollCqForCompletions(
    int rank,
    IbvCq& cq,
    int expectedCompletions,
    int timeoutMs) {
  int completedCount = 0;
  auto startTime = std::chrono::steady_clock::now();

  while (completedCount < expectedCompletions) {
    auto maybeWcsVector = cq.pollCq(expectedCompletions);
    if (maybeWcsVector.hasError()) {
      return folly::makeUnexpected(Error(
          maybeWcsVector.error().errNum,
          fmt::format(
              "rank {}: CQ poll failed: {}",
              rank,
              maybeWcsVector.error().errStr)));
    }

    auto numWc = maybeWcsVector->size();
    for (size_t i = 0; i < numWc; ++i) {
      const auto& wc = maybeWcsVector->at(i);
      if (wc.status != IBV_WC_SUCCESS) {
        return folly::makeUnexpected(Error(
            static_cast<int>(wc.status),
            fmt::format(
                "rank {} got WC status {}, opcode={}, wr_id={}, vendor_err={}",
                rank,
                wc.status,
                wc.opcode,
                wc.wr_id,
                wc.vendor_err)));
      }
      completedCount++;
      XLOGF(
          DBG1,
          "Rank {} got WC {}/{}: wr_id={}, opcode={}",
          rank,
          completedCount,
          expectedCompletions,
          wc.wr_id,
          wc.opcode);
    }

    if (completedCount < expectedCompletions && numWc == 0) {
      auto elapsed = std::chrono::steady_clock::now() - startTime;
      auto elapsedMs =
          std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
              .count();
      if (elapsedMs >= timeoutMs) {
        return folly::makeUnexpected(Error(
            ETIMEDOUT,
            fmt::format(
                "rank {}: CQ poll timed out after {}ms, got {}/{} completions",
                rank,
                timeoutMs,
                completedCount,
                expectedCompletions)));
      }
    }
  }

  return folly::unit;
}

} // namespace ibverbx
