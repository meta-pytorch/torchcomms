// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/cvars/nccl_cvars.h"

#include "comms/ctran/backends/ib/CtranIbQpUtils.h"

namespace ctran::ib {
folly::Expected<ibverbx::IbvQp, ibverbx::Error> ctranIbQpCreate(
    const ibverbx::IbvPd* ibvPd,
    ibverbx::ibv_cq* cq) {
  ibverbx::ibv_qp_init_attr initAttr;
  memset(&initAttr, 0, sizeof(ibverbx::ibv_qp_init_attr));
  initAttr.send_cq = cq;
  initAttr.recv_cq = cq;
  initAttr.qp_type = ibverbx::IBV_QPT_RC;
  initAttr.sq_sig_all = 0;
  initAttr.cap.max_send_wr = MAX_SEND_WR;
  initAttr.cap.max_recv_wr = MAX_CONTROL_MSGS;
  initAttr.cap.max_send_sge = 1;
  initAttr.cap.max_recv_sge = 1;
  // atomicSet uses inline uint64_t, so max_inline_data should
  // at least be 8 bytes, but Broadcom NIC's driver lib requires
  // max_inline_data to be at least 16 bytes to use IBV_SEND_INLINE,
  // otherwise you get IBV_WC_LOC_QP_OP_ERR.
  initAttr.cap.max_inline_data = 16;
  return ibvPd->createQp(&initAttr);
}

folly::Expected<folly::Unit, ibverbx::Error>
ctranIbQpInit(ibverbx::IbvQp& ibvQp, int port, int qp_access_flags) {
  ibverbx::ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(ibverbx::ibv_qp_attr));
  qpAttr.qp_state = ibverbx::IBV_QPS_INIT;
  qpAttr.pkey_index = NCCL_IB_PKEY;
  qpAttr.port_num = port;
  qpAttr.qp_access_flags = qp_access_flags;
  return ibvQp.modifyQp(
      &qpAttr,
      ibverbx::IBV_QP_STATE | ibverbx::IBV_QP_PKEY_INDEX |
          ibverbx::IBV_QP_PORT | ibverbx::IBV_QP_ACCESS_FLAGS);
}

folly::Expected<folly::Unit, ibverbx::Error> ctranIbQpRTR(
    const CtranIbRemoteQpInfo& remoteQpInfo,
    ibverbx::IbvQp& ibvQp,
    uint8_t trafficClass) {
  ibverbx::ibv_qp_attr qpAttr;

  memset(&qpAttr, 0, sizeof(ibverbx::ibv_qp_attr));
  qpAttr.qp_state = ibverbx::IBV_QPS_RTR;
  qpAttr.path_mtu = remoteQpInfo.mtu;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;

  if (remoteQpInfo.linkLayer == ibverbx::IBV_LINK_LAYER_ETHERNET) {
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid.global.subnet_prefix = remoteQpInfo.u.eth.spn;
    qpAttr.ah_attr.grh.dgid.global.interface_id = remoteQpInfo.u.eth.iid;
    qpAttr.ah_attr.grh.flow_label = 0;
    qpAttr.ah_attr.grh.sgid_index = NCCL_IB_GID_INDEX;
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.traffic_class = trafficClass;
  } else {
    qpAttr.ah_attr.is_global = 0;
    qpAttr.ah_attr.dlid = remoteQpInfo.u.ib.lid;
  }
  qpAttr.ah_attr.sl = NCCL_IB_SL;
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = remoteQpInfo.port;

  qpAttr.dest_qp_num = remoteQpInfo.qpn;

  return ibvQp.modifyQp(
      &qpAttr,
      ibverbx::IBV_QP_STATE | ibverbx::IBV_QP_AV | ibverbx::IBV_QP_PATH_MTU |
          ibverbx::IBV_QP_DEST_QPN | ibverbx::IBV_QP_RQ_PSN |
          ibverbx::IBV_QP_MAX_DEST_RD_ATOMIC | ibverbx::IBV_QP_MIN_RNR_TIMER);
}

folly::Expected<folly::Unit, ibverbx::Error> ctranIbQpRTS(
    ibverbx::IbvQp& ibvQp) {
  ibverbx::ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(ibverbx::ibv_qp_attr));
  qpAttr.qp_state = ibverbx::IBV_QPS_RTS;
  qpAttr.timeout = NCCL_IB_TIMEOUT;
  qpAttr.retry_cnt = NCCL_IB_RETRY_CNT;
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;

  return ibvQp.modifyQp(
      &qpAttr,
      ibverbx::IBV_QP_STATE | ibverbx::IBV_QP_TIMEOUT |
          ibverbx::IBV_QP_RETRY_CNT | ibverbx::IBV_QP_RNR_RETRY |
          ibverbx::IBV_QP_SQ_PSN | ibverbx::IBV_QP_MAX_QP_RD_ATOMIC);
}
} // namespace ctran::ib
