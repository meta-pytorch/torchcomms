// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <unistd.h>
#include <memory>
#include <string>
#include <vector>

#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/backends/ib/CtranIbVc.h"
#include "comms/ctran/ibverbx/IbvQpUtils.h"
#include "comms/ctran/utils/Checks.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

#define CTRAN_HARDCODED_MAX_QPS (128)

using namespace ibverbx;
using namespace ctran::ibvwrap;

// Business card describing the local IB connection info.
struct BusCard {
  enum ibverbx::ibv_mtu mtu;
  uint32_t controlQpn;
  uint32_t notifQpn;
  uint32_t atomicQpn;
  uint32_t dataQpn[CTRAN_HARDCODED_MAX_QPS];
  uint8_t ports[CTRAN_MAX_IB_DEVICES_PER_RANK];
  union {
    struct {
      uint64_t spns[CTRAN_MAX_IB_DEVICES_PER_RANK];
      uint64_t iids[CTRAN_MAX_IB_DEVICES_PER_RANK];
    } eth;
    struct {
      uint16_t lids[CTRAN_MAX_IB_DEVICES_PER_RANK];
    } ib;
  } u;
};

// set the max number of QPs,  QP scaling threshold and  VC mode to use for a
// peer based on the topology and corresponding CVARs, if available, otherwise,
// use default value of NCCL_CTRAN_IB_MAX_QPS,
// NCCL_CTRAN_IB_QP_SCALING_THRESHOLD, and NCCL_CTRAN_IB_VC_MODE
inline commResult_t CtranIbVirtualConn::setDefaultQPConfig() {
  maxNumQps_ = NCCL_CTRAN_IB_MAX_QPS;
  qpScalingTh_ = NCCL_CTRAN_IB_QP_SCALING_THRESHOLD;
  vcMode_ = NCCL_CTRAN_IB_VC_MODE;
  maxQpMsgs_ = NCCL_CTRAN_IB_QP_MAX_MSGS;
  ConnectionType connTyp = ConnectionType::NO_STATEX;
  std::vector<std::string>* configList{nullptr};
  if (comm_ && comm_->statex_) {
    if (comm_->statex_->isSameRack(comm_->statex_->rank(), peerRank)) {
      connTyp = ConnectionType::SAME_RACK;
    } else if (comm_->statex_->isSameZone(comm_->statex_->rank(), peerRank)) {
      connTyp = ConnectionType::SAME_ZONE;
      configList = &NCCL_CTRAN_IB_QP_CONFIG_XRACK;
    } else if (comm_->statex_->isSameDc(comm_->statex_->rank(), peerRank)) {
      connTyp = ConnectionType::SAME_DC;
      configList = &NCCL_CTRAN_IB_QP_CONFIG_XZONE;
    } else {
      connTyp = ConnectionType::DIFF_DC;
      configList = &NCCL_CTRAN_IB_QP_CONFIG_XDC;
    }
  } else if (!comm_) {
    connTyp = ConnectionType::CTRAN_EX;
    configList = &NCCL_CTRAN_EX_IB_QP_CONFIG;
  }

  if ((configList != nullptr) && (configList->size() > 0)) {
    FB_CHECKABORT(
        configList->size() == kExpectedQpConfigLength,
        "XRACK, XZONE, XDC QP Config strings must have exactly 4 elements");
    qpScalingTh_ = stoul(configList->at(qpConfigIndex::QP_SCALING_THRESHOLD));
    maxNumQps_ = stoi(configList->at(qpConfigIndex::MAX_QPS));
    if (configList->at(qpConfigIndex::VC_MODE) == "spray") {
      vcMode_ = NCCL_CTRAN_IB_VC_MODE::spray;
    } else {
      FB_CHECKABORT(
          configList->at(qpConfigIndex::VC_MODE) == "dqplb",
          "IBVC mode must be one of spray or dqplb");
      vcMode_ = NCCL_CTRAN_IB_VC_MODE::dqplb;
    }
    maxQpMsgs_ = stoi(configList->at(qpConfigIndex::MAX_QP_MSGS));
  }

  // cannot execeed the hardcoded max number of QPs
  if (maxNumQps_ > CTRAN_HARDCODED_MAX_QPS) {
    CLOGF(
        WARN,
        "CTRAN-IB: CTRAN_MAX_QPS set to more than the hardcoded max value ({} > {}), use {} instead",
        maxNumQps_,
        CTRAN_HARDCODED_MAX_QPS,
        CTRAN_HARDCODED_MAX_QPS);
    maxNumQps_ = CTRAN_HARDCODED_MAX_QPS;
  }

  // assure maxNumQps_ to be a multiple of NCCL_CTRAN_IB_DEVICES_PER_RANK
  if (maxNumQps_ % NCCL_CTRAN_IB_DEVICES_PER_RANK) {
    int originalMaxNumQps = maxNumQps_;
    maxNumQps_ = maxNumQps_ - (maxNumQps_ % NCCL_CTRAN_IB_DEVICES_PER_RANK);
    CLOGF(
        WARN,
        "CTRAN-IB: CTRAN_MAX_QPS is not a multiple of NCCL_CTRAN_IB_DEVICES_PER_RANK  ({} > {}), use {} instead",
        originalMaxNumQps,
        NCCL_CTRAN_IB_DEVICES_PER_RANK,
        maxNumQps_);
  }

  if (maxQpMsgs_ > MAX_SEND_WR) {
    CLOGF(
        WARN,
        "CTRAN-IB: Max messages per QP set to more than the hardcoded max value ({} > {}), use {} instead",
        maxQpMsgs_,
        MAX_SEND_WR,
        MAX_SEND_WR);
    maxQpMsgs_ = MAX_SEND_WR;
  }

  pendingWqeQs_.resize(maxNumQps_);

  // Only logs once per connection type across CtranIbVc instances
  logConnectionConfig(connTyp);

  return commSuccess;
}

std::string CtranIbVirtualConn::connTypeName(ConnectionType connTyp) {
  switch (connTyp) {
    case ConnectionType::NO_STATEX:
      return "NO_STATEX";
    case ConnectionType::SAME_RACK:
      return "SAME_RACK";
    case ConnectionType::SAME_ZONE:
      return "SAME_ZONE";
    case ConnectionType::SAME_DC:
      return "SAME_DC";
    case ConnectionType::DIFF_DC:
      return "DIFF_DC";
    case ConnectionType::CTRAN_EX:
      return "CTRAN_EX";
  }
  return "";
}

std::string CtranIbVirtualConn::vcModeName(enum NCCL_CTRAN_IB_VC_MODE mode) {
  switch (mode) {
    case NCCL_CTRAN_IB_VC_MODE::spray:
      return "SPRAY";
    case NCCL_CTRAN_IB_VC_MODE::dqplb:
      return "DQPLB";
  }
  return "N/A";
}

// Global variable to ensure we only log once per connection type across all
// CtranIbVc instances.
static folly::Synchronized<std::unordered_map<ConnectionType, bool>>
    connectionLogMap;

void CtranIbVirtualConn::logConnectionConfig(ConnectionType connTyp) {
  auto lockedMap = connectionLogMap.wlock();
  auto it = lockedMap->find(connTyp);
  if (it == lockedMap->end()) {
    lockedMap->insert({connTyp, false});
  }
  if (!lockedMap->at(connTyp)) {
    if (comm_ && comm_->statex_) {
      const auto& statex = comm_->statex_.get();
      CLOGF_SUBSYS(
          INFO,
          INIT,
          "CTRAN-IB-VC: QP setting for connection type {} (sameDC {}, sameZone {}, sameRack {}): maxNumQps_={}, qpScalingTh_={}, vcMode_={}, maxQpMsgs_={}, "
          "rank {} peerRank {} commHash {:x} commDesc {}",
          connTypeName(connTyp),
          statex->isSameDc(statex->rank(), peerRank),
          statex->isSameZone(statex->rank(), peerRank),
          statex->isSameRack(statex->rank(), peerRank),
          maxNumQps_,
          qpScalingTh_,
          vcModeName(vcMode_),
          maxQpMsgs_,
          statex->rank(),
          peerRank,
          statex->commHash(),
          statex->commDesc());
    } else {
      CLOGF_SUBSYS(
          INFO,
          INIT,
          "CTRAN-IB-VC: QP setting for connection type {} (no statex): maxNumQps_={}, qpScalingTh_={}, vcMode_={}, maxQpMsgs_={}",
          connTypeName(connTyp),
          maxNumQps_,
          qpScalingTh_,
          vcModeName(vcMode_),
          maxQpMsgs_);
    }
    lockedMap->at(connTyp) = true;
  }
}

commResult_t CtranIbVirtualConn::prepCtrlMsgs() {
  ibverbx::ibv_access_flags access = static_cast<ibverbx::ibv_access_flags>(
      ibverbx::IBV_ACCESS_LOCAL_WRITE | ibverbx::IBV_ACCESS_REMOTE_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_READ);
  this->sendCtrl_.packets_.resize(MAX_RECV_WR);

  const auto totalSize = MAX_RECV_WR * sizeof(CtrlPacket);
  auto maybeSendCtrlMr = devices_[0].ibvPd->regMr(
      (void*)this->sendCtrl_.packets_.data(), totalSize, access);
  FOLLY_EXPECTED_CHECK(maybeSendCtrlMr);
  this->sendCtrl_.ibvMr_ = std::move(*maybeSendCtrlMr);

  this->recvCtrl_.packets_.resize(MAX_RECV_WR);

  auto maybeRecvCtrlMr = devices_[0].ibvPd->regMr(
      (void*)this->recvCtrl_.packets_.data(), totalSize, access);
  FOLLY_EXPECTED_CHECK(maybeRecvCtrlMr);
  this->recvCtrl_.ibvMr_ = std::move(*maybeRecvCtrlMr);

  CLOGF_TRACE(
      INIT,
      "CTRAN-IB-VC: CMsg packets pre-registered to device 0: sendCtrl {}, recvCtrl {}, size {} (packetSize {} * MAX_RECV_WR {})",
      (void*)this->sendCtrl_.packets_.data(),
      (void*)this->recvCtrl_.packets_.data(),
      totalSize,
      sizeof(CtrlPacket),
      MAX_RECV_WR);
  for (int i = 0; i < MAX_RECV_WR; i++) {
    this->sendCtrl_.freePkts_.push_back(this->sendCtrl_.packets_.at(i));
  }

  return commSuccess;
}

commResult_t CtranIbVirtualConn::prepIbvWrs() {
  // send ctrl wr
  memset(&sendCtrlSg_, 0, sizeof(sendCtrlSg_));
  sendCtrlSg_.length = sizeof(CtrlPacket);
  sendCtrlSg_.lkey = this->sendCtrl_.ibvMr_->mr()->lkey;
  memset(&sendCtrlWr_, 0, sizeof(sendCtrlWr_));
  sendCtrlWr_.wr_id = 0;
  sendCtrlWr_.next = nullptr;
  sendCtrlWr_.sg_list = &sendCtrlSg_;
  sendCtrlWr_.num_sge = 1;
  sendCtrlWr_.opcode = ibverbx::IBV_WR_SEND;
  sendCtrlWr_.send_flags = ibverbx::IBV_SEND_SIGNALED;

  // recv ctrl wr
  memset(&recvCtrlSg_, 0, sizeof(recvCtrlSg_));
  recvCtrlSg_.length = sizeof(CtrlPacket);
  recvCtrlSg_.lkey = this->recvCtrl_.ibvMr_->mr()->lkey;
  memset(&recvCtrlWr_, 0, sizeof(recvCtrlWr_));
  recvCtrlWr_.wr_id = 0;
  recvCtrlWr_.next = nullptr;
  recvCtrlWr_.sg_list = &recvCtrlSg_;
  recvCtrlWr_.num_sge = 1;

  // send notify wr
  memset(&sendNotifyWr_, 0, sizeof(sendNotifyWr_));
  sendNotifyWr_.next = nullptr;
  sendNotifyWr_.num_sge = 0;
  sendNotifyWr_.opcode = ibverbx::IBV_WR_RDMA_WRITE_WITH_IMM;
  sendNotifyWr_.send_flags = ibverbx::IBV_SEND_SIGNALED;

  // send put wr
  memset(&sendPutSg_, 0, sizeof(sendPutSg_));
  memset(&sendPutWr_, 0, sizeof(sendPutWr_));
  sendPutWr_.next = nullptr;
  sendPutWr_.sg_list = &sendPutSg_;
  sendPutWr_.num_sge = 1;

  // send get wr
  memset(&sendGetSg_, 0, sizeof(sendGetSg_));
  memset(&sendGetWr_, 0, sizeof(sendGetWr_));
  sendGetWr_.next = nullptr;
  sendGetWr_.sg_list = &sendGetSg_;
  sendGetWr_.num_sge = 1;
  sendGetWr_.opcode = ibverbx::IBV_WR_RDMA_READ;
  sendGetWr_.send_flags = ibverbx::IBV_SEND_SIGNALED;

  // recv notify wr
  memset(&recvNotifyWr_, 0, sizeof(recvNotifyWr_));
  recvNotifyWr_.wr_id = 0;
  recvNotifyWr_.next = nullptr;
  recvNotifyWr_.num_sge = 0;

  // atomic fetch-and-add wr
  memset(&fetchAddSg_, 0, sizeof(fetchAddSg_));
  fetchAddSg_.length = sizeof(uint64_t);
  memset(&fetchAddWr_, 0, sizeof(fetchAddWr_));
  fetchAddWr_.wr_id = 0;
  fetchAddWr_.next = nullptr;
  fetchAddWr_.sg_list = &fetchAddSg_;
  fetchAddWr_.num_sge = 1;
  fetchAddWr_.opcode = ibverbx::IBV_WR_ATOMIC_FETCH_AND_ADD;
  fetchAddWr_.send_flags = ibverbx::IBV_SEND_SIGNALED;

  // atomic set wr
  memset(&amoSetSg_, 0, sizeof(amoSetSg_));
  amoSetSg_.length = sizeof(uint64_t);
  amoSetSg_.lkey = 0;
  memset(&amoSetWr_, 0, sizeof(amoSetWr_));
  amoSetWr_.wr_id = 0;
  amoSetWr_.next = nullptr;
  amoSetWr_.sg_list = &amoSetSg_;
  amoSetWr_.num_sge = 1;
  amoSetWr_.opcode = ibverbx::IBV_WR_RDMA_WRITE;
  amoSetWr_.send_flags = ibverbx::IBV_SEND_SIGNALED | ibverbx::IBV_SEND_INLINE;

  return commSuccess;
}

CtranIbVirtualConn::CtranIbVirtualConn(
    std::vector<CtranIbDevice>& devices,
    int peerRank,
    CtranComm* comm,
    CtranCtrlManager* ctrlMgr,
    uint32_t pgTrafficClass,
    int cudaDev)
    : peerRank(peerRank),
      devices_(devices),
      maxNumQps_(NCCL_CTRAN_IB_MAX_QPS),
      comm_(comm),
      ctrlMgr_(ctrlMgr),
      pgTrafficClass_(pgTrafficClass),
      cudaDev_(cudaDev) {
  // set default QP configs based on topology and user-specified CVARs, if
  // provided
  FB_COMMCHECKTHROW_EX(setDefaultQPConfig(), comm->logMetaData_);
}

CtranIbVirtualConn::~CtranIbVirtualConn() {
  // clean up control messages only if it gets initialized
  if (isReady_) {
    recvCtrl_.packets_.clear();
    sendCtrl_.packets_.clear();
  }

  // Dot not throw exception in destructor to avoid early termination in stack
  // unwind. See discussion in
  // https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
}

std::size_t CtranIbVirtualConn::getBusCardSize() {
  return sizeof(BusCard);
}

commResult_t CtranIbVirtualConn::getLocalBusCard(void* localBusCard) {
  BusCard* busCard = reinterpret_cast<BusCard*>(localBusCard);
  // assuming all devices portAttr
  ibverbx::ibv_port_attr portAttr;
  for (int device = 0; device < NCCL_CTRAN_IB_DEVICES_PER_RANK; device++) {
    auto maybePortAttr =
        devices_[device].ibvDevice->queryPort(devices_[device].port);
    FOLLY_EXPECTED_CHECK(maybePortAttr);
    portAttr = std::move(*maybePortAttr);
    // assuming all devices have the same link layer
    this->linkLayer_ = portAttr.link_layer;

    if (portAttr.max_msg_sz < qpScalingTh_) {
      // Every message we post will be NCCL_CTRAN_IB_QP_SCALING_THRESHOLD or
      // smaller; clamp if this was set beyond what we can support
      CLOGF(
          WARN,
          "CTRAN-IB-VC: QP Scaling threshold {} higher than max message size {}; clamping",
          qpScalingTh_,
          portAttr.max_msg_sz);
      qpScalingTh_ = portAttr.max_msg_sz;
    }

    int qpAccessFlags = ibverbx::IBV_ACCESS_REMOTE_WRITE |
        ibverbx::IBV_ACCESS_LOCAL_WRITE | ibverbx::IBV_ACCESS_REMOTE_READ;

    /* create QPs and set them to INIT state */
    // only create one control and one notify QP per VC connection, create them
    // on the first device
    if (device == 0) {
      auto ibvControlQpCreateResult = createRcQp(
          devices_[device].ibvPd,
          devices_[device].ibvCq->cq(),
          MAX_SEND_WR,
          MAX_RECV_WR);
      FOLLY_EXPECTED_CHECK(ibvControlQpCreateResult);
      ibvControlQp_ = std::move(*ibvControlQpCreateResult);
      auto ibvNotifyQpCreateResult = createRcQp(
          devices_[device].ibvPd,
          devices_[device].ibvCq->cq(),
          MAX_SEND_WR,
          MAX_RECV_WR);
      FOLLY_EXPECTED_CHECK(ibvNotifyQpCreateResult);
      ibvNotifyQp_ = std::move(*ibvNotifyQpCreateResult);
      auto ibvAtomicQpCreateResult = createRcQp(
          devices_[device].ibvPd,
          devices_[device].ibvCq->cq(),
          MAX_SEND_WR,
          MAX_RECV_WR);
      FOLLY_EXPECTED_CHECK(ibvAtomicQpCreateResult);
      ibvAtomicQp_ = std::move(*ibvAtomicQpCreateResult);

      FOLLY_EXPECTED_CHECK(
          initQp(*ibvControlQp_, devices_[device].port, qpAccessFlags));
      FOLLY_EXPECTED_CHECK(
          initQp(*ibvNotifyQp_, devices_[device].port, qpAccessFlags));
      FOLLY_EXPECTED_CHECK(initQp(
          *ibvAtomicQp_,
          devices_[device].port,
          qpAccessFlags | ibverbx::IBV_ACCESS_REMOTE_ATOMIC));
    }
    // maxNumQps_ is always a multiple of NCCL_CTRAN_IB_DEVICES_PER_RANK
    for (int i = 0; i < maxNumQps_ / NCCL_CTRAN_IB_DEVICES_PER_RANK; i++) {
      auto maybeQp = createRcQp(
          devices_[device].ibvPd,
          devices_[device].ibvCq->cq(),
          MAX_SEND_WR,
          MAX_RECV_WR);
      FOLLY_EXPECTED_CHECK(maybeQp);
      FOLLY_EXPECTED_CHECK(
          initQp(*maybeQp, devices_[device].port, qpAccessFlags));
      ibvDataQps_.emplace_back(std::move(*maybeQp));
    }

    if (this->linkLayer_ == ibverbx::IBV_LINK_LAYER_ETHERNET) {
      union ibverbx::ibv_gid gid;
      auto maybeGid = devices_[device].ibvDevice->queryGid(
          devices_[device].port, NCCL_IB_GID_INDEX);
      FOLLY_EXPECTED_CHECK(maybeGid);
      gid = std::move(*maybeGid);
      busCard->u.eth.spns[device] = gid.global.subnet_prefix;
      busCard->u.eth.iids[device] = gid.global.interface_id;
    } else {
      busCard->u.ib.lids[device] = portAttr.lid;
    }
  }
  /* create local business card */
  for (int device = 0; device < NCCL_CTRAN_IB_DEVICES_PER_RANK; device++) {
    busCard->ports[device] = devices_[device].port;
  }
  busCard->controlQpn = ibvControlQp_->qp()->qp_num;
  busCard->notifQpn = ibvNotifyQp_->qp()->qp_num;
  busCard->atomicQpn = ibvAtomicQp_->qp()->qp_num;
  for (int i = 0; i < maxNumQps_; i++) {
    busCard->dataQpn[i] = this->ibvDataQps_[i].qp()->qp_num;
  }
  busCard->mtu = portAttr.active_mtu;
  mtu_ = 128 * (1 << static_cast<int>(portAttr.active_mtu));
  maxMsgSize_ = portAttr.max_msg_sz;

  return commSuccess;
}

commResult_t CtranIbVirtualConn::setupVc(void* remoteBusCard) {
  BusCard* remoteBusCardStruct = reinterpret_cast<BusCard*>(remoteBusCard);

  // allocate and register control messages
  FB_COMMCHECK(prepCtrlMsgs());
  // prepare wrs used in send/recv ctrl, put, notify
  FB_COMMCHECK(prepIbvWrs());

  for (int i = 0; i < maxNumQps_; i++) {
    int ibDevice = getIbDevFromQpIdx(i);
    QpUniqueId qpId =
        std::make_pair(this->ibvDataQps_[i].qp()->qp_num, ibDevice);
    if (qpNumToIdx_.find(qpId) != qpNumToIdx_.end()) {
      CLOGF(
          ERR,
          "CTRAN-IB-VC: QP {} on device {} already exists",
          this->ibvDataQps_[i].qp()->qp_num,
          ibDevice);
      return commInternalError;
    }
    qpNumToIdx_.emplace(qpId, i);
  }

  /* set QP to RTR state for control and notify QP first*/
  RemoteQpInfo remoteQpInfo = {
      .mtu = remoteBusCardStruct->mtu,
      .port = remoteBusCardStruct->ports[0],
      .linkLayer = linkLayer_,
  };

  if (this->linkLayer_ == ibverbx::IBV_LINK_LAYER_ETHERNET) {
    remoteQpInfo.u.eth.spn = remoteBusCardStruct->u.eth.spns[0];
    remoteQpInfo.u.eth.iid = remoteBusCardStruct->u.eth.iids[0];
  } else {
    remoteQpInfo.u.ib.lid = remoteBusCardStruct->u.ib.lids[0];
  };

  remoteQpInfo.qpn = remoteBusCardStruct->controlQpn;
  FOLLY_EXPECTED_CHECK(
      rtrQp(remoteQpInfo, *ibvControlQp_, NCCL_CTRAN_IB_CTRL_TC));
  remoteQpInfo.qpn = remoteBusCardStruct->notifQpn;
  FOLLY_EXPECTED_CHECK(
      rtrQp(remoteQpInfo, *ibvNotifyQp_, NCCL_CTRAN_IB_CTRL_TC));
  remoteQpInfo.qpn = remoteBusCardStruct->atomicQpn;
  FOLLY_EXPECTED_CHECK(
      rtrQp(remoteQpInfo, *ibvAtomicQp_, NCCL_CTRAN_IB_CTRL_TC));
  /* Then, set QP to RTR state for data QPs*/
  for (int i = 0; i < maxNumQps_; i++) {
    remoteQpInfo.qpn = remoteBusCardStruct->dataQpn[i];
    int device = getIbDevFromQpIdx(i);

    remoteQpInfo.port = remoteBusCardStruct->ports[device];
    if (this->linkLayer_ == ibverbx::IBV_LINK_LAYER_ETHERNET) {
      remoteQpInfo.u.eth.spn = remoteBusCardStruct->u.eth.spns[device];
      remoteQpInfo.u.eth.iid = remoteBusCardStruct->u.eth.iids[device];
    } else {
      remoteQpInfo.u.ib.lid = remoteBusCardStruct->u.ib.lids[device];
    };
    // Only use NCCL_CTRAN_IB_CTRL_TC for the control QP; switch back to
    // NCCL_IB_TC for data QPs

    FOLLY_EXPECTED_CHECK(rtrQp(remoteQpInfo, ibvDataQps_[i], pgTrafficClass_));
  }

  /* set QP to RTS state */
  FOLLY_EXPECTED_CHECK(rtsQp(*ibvControlQp_));
  FOLLY_EXPECTED_CHECK(rtsQp(*ibvNotifyQp_));
  FOLLY_EXPECTED_CHECK(rtsQp(*ibvAtomicQp_));
  for (int i = 0; i < maxNumQps_; i++) {
    FOLLY_EXPECTED_CHECK(rtsQp(ibvDataQps_[i]));
  }

  /* post control WQEs */
  for (int i = 0; i < MAX_RECV_WR; i++) {
    FB_COMMCHECK(this->postRecvCtrlMsg(this->recvCtrl_.packets_.at(i)));
    this->recvCtrl_.postedPkts_.push_back(this->recvCtrl_.packets_.at(i));

    // Pre populate recv on notifyQp
    this->postRecvNotifyMsg(kNotifyQpIdx);

    // In case dqplb is used, pre populate recv on dataQps
    for (int j = 0; j < maxNumQps_; j++) {
      FB_COMMCHECK(this->postRecvNotifyMsg(j));
    }
  }

  isReady_ = true;

  return commSuccess;
}
