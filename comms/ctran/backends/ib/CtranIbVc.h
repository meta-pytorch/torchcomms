// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_IB_VC_H_
#define CTRAN_IB_VC_H_

#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/backends/ib/CtranIbQpUtils.h"
#include "comms/ctran/backends/ib/CtranIbSingleton.h"
#include "comms/ctran/backends/ib/IbvWrap.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CtranPerf.h"
#include "comms/ctran/utils/Exception.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/ctran/utils/Utils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/LogUtils.h"

#define CTRAN_IB_FAST_PATH_MSG_MAX_SIZE 1073741824LU
// a QpUniqueId is defined by a pair of <qpnum, ibDeviceId>
typedef std::pair<int, int> QpUniqueId;

// Fix-sized payload buffer for IB transport to prepare and register the
// temporary buffers for control messages
constexpr int MAX_PAYLOAD_SIZE{4096};
struct CtrlPacket {
  int type{0}; // for callback check
  size_t size{0}; // size of actual data in payload
  char payload[MAX_PAYLOAD_SIZE];

  inline void
  copyFrom(const int srcType, const void* srcPayload, const size_t srcSize) {
    FB_CHECKABORT(
        srcSize <= sizeof(payload),
        "Unexpected payload size {} > packet max payload size {}",
        srcSize,
        sizeof(payload));

    memcpy(payload, srcPayload, srcSize);
    size = srcSize;
    type = srcType;
  }

  inline void copyTo(void* dstPayload, const size_t dstSize) {
    FB_CHECKABORT(
        size == dstSize,
        "Unexpected packet payload size {} != input payload size {}",
        size,
        dstSize);

    memcpy(dstPayload, payload, dstSize);
  }

  inline void copyFrom(const CtrlPacket& src) {
    type = src.type;
    size = src.size;
    memcpy(payload, src.payload, src.size);
  }

  inline void copyTo(CtrlPacket& dst) {
    dst.type = type;
    dst.size = size;
    memcpy(dst.payload, payload, size);
  }

  inline size_t getPacketSize() const {
    return offsetof(CtrlPacket, payload) +
        size; // transfer header + actual size of payload
  }

  std::string toString() const {
    return fmt::format(
        "addr {} type {} payloadSize {} packetSize {}",
        (void*)this,
        type,
        size,
        getPacketSize());
  }
};

/**
 * Virtual connection internal work request (WR) structure holding a pending
 * or unexpected control message.
 */
struct ControlPendingSendWr {
  // Pending send control message in case no more free msg to use
  int type{0};
  void* payload{nullptr};
  size_t size{0};

  CtranIbRequest& req;

  ControlPendingSendWr(
      int type,
      void* payload,
      size_t size,
      CtranIbRequest& req)
      : type(type), payload(payload), size(size), req(req) {};
  ~ControlPendingSendWr() = default;
  std::string toString() {
    return fmt::format(
        "ControlPendingSendWr: type {}, payload {}, size {}, req {}",
        type,
        payload,
        size,
        (void*)&req);
  }
};

struct ControlPostedRecvWr {
  // Receive control message posted before the incoming message has arrived
  void* payload{nullptr};
  size_t size{0};

  CtranIbRequest& req;

  ControlPostedRecvWr(void* payload, size_t size, CtranIbRequest& req)
      : payload(payload), size(size), req(req) {};
  ~ControlPostedRecvWr() = default;
  std::string toString() {
    return fmt::format(
        "ControlPostedRecvWr: payload {}, size {}, req {}",
        payload,
        size,
        (void*)&req);
  }
};

struct ControlUnexpWr {
  // Unexpected incoming control message, to be matched with a consequent recv.
  // Copied from postedPkt to free up it for next post.
  CtrlPacket packet;
  ControlUnexpWr() = default;
  ~ControlUnexpWr() = default;
  std::string toString() {
    return fmt::format("ControlUnexpWr: packet [{}]", packet.toString());
  }
};

class CtranIb;

using ConnectionType = enum {
  NO_STATEX = 0,
  SAME_RACK = 1,
  SAME_ZONE = 2,
  SAME_DC = 3,
  DIFF_DC = 4,
  CTRAN_EX = 5,
};

namespace {
enum qpConfigIndex {
  QP_SCALING_THRESHOLD = 0,
  MAX_QPS = 1,
  VC_MODE = 2,
  MAX_QP_MSGS = 3,
  TRAFFIC_CLASS = 4, // Only used in NCCL_CTRAN_IB_QP_CONFIG_ALGO
};
static constexpr int kExpectedQpConfigLength = 4;
}; // namespace

struct PutIbMsg {
  const void* sbuf;
  void* dbuf;
  std::size_t len;
  void* ibRegElem;
  CtranIbRemoteAccessKey remoteAccessKey;
  bool notify;
  CtranIbConfig* config;
  CtranIbRequest* req;
};

/**
 * Virtual connection to manage the IB backend connection between two peers
 * and the internal data transfer.
 * Virtual connection functions are not thread-safe, thus require a lock to be
 * acquired via vc->mutex before calling any of the functions.
 */
class CtranIbVirtualConn {
 public:
  // Prepare local resources for the virtual connection.
  // Actual connection happens only when setupVc is called.
  CtranIbVirtualConn(
      std::vector<CtranIbDevice>& devices,
      int peerRank,
      CtranComm* comm,
      CtranCtrlManager* ctrlMgr,
      uint32_t pgTrafficClass,
      int cudaDev);
  ~CtranIbVirtualConn();

  // The data channel may be temporarily unavailable due to run out of local
  // send queue WQE. VC has already interally scheduled an implicit signal to
  // flush out queued WQEs and is waiting for the CQE completion.
  // Caller needs to call this function before posting a put. If it returns
  // false, progress needs to be made to poll CQE completion.
  inline bool canTransferData() {
    for (const auto& q : pendingWqeQs_) {
      if (q.size() < maxQpMsgs_) {
        return true;
      }
    }
    return false;
  }

  // Get the size of local business card so that socket knows the bytes sent
  // to the other rank.
  std::size_t getBusCardSize();

  // Create local IB connection resource (QPs) and get the local business card
  // that describes the local resource. It will be exchanged with the peer via
  // socket (see bootstrapAccept|bootstrapConnect) to setup the remote
  // connection (see setupVc).
  commResult_t getLocalBusCard(void* busCard);

  // Setup the IB connection between two peers.
  // Specifically, it updates the local control and data QPs with remote
  // business card info to establish the connection.
  commResult_t setupVc(void* remoteBusCard);

  // Implementation to send generic control message over the established IB
  // connection. The msg may not be issued and complete at return of this call.
  // The msg will be eventually progressed and complete whenever the VC is
  // progressed. Caller uses request to track the completion.
  //
  // Message matching:
  // - Each control message send either matches with a control message receive,
  //   or trigger a pre-registered callback handler.
  // - For one requires a receive, it matches the control message receive on the
  //   receiver rank as in the same issuing order.
  // - For one triggers a callback, it doesn't consume any receive posted on the
  //   receiver side.
  //
  // Input arguments:
  //   - type: the control message type to be sent
  //   - payload: pointer to the payload of the control message to be sent
  //   - size: size of the payload of the control message to be sent
  // Output arguments:
  //   - req: the request object to track the progress of the control message.
  //          The caller is responsible for request creation and release.
  inline commResult_t isendCtrlMsg(
      const int type,
      const void* payload,
      const size_t size,
      CtranIbRequest& req) {
    return isendCtrlMsgImpl(type, payload, size, req);
  }

  // Implementation to receive a generic control message over the established IB
  // connection. The msg may be queued and not complete at return of this call.
  // The msg will be eventually progressed and complete whenever the VC is
  // progressed. Caller uses request to track the completion.
  //
  // Message matching:
  // - Each control message receive would match with a control message send in
  //   the same issuing order.
  // - The callsite should guanrantee the receiving payload buffer has
  //   sufficient space to store the incoming control message (i.e., size in
  //   irecvCtrlMsg >= size in isendCtrlMsg). Otherwise, it would be treated as
  //   an internal bug and abort the process.
  //
  // Input arguments:
  //   - type: the control message type to be sent
  //   - payload: pointer to the payload of the control message to be sent
  //   - size: size of the payload of the control message to be sent
  // Output arguments:
  //   - req: the request object to track the progress of the control message.
  //          The caller is responsible for request creation and release.
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t
  irecvCtrlMsg(void* payload, const size_t size, CtranIbRequest& req) {
    return irecvCtrlMsgImpl<PerfConfig>(payload, size, req);
  }

  // Implementation to put data from local sbuf to a dbuf in remote rank over
  // the established IB connection.
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t iput(
      const void* sbuf,
      void* dbuf,
      std::size_t len,
      void* ibRegElem,
      CtranIbRemoteAccessKey remoteAccessKey,
      bool notify,
      CtranIbConfig* config,
      CtranIbRequest* req,
      bool fast) {
    return iputImpl<PerfConfig>(
        sbuf, dbuf, len, ibRegElem, remoteAccessKey, notify, config, req, fast);
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t iputBatch(const std::vector<PutIbMsg>& msgs) {
    return iputBatchImpl<PerfConfig>(msgs);
  }

  // Implementation to get data from a sbuf in remote rank to a local dbuf over
  // the established IB connection.
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t iget(
      const void* sbuf, // remote buffer
      void* dbuf, // local buffer
      std::size_t len,
      void* ibRegElem,
      CtranIbRemoteAccessKey remoteAccessKey,
      CtranIbConfig* config,
      CtranIbRequest* req,
      bool fast) {
    return igetImpl<PerfConfig>(
        sbuf, dbuf, len, ibRegElem, remoteAccessKey, config, req, fast);
  }

  inline commResult_t ifetchAndAdd(
      const void* sbuf,
      void* dbuf,
      uint64_t addVal,
      void* ibRegElem,
      struct CtranIbRemoteAccessKey remoteAccessKey,
      CtranIbRequest* req) {
    return ifetchAndAddImpl(
        sbuf, dbuf, addVal, ibRegElem, remoteAccessKey, req);
  }

  inline commResult_t iatomicSet(
      void* dbuf,
      uint64_t val,
      struct CtranIbRemoteAccessKey remoteAccessKey,
      CtranIbRequest* req) {
    return iatomicSetImpl(dbuf, val, remoteAccessKey, req);
  }

  // Implementation to process a compeletion queue element (CQE) that received
  // in ctranIb::progress.
  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t processCqe(
      enum ibverbx::ibv_wc_opcode opcode,
      int qpNum,
      uint32_t immData,
      uint64_t wrId,
      int ibDevice) {
    return processCqeImpl<PerfConfig>(opcode, qpNum, immData, wrId, ibDevice);
  }

  // Implementation to notify the remote peer without data.
  // putId is optionally passed when the notify is triggered by a put w/ notify.
  inline commResult_t notify(
      CtranIbRequest* req,
      std::optional<uint64_t> putId = std::nullopt) {
    pendingNotifies_.push_back(
        std::make_unique<NotifyInfo>(nextWrId_++, req, putId));
    return tryPostImmNotifyMsg();
  }

  // Implementation to check the notification associated with a remote put.
  // It will set notify to true if the notification is received, which
  // indicates the completion of the remote put. commSystemError can be
  // returned if notification encoded in remote put is invalid.
  inline commResult_t checkNotify(bool* notify) {
    FB_CHECKABORT(notifyCount_ >= 0, "notifyCount should not be less than 0");
    if (notifyCount_ > 0) {
      --notifyCount_;
      *notify = true;
    }
    return commSuccess;
  }

  // Implementation to check the notification associated with remote puts.
  // It will deduct pendingNotifyCount_ if the notifications are received, which
  // indicates the completion of the remote puts. commSystemError can be
  // returned if notification encoded in remote put is invalid.
  inline commResult_t checkNotifies(int& pendingNotifyCnt) {
    FB_CHECKABORT(notifyCount_ >= 0, "notifyCount should not be less than 0");
    if (notifyCount_ > 0) {
      int notified = std::min(notifyCount_, pendingNotifyCnt);
      pendingNotifyCnt -= notified;
      notifyCount_ -= notified;
    }
    return commSuccess;
  }

  commResult_t iflush(CtranIbRequest* req);

  // Set the default QP configs, i.e., max number of QPs, QP scaling
  // threshold, and VC mode, for a given peer based on topology or
  // user-specified config, if provided
  inline commResult_t setDefaultQPConfig();

  // Getter function for maxNumQps_
  inline int getMaxNumQp() {
    return maxNumQps_;
  }

  // Getter function for qpScalingTh_
  inline size_t getQpScalingTh() {
    return qpScalingTh_;
  }

  // Getter function for vcMode_
  inline enum NCCL_CTRAN_IB_VC_MODE getVcMode() {
    return vcMode_;
  }

  inline int getMaxQpMsgs() {
    return maxQpMsgs_;
  }

  inline uint32_t getControlQpNum() const {
    return ibvControlQp_->qp()->qp_num;
  }

  inline uint32_t getNotifyQpNum() const {
    return ibvNotifyQp_->qp()->qp_num;
  }

  inline uint32_t getAtomicQpNum() const {
    return ibvAtomicQp_->qp()->qp_num;
  }

  inline std::vector<uint32_t> getDataQpNums() const {
    std::vector<uint32_t> dataQps;
    dataQps.reserve(ibvDataQps_.size());
    for (const auto& dataQp : ibvDataQps_) {
      dataQps.emplace_back(dataQp.getQpNum());
    }
    return dataQps;
  }

  // derive device index from qp idx
  inline int getIbDevFromQpIdx(int qpIdx) {
    return qpIdx / (maxNumQps_ / NCCL_CTRAN_IB_DEVICES_PER_RANK);
  }

  inline bool isNotifyQp(QpUniqueId qpId) const {
    return ibvNotifyQp_->qp()->qp_num == qpId.first && qpId.second == 0;
  }

  inline bool isControlQp(QpUniqueId qpId) const {
    return ibvControlQp_->qp()->qp_num == qpId.first && qpId.second == 0;
  }

  inline bool isAtomicQp(QpUniqueId qpId) const {
    return ibvAtomicQp_->qp()->qp_num == qpId.first && qpId.second == 0;
  }

  // Global rank of remote peer.
  int peerRank;

  // Lock VC before use since it may be accessed by multiple threads
  std::mutex mutex;

 private:
  commResult_t prepCtrlMsgs();
  commResult_t prepIbvWrs();

  inline commResult_t isendCtrlMsgImpl(
      const int type,
      const void* payload,
      const size_t size,
      CtranIbRequest& req) {
    if (this->sendCtrl_.freePkts_.empty()) {
      // FIXME: need refactor to keep the constness if possible
      auto wr = std::make_unique<ControlPendingSendWr>(
          type, (void*)payload, size, req);
      CLOGF_TRACE(
          COLL,
          "CTRAN-IB-VC: enqueue wr [{}] peer {}",
          wr->toString(),
          this->peerRank);
      this->sendCtrl_.enqueuedWrs_.push_back(std::move(wr));
    } else {
      // Copy ctrl msg to pre-registered buffer
      auto& packet = dequeFront(this->sendCtrl_.freePkts_).get();
      packet.copyFrom(type, payload, size);

      CLOGF_TRACE(
          COLL,
          "CTRAN-IB-VC: post packet [{}] req {} peer {}, copied from payload {}",
          packet.toString(),
          (void*)&req,
          this->peerRank,
          payload);
      FB_COMMCHECK(this->postSendCtrlMsg(packet));

      this->sendCtrl_.postedPkts_.push_back(packet);
      this->sendCtrl_.postedReqs_.push_back(req);
    }

    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t
  irecvCtrlMsgImpl(void* payload, const size_t size, CtranIbRequest& req) {
    if (this->recvCtrl_.unexpWrs_.empty()) {
      auto wr = std::make_unique<ControlPostedRecvWr>(payload, size, req);
      CLOGF_TRACE(
          COLL,
          "CTRAN-IB-VC: enqueue wr [{}] peer {}",
          wr->toString(),
          this->peerRank);
      this->recvCtrl_.enqueuedWrs_.push_back(std::move(wr));
    } else {
      auto unexpWr = dequeFront(this->recvCtrl_.unexpWrs_);
      CLOGF_TRACE(
          COLL,
          "CTRAN-IB-VC: matched wr [{}] peer {}",
          unexpWr->toString(),
          this->peerRank);

      // Copy previous unexpected ctrl msg to receive buffer.
      unexpWr->packet.copyTo(payload, size);
      FB_COMMCHECK(req.complete());
    }

    return commSuccess;
  }

  inline int getOpQps(const CtranIbConfig* config) {
    return ctran::utils::getConfigValue(
        config, &CtranIbConfig::numQps, maxNumQps_);
  }

  inline enum NCCL_CTRAN_IB_VC_MODE getOpVcMode(const CtranIbConfig* config) {
    return ctran::utils::getConfigValue(
        config, &CtranIbConfig::vcMode, vcMode_);
  }

  inline int getOpMaxQpMsgs(const CtranIbConfig* config) {
    return ctran::utils::getConfigValue(
        config, &CtranIbConfig::qpMsgs, maxQpMsgs_);
  }

  inline size_t getOpScalingTh(const CtranIbConfig* config) {
    return ctran::utils::getConfigValue(
        config, &CtranIbConfig::qpScalingTh, qpScalingTh_);
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline bool
  isFastPutValid(CtranIbConfig* config, size_t len, size_t numMessages) {
    if (outstandingPuts_.size() > 0 || pendingPuts_.size() > 0) {
      CLOGF(
          INFO,
          "iputFast issued when previous regular puts are still in progress: outstandingPuts {}, pendingPuts {}",
          outstandingPuts_.size(),
          pendingPuts_.size());
      return false;
    }

    if (outstandingFastPuts_.size() + numMessages > maxQpMsgs_) {
      CLOGF(
          ERR,
          "iputFast issued when outstanding fast puts {} > maxQpMsgs_ {}",
          outstandingFastPuts_.size() + numMessages,
          maxQpMsgs_);
      return false;
    }

    auto perPutScalingTh = getOpScalingTh(config);
    uint64_t maxWqeSize = perPutScalingTh > 0 ? perPutScalingTh : len;
    // Lock max WQE size to the range [1MTU, (max size supported by port)]
    maxWqeSize = std::min(maxMsgSize_, std::max(maxWqeSize, mtu_));

    if (len > maxWqeSize) {
      CLOGF(
          INFO, "iputFast issued with len {} > maxWqeSize {}", len, maxWqeSize);
      return false;
    }
    return true;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t processFastWriteWqe(
      ibverbx::ibv_send_wr& sendPutWr,
      ibverbx::ibv_sge& sendPutSg,
      uint64_t putId,
      const PutIbMsg& put,
      int device) {
    auto smrs = reinterpret_cast<std::vector<ibverbx::IbvMr>*>(put.ibRegElem);
    if (smrs == nullptr) {
      CLOGF(
          ERR,
          "CTRAN-IB: memory registration not found for addr {}",
          (void*)put.sbuf);
      return commSystemError;
    }

    auto lkey = (*smrs)[device].mr()->lkey;
    auto rkey = put.remoteAccessKey.rkeys[device];

    sendPutSg.addr = reinterpret_cast<uint64_t>(put.sbuf);
    sendPutSg.length = put.len;
    sendPutSg.lkey = lkey;

    // Question, should we ignore spray vc mode?
    ibverbx::ibv_wr_opcode opcode = ibverbx::IBV_WR_RDMA_WRITE_WITH_IMM;
    sendPutWr.sg_list = &sendPutSg;
    sendPutWr.num_sge = 1;
    sendPutWr.wr_id = nextWrId_++;
    sendPutWr.opcode = opcode;
    sendPutWr.send_flags = ibverbx::IBV_SEND_SIGNALED;
    sendPutWr.imm_data = (1 << kFastPutBit);

    sendPutWr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(put.dbuf);
    sendPutWr.wr.rdma.rkey = rkey;
    sndNxt_ = (sndNxt_ + 1) & kSeqNumMask;

    if (put.notify) {
      sendPutWr.imm_data |= (1 << kNotifyBit);
    }

    outstandingFastPuts_.emplace_back(putId, put.req);
    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t iputBatchImpl(const std::vector<PutIbMsg>& msgs) {
    std::vector<ibverbx::ibv_send_wr> sendBatchWrs(msgs.size());
    std::vector<ibverbx::ibv_sge> sendBatchSges(msgs.size());

    if (msgs.empty()) {
      return commSuccess;
    }

    bool sendChained = true;
    // first check if all the messages can be sent over the fast path
    for (auto& put : msgs) {
      if (!isFastPutValid<PerfConfig>(put.config, put.len, msgs.size())) {
        // Fallback to slow path and non batched writes
        sendChained = false;
        break;
      }
    }

    // Fallback path if chained sends won't work.
    if (!sendChained) {
      CLOGF(
          INFO,
          "CTRAN-IB: fallback to non-chained sends for batch {}",
          msgs.size());
      for (auto& put : msgs) {
        FB_COMMCHECK(
            iputImpl<PerfConfig>(
                put.sbuf,
                put.dbuf,
                put.len,
                put.ibRegElem,
                put.remoteAccessKey,
                put.notify,
                put.config,
                put.req,
                false));
      }
      return commSuccess;
    }

    // to get the performance of a batched write, we have to send it over a
    // single qp. mixing regular writes and batched writes would add yet another
    // mode to ctran ib, instead treat batched writes like a fast write.
    int chosenQp = iputFastQpIdx_;
    int device = getIbDevFromQpIdx(chosenQp);

    for (size_t i = 0; i < msgs.size(); ++i) {
      auto& put = msgs[i];
      const auto putId = getPutId();

      auto& sendPutSg = sendBatchSges[i];
      auto& sendPutWr = sendBatchWrs[i];
      FB_COMMCHECK(
          processFastWriteWqe<PerfConfig>(
              sendPutWr, sendPutSg, putId, put, device));

      if (i == msgs.size() - 1) {
        sendPutWr.next = nullptr;
      } else {
        sendPutWr.next = &sendBatchWrs[i + 1];
      }

      CLOGF(
          INFO,
          "CTRAN-IB-VC: Batch message {} notify {} wrId {}",
          i,
          put.notify,
          sendPutWr.wr_id);
    }

    auto maybeSend =
        ibvDataQps_[chosenQp].postSend(&sendBatchWrs[0], &badSendPutWr_);
    FOLLY_EXPECTED_CHECK(maybeSend);
    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t iputImpl(
      const void* sbuf,
      void* dbuf,
      std::size_t len,
      void* ibRegElem,
      CtranIbRemoteAccessKey remoteAccessKey,
      bool notify,
      CtranIbConfig* config,
      CtranIbRequest* req,
      bool fast) {
    std::vector<uint32_t> lkeys;
    std::vector<uint32_t> rkeys;

    auto smrs = reinterpret_cast<std::vector<ibverbx::IbvMr>*>(ibRegElem);
    if (smrs == nullptr) {
      CLOGF(
          ERR,
          "CTRAN-IB: memory registration not found for addr {}",
          (void*)sbuf);
      return commSystemError;
    }
    for (int device = 0; device < NCCL_CTRAN_IB_DEVICES_PER_RANK; device++) {
      lkeys.push_back((*smrs)[device].mr()->lkey);
      rkeys.push_back(remoteAccessKey.rkeys[device]);
      CLOGF_TRACE(
          COLL,
          "CTRAN-IB-VC: iput sbuf {} dbuf {} len {} rkey {}",
          (void*)sbuf,
          (void*)dbuf,
          len,
          rkeys[device]);
    }

    const auto putId = getPutId();

    // iput fast path: post a single RDMA_WRITE_WITH_IMM WQE on data QP 0
    // To be eligible for the fast path, the following conditions must be met:
    // 1. There is no outstanding or pending regular put requests (there can be
    // outstanding fast put requests)
    // 2. The message size is less than or equal to the maximum WQE size so the
    // put can fit in a single WQE
    // 3. The number of pending WQEs on the fast QP is less than the maximum
    // number of QP messages
    if (fast) {
      if (!isFastPutValid<PerfConfig>(config, len, 1)) {
        return commSystemError;
      }
      PutIbMsg put{
          .sbuf = sbuf,
          .dbuf = dbuf,
          .len = len,
          .ibRegElem = ibRegElem,
          .remoteAccessKey = remoteAccessKey,
          .notify = notify,
          .config = config,
          .req = req};
      int device = getIbDevFromQpIdx(iputFastQpIdx_);

      // sanity check
      CLOGF_TRACE(
          COLL,
          "CTRAN-IB-VC: issue the put through fast path, putId {} sbuf {} dbuf {} len {}",
          putId,
          sbuf,
          dbuf,
          len);

      FB_COMMCHECK(
          processFastWriteWqe<PerfConfig>(
              sendPutWr_, sendPutSg_, putId, put, device));

      auto maybeSend =
          ibvDataQps_[iputFastQpIdx_].postSend(&sendPutWr_, &badSendPutWr_);
      FOLLY_EXPECTED_CHECK(maybeSend);
    } else {
      CLOGF_TRACE(
          COLL,
          "CTRAN-IB-VC: issue the put through regular path, putId {} sbuf {} dbuf {} len {} notify {} vcMode {}",
          putId,
          sbuf,
          dbuf,
          len,
          notify,
          vcModeName(getOpVcMode(config)));
      pendingPuts_.push_back(
          std::make_unique<PutInfo>(
              sbuf,
              dbuf,
              len,
              std::move(lkeys),
              std::move(rkeys),
              notify,
              req,
              config,
              getOpVcMode(config),
              putId));
      FB_COMMCHECK(tryToPostOp(
          pendingPuts_,
          outstandingFastPuts_,
          [&](int qpIdx) { return queueWriteOnQp(qpIdx); },
          iputFastQpIdx_));
    }

    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t igetImpl(
      const void* sbuf, // remote buffer
      void* dbuf, // local buffer
      std::size_t len,
      void* ibRegElem,
      CtranIbRemoteAccessKey remoteAccessKey,
      CtranIbConfig* config,
      CtranIbRequest* req,
      bool fast) {
    std::vector<uint32_t> lkeys;
    std::vector<uint32_t> rkeys;

    auto smrs = reinterpret_cast<std::vector<ibverbx::IbvMr>*>(ibRegElem);
    if (smrs == nullptr) {
      CLOGF(
          ERR,
          "CTRAN-IB: memory registration not found for addr {}",
          (void*)dbuf);
      return commSystemError;
    }
    for (int device = 0; device < NCCL_CTRAN_IB_DEVICES_PER_RANK; device++) {
      lkeys.push_back((*smrs)[device].mr()->lkey);
      rkeys.push_back(remoteAccessKey.rkeys[device]);
      CLOGF_TRACE(
          COLL,
          "CTRAN-IB-VC: iget sbuf {} dbuf {} len {} rkey {}",
          (void*)sbuf,
          (void*)dbuf,
          len,
          rkeys[device]);
    }

    const auto getId = getGetId();

    // iget fast path: post a single RDMA_READ WQE on data QP 0
    // To be eligible for the fast path, the following conditions must be met:
    // 1. There is no outstanding or pending regular get requests (there can be
    // outstanding fast get requests)
    // 2. The message size is less than or equal to the maximum WQE size so the
    // get can fit in a single WQE
    // 3. the number of pending WQEs on the fast QP is less than the maximum
    // number of QP messages
    if (fast) {
      // sanity check
      // all previous spray messages should be completed
      if (outstandingGets_.size() > 0 || pendingGets_.size() > 0) {
        CLOGF(
            ERR,
            "igetFast issued when previous regular gets are still in progress: outstandingGets {}, pendingGets {}",
            outstandingGets_.size(),
            pendingGets_.size());
        return commSystemError;
      }

      auto perGetScalingTh = getOpScalingTh(config);
      uint64_t maxWqeSize = perGetScalingTh > 0 ? perGetScalingTh : len;
      // Lock max WQE size to the range [1MTU, (max size supported by port)]
      maxWqeSize = std::min(maxMsgSize_, std::max(maxWqeSize, mtu_));

      if (len > maxWqeSize) {
        CLOGF(
            ERR,
            "igetFast issued with len {} > maxWqeSize {}",
            len,
            maxWqeSize);
        return commSystemError;
      }

      if (outstandingFastGets_.size() >= maxQpMsgs_) {
        CLOGF(
            ERR,
            "igetFast issued when outstanding fast gets {} > maxQpMsgs_ {}",
            outstandingFastGets_.size(),
            maxQpMsgs_);
        return commSystemError;
      }
      int device = getIbDevFromQpIdx(igetFastQpIdx_);

      sendGetSg_.addr = reinterpret_cast<uint64_t>(dbuf);
      sendGetSg_.length = len;
      sendGetSg_.lkey = lkeys[device];

      sendGetWr_.wr_id = nextWrId_++;

      sendGetWr_.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(sbuf);
      sendGetWr_.wr.rdma.rkey = rkeys[device];

      outstandingFastGets_.emplace_back(getId, req);
      auto maybeSend =
          ibvDataQps_[igetFastQpIdx_].postSend(&sendGetWr_, &badSendGetWr_);
      FOLLY_EXPECTED_CHECK(maybeSend);
    } else {
      CLOGF_TRACE(
          COLL,
          "CTRAN-IB-VC: issue the get through regular path, getId {} dbuf {} sbuf {} len {} vcMode {}",
          getId,
          dbuf,
          sbuf,
          len,
          vcModeName(getOpVcMode(config)));
      pendingGets_.push_back(
          std::make_unique<GetInfo>(
              sbuf,
              dbuf,
              len,
              std::move(lkeys),
              std::move(rkeys),
              req,
              config,
              getOpVcMode(config),
              getId));
      FB_COMMCHECK(tryToPostOp(
          pendingGets_,
          outstandingFastGets_,
          [&](int qpIdx) { return queueReadOnQp(qpIdx); },
          igetFastQpIdx_));
    }

    return commSuccess;
  }

  inline commResult_t ifetchAndAddImpl(
      const void* sbuf,
      void* dbuf,
      uint64_t addVal,
      void* ibRegElem,
      struct CtranIbRemoteAccessKey remoteAccessKey,
      CtranIbRequest* req) {
    auto smrs = reinterpret_cast<std::vector<ibverbx::IbvMr>*>(ibRegElem);
    if (smrs == nullptr) {
      CLOGF(
          ERR,
          "CTRAN-IB: memory registration not found for addr {}",
          (void*)sbuf);
      return commSystemError;
    }
    // For atomic operations, we always use device 0
    int device = 0;
    uint32_t lkey = (*smrs)[device].mr()->lkey;
    uint32_t rkey = remoteAccessKey.rkeys[device];
    CLOGF_TRACE(
        COLL,
        "CTRAN-IB-VC: ifetchAndAdd sbuf {} dbuf {} addVal {} rkey {}",
        (void*)sbuf,
        (void*)dbuf,
        addVal,
        rkey);
    // Set up the send work request
    fetchAddSg_.addr = reinterpret_cast<uint64_t>(sbuf);
    fetchAddSg_.lkey = lkey;
    fetchAddWr_.wr.atomic.remote_addr = reinterpret_cast<uint64_t>(dbuf);
    fetchAddWr_.wr.atomic.rkey = rkey;
    fetchAddWr_.wr.atomic.compare_add = addVal;
    // Track the request
    pendingAtomicReqs_.push_back(req);
    // Post the atomic operation on the atomicQp
    FOLLY_EXPECTED_CHECK(ibvAtomicQp_->postSend(&fetchAddWr_, &badFetchAddWr_));
    return commSuccess;
  }

  inline commResult_t iatomicSetImpl(
      void* dbuf,
      uint64_t val,
      struct CtranIbRemoteAccessKey remoteAccessKey,
      CtranIbRequest* req) {
    // For atomic operations, we always use device 0
    int device = 0;
    uint32_t rkey = remoteAccessKey.rkeys[device];
    CLOGF_TRACE(
        COLL,
        "CTRAN-IB-VC: iatomicSet dbuf {} val {} rkey {}",
        (void*)dbuf,
        val,
        rkey);
    amoSetSg_.addr = reinterpret_cast<uint64_t>(&val);
    amoSetWr_.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(dbuf);
    amoSetWr_.wr.rdma.rkey = rkey;
    // Track the request
    pendingAtomicReqs_.push_back(req);
    // Post the atomic operation on the atomicQp
    FOLLY_EXPECTED_CHECK(ibvAtomicQp_->postSend(&amoSetWr_, &badAmoSetWr_));
    return commSuccess;
  }

  inline commResult_t postEnqueuedSendWr() {
    auto wr = dequeFront(sendCtrl_.enqueuedWrs_);
    auto& packet = dequeFront(sendCtrl_.freePkts_).get();
    CtranIbRequest* req = nullptr;

    // Copy ctrl msg to pre-registered buffer
    packet.copyFrom(wr->type, wr->payload, wr->size);
    req = &wr->req;

    CLOGF_TRACE(
        COLL,
        "CTRAN-IB-VC: post packet [{}] peer {}, copied from wr [{}]",
        packet.toString(),
        this->peerRank,
        wr->toString());

    FB_COMMCHECK(this->postSendCtrlMsg(packet));

    this->sendCtrl_.postedPkts_.push_back(packet);
    this->sendCtrl_.postedReqs_.push_back(*req);
    return commSuccess;
  }

  inline commResult_t enqueueUnexpWr(CtrlPacket& packet) {
    auto wr = std::make_unique<ControlUnexpWr>();
    packet.copyTo(wr->packet);
    CLOGF_TRACE(
        COLL,
        "CTRAN-IB-VC: received and enqueued unexp wr [{}] peer {}",
        wr->toString(),
        this->peerRank);
    this->recvCtrl_.unexpWrs_.emplace_back(std::move(wr));
    return commSuccess;
  }

  inline commResult_t matchRecvWr(CtrlPacket& packet) {
    auto wr = dequeFront(this->recvCtrl_.enqueuedWrs_);
    packet.copyTo(wr->payload, wr->size);
    CLOGF_TRACE(
        COLL,
        "CTRAN-IB-VC: received and matched wr [{}] peer {}",
        wr->toString(),
        peerRank);
    FB_COMMCHECK(wr->req.complete());
    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t processCqeImpl(
      enum ibverbx::ibv_wc_opcode opcode,
      int qpNum,
      uint32_t immData,
      uint64_t wrId,
      int ibDevice) {
    QpUniqueId qpId = std::make_pair(qpNum, ibDevice);
    switch (opcode) {
      case ibverbx::IBV_WC_SEND: {
        // Complete the front sendCtrl request and put packet to free queue
        auto req = dequeFront(this->sendCtrl_.postedReqs_);
        FB_COMMCHECK(req.get().complete());

        auto& packet = dequeFront(this->sendCtrl_.postedPkts_).get();
        this->sendCtrl_.freePkts_.push_back(packet);

        // Now we have free cmsg; issue a previously enqueued request
        if (!this->sendCtrl_.enqueuedWrs_.empty()) {
          FB_COMMCHECK(postEnqueuedSendWr());
        }
      } break;

      case ibverbx::IBV_WC_RECV: {
        auto& packet = dequeFront(this->recvCtrl_.postedPkts_).get();
        if (this->ctrlMgr_ && this->ctrlMgr_->hasCb(packet.type)) {
          // This is a control message with registered callback.
          // Handle it in callback without matching recv
          CLOGF_TRACE(
              COLL,
              "CTRAN-IB-VC: received and invoke callback for packet [{}] peer {}",
              packet.toString(),
              this->peerRank);
          FB_COMMCHECK(this->ctrlMgr_->runCb(
              this->peerRank, packet.type, packet.payload));
        } else if (this->recvCtrl_.enqueuedWrs_.empty()) {
          // No queued receive msg. This is an unexpected message.
          // Copy received ctrl msg to unexp buffer
          FB_COMMCHECK(enqueueUnexpWr(packet));
        } else {
          // Match it with the queued recv msg.
          FB_COMMCHECK(matchRecvWr(packet));
        }
        // packet is now free, re-post for next message
        FB_COMMCHECK(this->postRecvCtrlMsg(packet));
        this->recvCtrl_.postedPkts_.push_back(packet);
      } break;

      case ibverbx::IBV_WC_RDMA_WRITE: {
        if (isAtomicQp(qpId)) {
          FB_COMMCHECK(atomicComplete());
        } else {
          FB_COMMCHECK(writeComplete<PerfConfig>(qpNum, wrId, ibDevice));
        }
      } break;

      case ibverbx::IBV_WC_RDMA_READ: {
        FB_COMMCHECK(readComplete<PerfConfig>(qpNum, wrId, ibDevice));
      } break;

      case ibverbx::IBV_WC_RECV_RDMA_WITH_IMM: {
        int idx = isNotifyQp(qpId) ? kNotifyQpIdx : this->qpNumToIdx_.at(qpId);
        writeImmRcvd<PerfConfig>(
            idx, immData); // Handle notification of receiver as necessary
        // Ensure RecvQ stays full
        FB_COMMCHECK(this->postRecvNotifyMsg(idx));
      } break;

      case ibverbx::IBV_WC_FETCH_ADD: {
        FB_CHECKABORT(
            isAtomicQp(qpId),
            "The qpNum {} on device {} of FAdd doesn't match atomicQpNum {} and atomicDevice {}",
            qpNum,
            ibDevice,
            getAtomicQpNum(),
            0);
        FB_COMMCHECK(atomicComplete());
      } break;

      default:
        CLOGF(ERR, "CTRAN-IB: Found unknown opcode: {}", opcode);
        return commSystemError;
    }

    return commSuccess;
  }

  // Always post recv buffer with max size
  inline commResult_t postRecvCtrlMsg(CtrlPacket& packet) {
    recvCtrlSg_.addr =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&packet));
    auto maybeRecv = ibvControlQp_->postRecv(&recvCtrlWr_, &badRecvCtrlWr_);
    FOLLY_EXPECTED_CHECK(maybeRecv);
    return commSuccess;
  }

  // Post send with actual payload size
  inline commResult_t postSendCtrlMsg(const CtrlPacket& packet) {
    sendCtrlSg_.addr =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&packet));
    sendCtrlSg_.length = packet.getPacketSize();
    auto maybeSend = ibvControlQp_->postSend(&sendCtrlWr_, &badSendCtrlWr_);
    FOLLY_EXPECTED_CHECK(maybeSend);

    return commSuccess;
  }

  inline commResult_t postRecvNotifyMsg(int qpIdx = -1) {
    auto qp = qpIdx == kNotifyQpIdx ? &(*this->ibvNotifyQp_)
                                    : &this->ibvDataQps_[qpIdx];

    auto maybeRecv = qp->postRecv(&recvNotifyWr_, &badRecvNotifyWr_);
    FOLLY_EXPECTED_CHECK(maybeRecv);

    return commSuccess;
  }

  inline commResult_t tryPostImmNotifyMsg() {
    if (outstandingNotifies_.size() >= ::ctran::ib::MAX_SEND_WR ||
        pendingNotifies_.size() == 0) {
      return commSuccess;
    }
    std::unique_ptr<NotifyInfo> notifyInfo =
        std::move(pendingNotifies_.front());
    pendingNotifies_.pop_front();

    CLOGF_TRACE(
        COLL,
        "CTRAN-IB-VC: post notify wrId {} req {} ({} pending, {} out)",
        notifyInfo->wrId,
        (void*)notifyInfo->req,
        pendingNotifies_.size(),
        outstandingNotifies_.size() + 1);

    sendNotifyWr_.wr_id = notifyInfo->wrId;
    auto maybeSend =
        this->ibvNotifyQp_->postSend(&sendNotifyWr_, &badSendNotifyWr_);
    FOLLY_EXPECTED_CHECK(maybeSend);

    outstandingNotifies_.push_back(std::move(notifyInfo));

    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline void writeImmRcvd(int idx, uint32_t immData) {
    bool onNotifyQp = idx == kNotifyQpIdx;

    if (onNotifyQp) {
      // A notify on the notifyQP is an instant update to the notify count.
      ++notifyCount_;
      return;
    }

    // In dqplb mode, if not on fast path, we need to track whether we've
    // received all previous writes before firing the notification for a write
    // with its notify bit set; on fast path, we don't need to track this, as
    // all writes are from same qp and are ordered.
    //
    // Every write that is posted by the sender has a sequential sequence
    // number starting at 0. This sequence number is written into the bits of
    // the immediate data masked by kSeqNumMask. If a given write is the final
    // write of a PUT with remote notify, the kNotifyBit of the immediate is
    // also set.
    //
    // If sequence # N has its notify bit set, we need to wait until all
    // sequence numbers <= N have been received. We do this in the following
    // way:
    //   - Sequence numbers that are received are written into the map
    //     rcvdSeqNums_ with the value being whether their notify bit was set
    //   - We keep track of the lowest sequence number not yet seen in rcvNxt_
    //   - If rcvNxt_ is present in the map, that means we have received it
    //   and we can increment rcvNxt_, firing a notification if the map value
    //   says so
    //   - We then continue until rcvNxt_ is again the lowest sequence number
    //   not yet received
    bool fastPut = immData & (1 << kFastPutBit);
    bool notify = immData & (1 << kNotifyBit);

    CLOGF_TRACE(
        COLL,
        "CTRAN-IB-VC: Received immData {} on qp {} fastPut {} notify {} notifyCount {}",
        immData,
        idx,
        fastPut,
        notify,
        notifyCount_);

    if (fastPut) {
      FB_CHECKABORT(
          idx == iputFastQpIdx_,
          "CTRAN-IB-VC: Received iputFast message from unexpected queue pair {}, expect {}",
          idx,
          this->iputFastQpIdx_);
      // For fast put, update rcvNxt_ to assist slow puts in tracking all
      // previously received dqplb puts.
      rcvNxt_ = (rcvNxt_ + 1) & kSeqNumMask;
      notifyCount_ += notify;
    } else {
      rcvdSeqNums_[immData & kSeqNumMask] = immData & (1U << kNotifyBit);
    }
    auto it = rcvdSeqNums_.find(rcvNxt_);
    while (it != rcvdSeqNums_.end()) {
      if (it->second) {
        ++notifyCount_;
      }
      rcvdSeqNums_.erase(it);
      rcvNxt_ = (rcvNxt_ + 1) & kSeqNumMask;
      it = rcvdSeqNums_.find(rcvNxt_);
    }
  }

  template <typename PendingOpQueue, typename FastOpQueue, typename QueueOpFunc>
  inline commResult_t tryToPostOp(
      PendingOpQueue& pendingOpQueue,
      FastOpQueue& outstandingFastQueue,
      QueueOpFunc queueOpOnQp,
      int fastQpIdx) {
    bool sent = true;
    while (sent && (pendingOpQueue.size() > 0)) {
      sent = false;
      auto& op = pendingOpQueue.front();
      auto qps = std::min(getOpQps(op->config), maxNumQps_);
      auto maxmsgs = std::min(getOpMaxQpMsgs(op->config), maxQpMsgs_);
      for (int i = 0; i < qps; i++) {
        auto pendingWqeSize = pendingWqeQs_.at(qpIdxRR_).size();
        if (qpIdxRR_ == fastQpIdx) {
          pendingWqeSize += outstandingFastQueue.size();
        }
        if (pendingWqeSize < maxmsgs) {
          FB_COMMCHECK(queueOpOnQp(qpIdxRR_));
          sent = true;
        }
        qpIdxRR_ = (qpIdxRR_ + 1) % qps;
        if (pendingOpQueue.size() == 0) {
          break;
        }
      }
    }
    return commSuccess;
  }

  inline commResult_t queueWriteOnQp(int i) {
    if (pendingPuts_.size() == 0) {
      return commSuccess;
    }
    int device = getIbDevFromQpIdx(i);
    auto& put = pendingPuts_.front();

    uint64_t remData = put->len - put->offset;
    // Write WQEs of max size qpScalingTh_ unless the threshold is 0. If it
    // is, divide message equally among QPs.
    auto perPutScalingTh = getOpScalingTh(put->config);
    auto putqps = std::min(getOpQps(put->config), maxNumQps_);
    uint64_t maxWqeSize =
        perPutScalingTh > 0 ? perPutScalingTh : put->len / putqps;
    // Lock max WQE size to the range [1MTU, (max size supported by port)]
    maxWqeSize = std::min(maxMsgSize_, std::max(maxWqeSize, mtu_));
    auto toSend = std::min(remData, maxWqeSize);
    bool finalWriteOfPut = toSend == remData;

    sendPutSg_.addr = reinterpret_cast<uint64_t>(put->sbuf) + put->offset;
    sendPutSg_.length = toSend;
    sendPutSg_.lkey = put->lkeys[device];

    sendPutWr_.wr_id = nextWrId_++;
    sendPutWr_.send_flags = ibverbx::IBV_SEND_SIGNALED;

    sendPutWr_.wr.rdma.remote_addr =
        reinterpret_cast<uint64_t>(put->dbuf) + put->offset;
    sendPutWr_.wr.rdma.rkey = put->rkeys[device];

    // RDMA_WRITE if in spray mode, or dqplb mode without notify
    if (put->vcMode == NCCL_CTRAN_IB_VC_MODE::spray || !put->notify) {
      sendPutWr_.opcode = ibverbx::IBV_WR_RDMA_WRITE;
    } else {
      // When in dqplb mode with notify, we need track when all packets have
      // been received on the receiver side. Because packets may be issued
      // through different QPs, we use RDMA_WRITE_WITH_IMM + per-packet seqNum
      // to track all packets belonging to the Put on receiver side, and use
      // notify bit to mark the seqNum of final write.
      // Also see writeImmRcvd() for receiver side logic.
      sendPutWr_.opcode = ibverbx::IBV_WR_RDMA_WRITE_WITH_IMM;
      sendPutWr_.imm_data = sndNxt_;
      sndNxt_ = (sndNxt_ + 1) & kSeqNumMask;
      if (finalWriteOfPut) {
        sendPutWr_.imm_data |= (1U << kNotifyBit);
      }
    }

    pendingWqeQs_.at(i).emplace_back(sendPutWr_.wr_id, put.get());
    put->outstandingWqes++;
    CLOGF_TRACE(
        COLL,
        "CTRAN-IB-VC: postSend occurred on qpIdx {} device {} wrId {} toSend {} putLen {}",
        i,
        device,
        sendPutWr_.wr_id,
        toSend,
        put->len);

    auto maybeSend = ibvDataQps_[i].postSend(&sendPutWr_, &badSendPutWr_);
    FOLLY_EXPECTED_CHECK(maybeSend);
    put->offset += toSend;
    if (finalWriteOfPut) {
      // If this was the final put, move the PutInfo descriptor from pending
      // to outstanding
      if (put->offset != put->len) {
        throw ctran::utils::Exception("Mismatch of length", commInternalError);
      }
      outstandingPuts_.push_back(std::move(pendingPuts_.front()));
      pendingPuts_.pop_front();
    }

    return commSuccess;
  }

  inline commResult_t queueReadOnQp(int i) {
    if (pendingGets_.size() == 0) {
      return commSuccess;
    }
    int device = getIbDevFromQpIdx(i);
    auto& get = pendingGets_.front();

    uint64_t remData = get->len - get->offset;
    // Write WQEs of max size qpScalingTh_ unless the threshold is 0. If it
    // is, divide message equally among QPs.
    auto perGetScalingTh = getOpScalingTh(get->config);
    auto getqps = std::min(getOpQps(get->config), maxNumQps_);
    uint64_t maxWqeSize =
        perGetScalingTh > 0 ? perGetScalingTh : get->len / getqps;
    // Lock max WQE size to the range [1MTU, (max size supported by port)]
    maxWqeSize = std::min(maxMsgSize_, std::max(maxWqeSize, mtu_));
    auto toSend = std::min(remData, maxWqeSize);
    bool finalReadOfGet = toSend == remData;

    sendGetSg_.addr = reinterpret_cast<uint64_t>(get->dbuf) + get->offset;
    sendGetSg_.length = toSend;
    sendGetSg_.lkey = get->lkeys[device];

    sendGetWr_.wr_id = nextWrId_++;

    sendGetWr_.wr.rdma.remote_addr =
        reinterpret_cast<uint64_t>(get->sbuf) + get->offset;
    sendGetWr_.wr.rdma.rkey = get->rkeys[device];

    CLOGF_TRACE(
        COLL,
        "CTRAN-IB-VC: queueReadOnQp post SendGetWr {}, local addr {} offest {} length {} remote addr {}, offsite {}",
        sendGetWr_.wr_id,
        get->dbuf,
        get->offset,
        toSend,
        get->sbuf,
        get->offset);

    pendingWqeQs_.at(i).emplace_back(sendGetWr_.wr_id, get.get());
    get->outstandingWqes++;
    auto maybeSend = ibvDataQps_[i].postSend(&sendGetWr_, &badSendGetWr_);
    FOLLY_EXPECTED_CHECK(maybeSend);
    get->offset += toSend;

    if (finalReadOfGet) {
      // If this was the final get, move the getInfo descriptor from pending
      // to outstanding
      if (get->offset != get->len) {
        throw ctran::utils::Exception("Mismatch of length", commInternalError);
      }
      outstandingGets_.push_back(std::move(pendingGets_.front()));
      pendingGets_.pop_front();
    }

    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t writeComplete(int qpNum, uint64_t wrId, int ibDevice) {
    // This is the final notify packet after all data packets
    if (isNotifyQp(std::make_pair(qpNum, ibDevice))) {
      FB_CHECKABORT(
          outstandingNotifies_.size() > 0,
          "Got a send CQE for a notification that wasn't sent");

      std::unique_ptr<NotifyInfo> notifyInfo =
          std::move(outstandingNotifies_.front());
      outstandingNotifies_.pop_front();
      FB_CHECKABORT(
          wrId == notifyInfo->wrId,
          "CQE wrId={} did not match queued wrId={}",
          wrId,
          notifyInfo->wrId);

      CLOGF_TRACE(
          COLL,
          "CTRAN-IB-VC: Complete notify for wrId {} req {} {}",
          notifyInfo->wrId,
          (void*)notifyInfo->req,
          notifyInfo->putId.has_value()
              ? fmt::format("with putId {}", notifyInfo->putId.value())
              : "without put");
      if (notifyInfo->req != nullptr) {
        // Req is nullptr if a put was posted with remote notification but no
        // local notification (i.e. no CtranIbRequest passed to ::iput).
        FB_COMMCHECK(notifyInfo->req->complete());
      }
      FB_COMMCHECK(tryPostImmNotifyMsg());
    } else {
      int qpIdx = this->qpNumToIdx_.at(std::make_pair(qpNum, ibDevice));

      // We always process outstandingFastPuts_ first.
      // Notice this puts restriction on iput() and inputFast(). All iput() must
      // complete before inputFast() can be called.
      if (qpIdx == 0 && outstandingFastPuts_.size() > 0) {
        // fire the completion on the req
        auto& putInfo = outstandingFastPuts_.front();
        if (putInfo.req != nullptr) {
          FB_COMMCHECK(putInfo.req->complete());
        }
        CLOGF_TRACE(
            COLL,
            "CTRAN-IB-VC: Complete fast path put for req {} putId {}",
            (void*)putInfo.req,
            putInfo.putId);

        outstandingFastPuts_.pop_front();
        if (outstandingFastPuts_.empty()) {
          FB_COMMCHECK(burnDownPuts());
        }
        // Post next data if exists
        FB_COMMCHECK(queueWriteOnQp(qpIdx));
      } else {
        auto& q = pendingWqeQs_.at(qpIdx);
        FB_CHECKABORT(q.size() > 0, "Unexpected empty pendingWqeQs");

        auto wqeInfo = dequeFront(q);
        auto& putInfo = *wqeInfo.putInfo;
        FB_CHECKABORT(
            wqeInfo.wrId == wrId,
            "wrId mismatch: {} != {}, qpIdx={}, outstandingFastPuts={}, outstandingPuts={}, pendingWqeQs={}",
            wqeInfo.wrId,
            wrId,
            qpIdx,
            outstandingFastPuts_.size(),
            outstandingPuts_.size(),
            pendingWqeQs_.size());
        FB_CHECKABORT(
            putInfo.outstandingWqes > 0,
            "Got CQE for put with 0 outstanding WQEs");

        --putInfo.outstandingWqes;

        // Check if put is complete and, if so, update req status
        FB_COMMCHECK(burnDownPuts());
        FB_COMMCHECK(queueWriteOnQp(qpIdx)); // Post next data immediately
      }
    }

    return commSuccess;
  }

  template <typename PerfConfig = DefaultPerfCollConfig>
  inline commResult_t readComplete(int qpNum, uint64_t wrId, int ibDevice) {
    int qpIdx = this->qpNumToIdx_.at(std::make_pair(qpNum, ibDevice));

    // We always process outstandingFastGets_ first.
    // Notice this gets restriction on iget() and igetFast(). All iget() must
    // complete before igetFast() can be called.
    if (qpIdx == 0 && outstandingFastGets_.size() > 0) {
      // fire the completion on the req
      auto& getInfo = outstandingFastGets_.front();
      if (getInfo.req != nullptr) {
        FB_COMMCHECK(getInfo.req->complete());
      }
      CLOGF_TRACE(
          COLL,
          "CTRAN-IB-VC: Complete fast path put for req {} opId {}",
          (void*)getInfo.req,
          getInfo.getId);

      outstandingFastGets_.pop_front();
      if (outstandingFastGets_.empty()) {
        FB_COMMCHECK(burnDownGets());
      }
      // Post next data if exists
      FB_COMMCHECK(queueReadOnQp(qpIdx));
    } else {
      auto& q = pendingWqeQs_.at(qpIdx);
      FB_CHECKABORT(q.size() > 0, "Unexpected empty pendingWqeQs");

      auto wqeInfo = dequeFront(q);
      auto& getInfo = *wqeInfo.getInfo;
      FB_CHECKABORT(
          wqeInfo.wrId == wrId, "wrId mismatch: {} != {}", wqeInfo.wrId, wrId);
      FB_CHECKABORT(
          getInfo.outstandingWqes > 0,
          "Got CQE for put with 0 outstanding WQEs");

      --getInfo.outstandingWqes;

      // Check if get is complete and, if so, update req status
      FB_COMMCHECK(burnDownGets());
      FB_COMMCHECK(queueReadOnQp(qpIdx)); // Post next data immediately
    }
    return commSuccess;
  }

  inline commResult_t atomicComplete() {
    if (pendingAtomicReqs_.size() > 0) {
      if (pendingAtomicReqs_.front() != nullptr) {
        pendingAtomicReqs_.front()->complete();
      }
      pendingAtomicReqs_.pop_front();
    } else {
      CLOGF(
          ERR,
          "CTRAN-IB-VC: Received completion for atomic Ops but no pending request");
      return commInternalError;
    }
    return commSuccess;
  }

  inline commResult_t burnDownPuts() {
    // Always burn down fast put first to ensure all fast put are completed.
    if (!outstandingFastPuts_.empty()) {
      return commSuccess;
    }
    while (outstandingPuts_.size() > 0) {
      auto& putInfo = *outstandingPuts_.front();
      if (putInfo.outstandingWqes > 0 || putInfo.offset < putInfo.len) {
        // If first request not yet complete, stop -- don't want to notify out
        // of order
        return commSuccess;
      }

      if (putInfo.vcMode == NCCL_CTRAN_IB_VC_MODE::spray && putInfo.notify) {
        // In spray mode, we aren't done yet; need to send a separate
        // notification signal to the far end
        FB_COMMCHECK(notify(putInfo.req, putInfo.putId));
      } else if (putInfo.req != nullptr) {
        // In DQPLB mode, we're done now, so we can fire the completion on the
        // req
        FB_COMMCHECK(putInfo.req->complete());
        CLOGF_TRACE(
            COLL,
            "CTRAN-IB-VC: Complete regular path put for req {} putId {}",
            (void*)putInfo.req,
            putInfo.putId);
      }
      outstandingPuts_.pop_front();
    }
    return commSuccess;
  }

  inline commResult_t burnDownGets() {
    // Always burn down fast get first to ensure all fast put are completed.
    if (!outstandingFastGets_.empty()) {
      return commSuccess;
    }
    while (outstandingGets_.size() > 0) {
      auto& getInfo = *outstandingGets_.front();
      if (getInfo.outstandingWqes > 0 || getInfo.offset < getInfo.len) {
        // If first request not yet complete, stop -- don't want to notify out
        // of order
        return commSuccess;
      }

      if (getInfo.req != nullptr) {
        // In DQPLB mode, we're done now, so we can fire the completion on the
        // req
        FB_COMMCHECK(getInfo.req->complete());
        CLOGF_TRACE(
            COLL,
            "CTRAN-IB-VC: Complete regular path get for req {} getId {}",
            (void*)getInfo.req,
            getInfo.getId);
      }
      outstandingGets_.pop_front();
    }
    return commSuccess;
  }

  std::optional<ibverbx::IbvQp> ibvControlQp_{std::nullopt};
  std::optional<ibverbx::IbvQp> ibvNotifyQp_{std::nullopt};
  std::optional<ibverbx::IbvQp> ibvAtomicQp_{std::nullopt};
  std::vector<ibverbx::IbvQp> ibvDataQps_;

  struct {
    std::vector<CtrlPacket> packets_;
    std::optional<ibverbx::IbvMr> ibvMr_;
    // Pointers referenenced to elements in packets_
    std::deque<std::reference_wrapper<CtrlPacket>> freePkts_;
    std::deque<std::reference_wrapper<CtrlPacket>> postedPkts_;
    std::deque<std::reference_wrapper<CtranIbRequest>> postedReqs_;
    std::deque<std::unique_ptr<ControlPendingSendWr>> enqueuedWrs_;
  } sendCtrl_;
  struct {
    std::vector<CtrlPacket> packets_;
    std::optional<ibverbx::IbvMr> ibvMr_;
    // Pointers referenenced to elements in packets_
    std::deque<std::reference_wrapper<CtrlPacket>> postedPkts_;
    std::deque<std::unique_ptr<ControlUnexpWr>> unexpWrs_;
    std::deque<std::unique_ptr<ControlPostedRecvWr>> enqueuedWrs_;
  } recvCtrl_;

  struct PutInfo {
    PutInfo(
        const void* sbuf,
        void* dbuf,
        size_t len,
        std::vector<uint32_t> lkeys,
        std::vector<uint32_t> rkeys,
        bool notify,
        CtranIbRequest* req,
        CtranIbConfig* config,
        enum NCCL_CTRAN_IB_VC_MODE vcMode,
        const uint64_t putId)
        : sbuf(sbuf),
          dbuf(dbuf),
          len(len),
          lkeys(std::move(lkeys)),
          rkeys(std::move(rkeys)),
          notify(notify),
          req(req),
          config(config),
          vcMode(vcMode),
          putId{putId} {}
    const void* sbuf;
    void* dbuf;
    size_t len;
    std::vector<uint32_t> lkeys;
    std::vector<uint32_t> rkeys;
    bool notify;
    CtranIbRequest* req;
    CtranIbConfig* config;
    enum NCCL_CTRAN_IB_VC_MODE vcMode;

    size_t offset{0};
    size_t outstandingWqes{0};
    uint64_t notifyWrId{0};
    const uint64_t putId{0};
  };

  struct GetInfo {
    GetInfo(
        const void* sbuf,
        void* dbuf,
        size_t len,
        std::vector<uint32_t> lkeys,
        std::vector<uint32_t> rkeys,
        CtranIbRequest* req,
        CtranIbConfig* config,
        enum NCCL_CTRAN_IB_VC_MODE vcMode,
        const uint64_t getId)
        : sbuf(sbuf),
          dbuf(dbuf),
          len(len),
          lkeys(std::move(lkeys)),
          rkeys(std::move(rkeys)),
          req(req),
          config(config),
          vcMode(vcMode),
          getId{getId} {}
    const void* sbuf;
    void* dbuf;
    size_t len;
    std::vector<uint32_t> lkeys;
    std::vector<uint32_t> rkeys;
    CtranIbRequest* req;
    CtranIbConfig* config;
    enum NCCL_CTRAN_IB_VC_MODE vcMode;

    size_t offset{0};
    size_t outstandingWqes{0};
    const uint64_t getId{0};
  };

  struct FastPutInfo {
    FastPutInfo(uint64_t putId, CtranIbRequest* req)
        : putId(putId), req(req) {};
    uint64_t putId;
    CtranIbRequest* req;
  };
  // Puts that have not yet been completely posted to wire
  std::deque<std::unique_ptr<PutInfo>> pendingPuts_;
  // Puts that have been completely sent, but not yet ACKed
  std::deque<std::unique_ptr<PutInfo>> outstandingPuts_;
  // Puts that have been sent though fast path, but not yet ACKed
  std::deque<FastPutInfo> outstandingFastPuts_;

  struct FastGetInfo {
    FastGetInfo(uint64_t getId, CtranIbRequest* req)
        : getId(getId), req(req) {};
    uint64_t getId;
    CtranIbRequest* req;
  };
  // Gets that have not yet been completely posted to wire
  std::deque<std::unique_ptr<GetInfo>> pendingGets_;
  // Gets that have been completely sent, but not yet ACKed
  std::deque<std::unique_ptr<GetInfo>> outstandingGets_;
  // Gets that have been sent though fast path, but not yet ACKed
  std::deque<FastGetInfo> outstandingFastGets_;

  std::deque<CtranIbRequest*> pendingAtomicReqs_;
  struct NotifyInfo {
    NotifyInfo(
        uint64_t wrId,
        CtranIbRequest* req,
        std::optional<uint64_t> putId)
        : wrId(wrId), req(req), putId(putId) {}
    uint64_t wrId;
    CtranIbRequest* req;
    // putId is optional for notify, since notify may be called standalone as an
    // exposed API at ctranIb
    std::optional<uint64_t> putId;
  };
  std::deque<std::unique_ptr<NotifyInfo>> pendingNotifies_;
  std::deque<std::unique_ptr<NotifyInfo>> outstandingNotifies_;

  struct WqeInfo {
    WqeInfo(uint64_t wrId, PutInfo* putInfo) : wrId(wrId), putInfo(putInfo) {}
    WqeInfo(uint64_t wrId, GetInfo* getInfo) : wrId(wrId), getInfo(getInfo) {}
    uint64_t wrId;
    PutInfo* putInfo;
    GetInfo* getInfo;
  };
  std::vector<std::deque<WqeInfo>> pendingWqeQs_;
  int qpIdxRR_{0};
  const int iputFastQpIdx_{0};
  const int igetFastQpIdx_{0};

  bool isReady_{false};
  std::vector<CtranIbDevice> devices_;
  uint8_t linkLayer_{0};
  int maxNumQps_{0};
  size_t qpScalingTh_{NCCL_CTRAN_IB_QP_SCALING_THRESHOLD};
  enum NCCL_CTRAN_IB_VC_MODE vcMode_ { NCCL_CTRAN_IB_VC_MODE::spray };
  int maxQpMsgs_;
  uint64_t mtu_{4096};
  uint64_t maxMsgSize_{CTRAN_IB_FAST_PATH_MSG_MAX_SIZE};

  std::unordered_map<QpUniqueId, int> qpNumToIdx_;

  uint32_t sndNxt_{0};
  uint32_t rcvNxt_{0};
  std::unordered_map<uint32_t, bool> rcvdSeqNums_;

  CtranComm* comm_{nullptr};
  CtranCtrlManager* ctrlMgr_{nullptr};

  // State for notifyQP notifications
  int32_t notifyCount_{0};
  uint64_t nextWrId_{65536};

  std::string connTypeName(ConnectionType typ);
  std::string vcModeName(enum NCCL_CTRAN_IB_VC_MODE mode);
  void logConnectionConfig(ConnectionType typ);
  uint32_t pgTrafficClass_;
  int cudaDev_;
  static constexpr int kNotifyBit = 31;
  static constexpr int kFastPutBit = 30;
  static constexpr int kNotifyQpIdx = -1;
  static constexpr uint32_t kSeqNumMask = 0xFFFFFF; // 24 bits
  ibverbx::ibv_sge sendCtrlSg_{}, recvCtrlSg_{}, sendPutSg_{}, sendGetSg_{},
      fetchAddSg_{}, amoSetSg_{};
  ibverbx::ibv_send_wr sendCtrlWr_{}, badSendCtrlWr_{}, sendNotifyWr_{},
      badSendNotifyWr_{}, sendPutWr_{}, badSendPutWr_{}, sendGetWr_{},
      badSendGetWr_{}, fetchAddWr_{}, badFetchAddWr_{}, amoSetWr_{},
      badAmoSetWr_{};
  ibverbx::ibv_recv_wr recvCtrlWr_{}, badRecvCtrlWr_{}, recvNotifyWr_{},
      badRecvNotifyWr_{};

  uint64_t putId_{0}; // id of put in a VC; increase by 1 for each put
  uint64_t getId_{0}; // id of put in a VC; increase by 1 for each put
  uint64_t getPutId() {
    return putId_++;
  }
  uint64_t getGetId() {
    return getId_++;
  }
};

#endif
