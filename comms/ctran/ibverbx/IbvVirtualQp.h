// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <folly/Expected.h>
#include <folly/dynamic.h>
#include <deque>
#include <optional>
#include <utility>

#include "comms/ctran/ibverbx/DqplbSeqTracker.h"
#include "comms/ctran/ibverbx/IbvCommon.h"
#include "comms/ctran/ibverbx/IbvQp.h"
#include "comms/ctran/ibverbx/IbvVirtualWr.h"
#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

class IbvVirtualCq;

// IbvVirtualQpBusinessCard
struct IbvVirtualQpBusinessCard {
  explicit IbvVirtualQpBusinessCard(
      std::vector<uint32_t> qpNums,
      uint32_t notifyQpNum = 0);
  IbvVirtualQpBusinessCard() = default;
  ~IbvVirtualQpBusinessCard() = default;

  // Default copy constructor and assignment operator
  IbvVirtualQpBusinessCard(const IbvVirtualQpBusinessCard& other) = default;
  IbvVirtualQpBusinessCard& operator=(const IbvVirtualQpBusinessCard& other) =
      default;

  // Default move constructor and assignment operator
  IbvVirtualQpBusinessCard(IbvVirtualQpBusinessCard&& other) = default;
  IbvVirtualQpBusinessCard& operator=(IbvVirtualQpBusinessCard&& other) =
      default;

  // Convert to/from folly::dynamic for serialization
  folly::dynamic toDynamic() const;
  static folly::Expected<IbvVirtualQpBusinessCard, Error> fromDynamic(
      const folly::dynamic& obj);

  // JSON serialization methods
  std::string serialize() const;
  static folly::Expected<IbvVirtualQpBusinessCard, Error> deserialize(
      const std::string& jsonStr);

  // The qpNums_ vector is ordered: the ith QP in qpNums_ will be
  // connected to the ith QP in the remote side's qpNums_ vector.
  std::vector<uint32_t> qpNums_;
  uint32_t notifyQpNum_{0};
};

// IbvVirtualQp is the user-facing interface for posting RDMA work requests.
// It abstracts multiple physical QPs behind a single virtual QP interface.
// When load balancing is enabled, large RDMA messages are fragmented into
// smaller chunks and distributed across the underlying physical QPs using a
// configurable load balancing scheme (SPRAY or DQPLB). Completions from all
// physical QPs are aggregated by a paired IbvVirtualCq into a single virtual
// completion per original message. When load balancing is not needed, it falls
// back to single-QP operations.
//
// Currently, each IbvVirtualQp is associated with exactly one IbvVirtualCq
// that serves as both the send and receive completion queue.
class IbvVirtualQp {
 public:
  IbvVirtualQp(
      std::vector<IbvQp>&& qps,
      IbvVirtualCq* virtualCq,
      int maxMsgCntPerQp = kIbMaxMsgCntPerQp,
      int maxMsgSize = kIbMaxMsgSizeByte,
      LoadBalancingScheme loadBalancingScheme = LoadBalancingScheme::SPRAY,
      std::optional<IbvQp>&& notifyQp = std::nullopt);
  ~IbvVirtualQp();

  // disable copy constructor
  IbvVirtualQp(const IbvVirtualQp&) = delete;
  IbvVirtualQp& operator=(const IbvVirtualQp&) = delete;

  // move constructor
  IbvVirtualQp(IbvVirtualQp&& other) noexcept;
  IbvVirtualQp& operator=(IbvVirtualQp&& other) noexcept;

  size_t getTotalQps() const;
  const std::vector<IbvQp>& getQpsRef() const;
  std::vector<IbvQp>& getQpsRef();
  const IbvQp& getNotifyQpRef() const;
  IbvQp& getNotifyQpRef();
  bool hasNotifyQp() const {
    CHECK(physicalQps_.size() == 1 || notifyQp_.has_value())
        << "notifyQp must be provided when using multiple data QPs!";
    return notifyQp_.has_value();
  }
  bool isMultiQp() const {
    return isMultiQp_;
  }
  uint32_t getVirtualQpNum() const;
  // If businessCard is not provided, all physical QPs will be updated with the
  // universal attributes specified in attr. This is typically used for changing
  // the state to INIT or RTS.
  // If businessCard is provided, attr.qp_num for each physical QP will be set
  // individually to the corresponding qpNum stored in qpNums_ within
  // businessCard. This is typically used for changing the state to RTR.
  folly::Expected<folly::Unit, Error> modifyVirtualQp(
      ibv_qp_attr* attr,
      int attrMask,
      const IbvVirtualQpBusinessCard& businessCard =
          IbvVirtualQpBusinessCard());
  IbvVirtualQpBusinessCard getVirtualQpBusinessCard() const;
  LoadBalancingScheme getLoadBalancingScheme() const;

  // post send: routes by opcode — single-QP passthrough for SEND/atomic
  // ops, multi-QP load-balanced fragmentation for RDMA ops.
  inline folly::Expected<folly::Unit, Error> postSend(
      const IbvVirtualSendWr& wr);

  // post recv: routes by opcode - single-QP passthrough for data recvs, tracked
  // notification recv for multi-QP SPRAY/DQPLB modes.
  inline folly::Expected<folly::Unit, Error> postRecv(
      const IbvVirtualRecvWr& wr);

  inline int findAvailableSendQp();

  // Completion processing: Route physical CQEs to virtual WR state.
  inline folly::Expected<std::vector<IbvVirtualWc>, Error> processCompletion(
      const ibv_wc& physicalWc,
      int32_t deviceId);

  // Completion processing: batch version
  inline folly::Expected<std::vector<IbvVirtualWc>, Error> processCompletions(
      const ibv_wc* physicalWcs,
      int count,
      int32_t deviceId = 0);

 private:
#ifdef IBVERBX_TEST_FRIENDS
  IBVERBX_TEST_FRIENDS
#endif

  friend class IbvPd;
  friend class IbvVirtualCq;

  // Pointer to the VirtualCq this VirtualQp is registered with
  IbvVirtualCq* virtualCq_{nullptr};

  // Flag indicating if this VirtualQp uses multiple physical QPs
  bool isMultiQp_{false};

  // WR Tracking: Separate trackers for send and recv
  WrTracker<ActiveVirtualWr> sendTracker_;
  WrTracker<ActiveVirtualWr> recvTracker_;

  // SPRAY notify tracking
  std::deque<uint64_t> pendingSendNotifyQue_;
  std::deque<uint64_t> pendingRecvNotifyQue_;

  inline static std::atomic<uint32_t> nextVirtualQpNum_{
      0}; // Static counter for assigning unique virtual QP numbers
  uint32_t virtualQpNum_{0}; // The unique virtual QP number assigned to
                             // instance of IbvVirtualQp.

  std::vector<IbvQp> physicalQps_;

  // Maps (deviceId, qpNum) -> index into physicalQps_.
  // Uses QpId to handle multi-NIC setups where different devices can assign
  // the same QP number.
  std::unordered_map<QpId, int, QpIdHash> qpNumToIdx_;

  int nextSendPhysicalQpIdx_{0};
  int nextRecvPhysicalQpIdx_{0};

  int maxMsgCntPerQp_{
      -1}; // Maximum number of messages that can be sent on each physical QP. A
           // value of -1 indicates there is no limit.
  int maxMsgSize_{0};

  uint64_t nextPhysicalWrId_{0}; // ID of the next physical work request to
                                 // be posted on the physical QP

  int deviceCnt_{0}; // Number of unique devices that the physical QPs span

  LoadBalancingScheme loadBalancingScheme_{
      LoadBalancingScheme::SPRAY}; // Load balancing scheme for this virtual QP

  // Spray mode specific fields
  std::optional<IbvQp> notifyQp_;

  // DQPLB mode specific fields and functions
  DqplbSeqTracker dqplbSeqTracker;
  bool dqplbReceiverInitialized_{
      false}; // flag to indicate if dqplb receiver is initialized
  inline folly::Expected<folly::Unit, Error> initializeDqplbReceiver();

  // Send helper functions
  // Posts a send WR directly to physical QP 0 without fragmentation or
  // load balancing.
  inline folly::Expected<folly::Unit, Error> postSendSingleQp(
      const IbvVirtualSendWr& wr);
  // Iterates pending sends in the tracker, fragments them into maxMsgSize_
  // chunks, and posts each fragment to an available physical QP.
  // If freedQpIdx is provided, that QP is tried first.
  inline folly::Expected<folly::Unit, Error> dispatchPendingSends(
      int freedQpIdx = -1);
  // Builds a physical ibv_send_wr and ibv_sge from an ActiveVirtualWr
  // fragment, applying the correct opcode and device-specific keys.
  // Caller must set sendWr.sg_list = &sendSge after destructuring.
  inline std::pair<ibv_send_wr, ibv_sge> buildPhysicalSendWr(
      const ActiveVirtualWr& pending,
      int32_t deviceId,
      uint32_t fragLen);
  // Returns true if the physical QP at qpIdx has room for more outstanding
  // send WRs (respects maxMsgCntPerQp_ limit).
  inline bool hasQpCapacity(int qpIdx) const;
  // Drains completed send WRs in order, emitting virtual completions into
  // results. For SPRAY mode, triggers notify posts when all data fragments
  // are done.
  inline folly::Expected<folly::Unit, Error> reportSendCompletions(
      std::vector<IbvVirtualWc>& results);
  // Posts a zero-byte RDMA_WRITE_WITH_IMM on the notify QP to signal the
  // receiver that all data fragments for this WR have been sent.
  inline folly::Expected<folly::Unit, Error> postSendToNotifyQp(
      uint64_t internalWrId);
  // Drains the pendingSendNotifyQue_ by posting queued notify messages
  // whenever the notify QP has capacity.
  inline folly::Expected<folly::Unit, Error> flushPendingSendNotifies();

  // Recv helper functions
  // Posts a recv WR directly to physical QP 0 without tracking.
  inline folly::Expected<folly::Unit, Error> postRecvSingleQp(
      const IbvVirtualRecvWr& wr);
  // Posts a zero-length recv on the notify QP for SPRAY mode notification.
  inline folly::Expected<folly::Unit, Error> postRecvToNotifyQp(
      uint64_t internalWrId);
  // Drains the pendingRecvNotifyQue_ by posting queued recv notifications
  // whenever the notify QP has capacity.
  inline folly::Expected<folly::Unit, Error> flushPendingRecvNotifies();
  // Re-posts a zero-length recv on the specified data QP after consuming
  // a DQPLB completion, keeping the recv pool at steady state.
  inline folly::Expected<folly::Unit, Error> replenishDqplbRecv(int qpIdx);
  // Drains completed recv WRs in order, emitting virtual completions
  // into results.
  inline folly::Expected<folly::Unit, Error> reportRecvCompletions(
      std::vector<IbvVirtualWc>& results);

  // Completion processing handlers (2x2 matrix: QP type x direction).
  // Each handler pops the physical WR status, updates the virtual WR state,
  // and reports any completed virtual WRs.
  //
  // Handles notify QP send completion — SPRAY sender's notify is done.
  inline folly::Expected<std::vector<IbvVirtualWc>, Error>
  processNotifyQpSendCompletion(
      const ibv_wc& physicalWc,
      std::vector<IbvVirtualWc>& results);
  // Handles notify QP recv completion — SPRAY receiver's notify arrived.
  inline folly::Expected<std::vector<IbvVirtualWc>, Error>
  processNotifyQpRecvCompletion(
      const ibv_wc& physicalWc,
      std::vector<IbvVirtualWc>& results);
  // Handles data QP send completion — a data fragment send finished,
  // dispatches more pending fragments if available.
  inline folly::Expected<std::vector<IbvVirtualWc>, Error>
  processDataQpSendCompletion(
      const ibv_wc& physicalWc,
      int qpIdx,
      std::vector<IbvVirtualWc>& results);
  // Handles data QP recv completion — processes DQPLB sequence numbers
  // and replenishes the recv pool.
  inline folly::Expected<std::vector<IbvVirtualWc>, Error>
  processDataQpRecvCompletion(
      const ibv_wc& physicalWc,
      int qpIdx,
      std::vector<IbvVirtualWc>& results);

  // Pops and validates the front entry of a physical WR status queue,
  // returning the associated internal (virtual) WR ID.
  template <typename PhysicalWrStatusT>
  inline folly::Expected<uint64_t, Error> popPhysicalQueueStatus(
      std::deque<PhysicalWrStatusT>& queStatus,
      uint64_t expectedPhysicalWrId,
      const char* queueName);

  // Common helpers
  inline bool isSendOpcode(ibv_wc_opcode opcode) const;
  inline folly::Expected<folly::Unit, Error> updateWrState(
      WrTracker<ActiveVirtualWr>& tracker,
      uint64_t internalWrId,
      ibv_wc_status status,
      ibv_wc_opcode wcOpcode);
  inline IbvVirtualWc buildVirtualWc(const ActiveVirtualWr& wr) const;

  // VirtualCq registration helpers (called from constructor/destructor)
  void registerWithVirtualCq();
  void unregisterFromVirtualCq();
};

// IbvVirtualQp inline functions

inline int IbvVirtualQp::findAvailableSendQp() {
  // maxMsgCntPerQp_ with a value of -1 indicates there is no limit.
  if (maxMsgCntPerQp_ == -1) {
    auto availableQpIdx = nextSendPhysicalQpIdx_;
    nextSendPhysicalQpIdx_ = (nextSendPhysicalQpIdx_ + 1) % physicalQps_.size();
    return availableQpIdx;
  }

  for (int i = 0; i < physicalQps_.size(); i++) {
    if (physicalQps_.at(nextSendPhysicalQpIdx_).physicalSendWrStatus_.size() <
        maxMsgCntPerQp_) {
      auto availableQpIdx = nextSendPhysicalQpIdx_;
      nextSendPhysicalQpIdx_ =
          (nextSendPhysicalQpIdx_ + 1) % physicalQps_.size();
      return availableQpIdx;
    }
    nextSendPhysicalQpIdx_ = (nextSendPhysicalQpIdx_ + 1) % physicalQps_.size();
  }
  return -1;
}

inline folly::Expected<folly::Unit, Error>
IbvVirtualQp::initializeDqplbReceiver() {
  ibv_recv_wr recvWr_{};
  ibv_recv_wr badRecvWr_{};
  ibv_sge recvSg_{};
  recvWr_.next = nullptr;
  recvWr_.sg_list = &recvSg_;
  recvWr_.num_sge = 0;
  for (int i = 0; i < maxMsgCntPerQp_; i++) {
    for (int j = 0; j < physicalQps_.size(); j++) {
      recvWr_.wr_id = nextPhysicalWrId_++;
      auto maybeRecv = physicalQps_.at(j).postRecv(&recvWr_, &badRecvWr_);
      if (maybeRecv.hasError()) {
        return folly::makeUnexpected(maybeRecv.error());
      }
      physicalQps_.at(j).physicalRecvWrStatus_.emplace_back(recvWr_.wr_id, -1);
    }
  }

  dqplbReceiverInitialized_ = true;
  return folly::unit;
}

// ============================================================
// Send Path
// ============================================================

// Helper: Single-QP fast path (pure passthrough, no tracking)
inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postSendSingleQp(
    const IbvVirtualSendWr& wr) {
  if (wr.length == 0) {
    return folly::makeUnexpected(Error(
        EINVAL,
        "[Ibverbx]IbvVirtualQp::postSendSingleQp, length cannot be zero"));
  }

  ibv_send_wr sendWr{};
  ibv_sge sendSge{};

  sendSge.addr = reinterpret_cast<uint64_t>(wr.localAddr);
  sendSge.length = wr.length;
  int32_t deviceId = physicalQps_.at(0).getDeviceId();
  sendSge.lkey = wr.deviceKeys.at(deviceId).lkey;

  sendWr.wr_id = wr.wrId;
  sendWr.sg_list = &sendSge;
  sendWr.num_sge = 1;
  sendWr.opcode = wr.opcode;
  sendWr.send_flags = wr.sendFlags;

  if (wr.opcode == IBV_WR_ATOMIC_FETCH_AND_ADD ||
      wr.opcode == IBV_WR_ATOMIC_CMP_AND_SWP) {
    // Atomic operations use wr.atomic union
    sendWr.wr.atomic.remote_addr = wr.remoteAddr;
    sendWr.wr.atomic.rkey = wr.deviceKeys.at(deviceId).rkey;
    sendWr.wr.atomic.compare_add = wr.compareAdd;
    sendWr.wr.atomic.swap = wr.swap;
  } else {
    // RDMA / SEND operations use wr.rdma union
    sendWr.wr.rdma.remote_addr = wr.remoteAddr;
    sendWr.wr.rdma.rkey = wr.deviceKeys.at(deviceId).rkey;
    if (wr.opcode == IBV_WR_RDMA_WRITE_WITH_IMM) {
      sendWr.imm_data = wr.immData;
    }
  }

  ibv_send_wr badWr{};
  auto maybePost = physicalQps_.at(0).postSend(&sendWr, &badWr);
  if (maybePost.hasError()) {
    return folly::makeUnexpected(maybePost.error());
  }

  return folly::unit;
}

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postSend(
    const IbvVirtualSendWr& wr) {
  // Opcode routing: route by opcode first, then check isMultiQp_ for RDMA ops
  switch (wr.opcode) {
    // Always single-QP pass-through (no load balancing)
    case IBV_WR_SEND:
    case IBV_WR_ATOMIC_FETCH_AND_ADD:
    case IBV_WR_ATOMIC_CMP_AND_SWP:
      return postSendSingleQp(wr);

    // RDMA ops: single-QP pass-through or multi-QP load balancing
    case IBV_WR_RDMA_WRITE:
    case IBV_WR_RDMA_WRITE_WITH_IMM:
    case IBV_WR_RDMA_READ:
      if (!isMultiQp_) {
        return postSendSingleQp(wr);
      }
      break; // Fall through to multi-QP load balancing path

    default:
      return folly::makeUnexpected(Error(
          EINVAL,
          fmt::format(
              "[Ibverbx]IbvVirtualQp::postSend, unsupported opcode: {}",
              static_cast<int>(wr.opcode))));
  }

  // ============================================================
  // MULTI-QP RDMA LOAD BALANCING PATH (only RDMA ops reach here)
  // ============================================================

  // Parameter validation
  if (wr.length == 0) {
    return folly::makeUnexpected(Error(
        EINVAL, "[Ibverbx]IbvVirtualQp::postSend, RDMA length cannot be zero"));
  }

  if (!(wr.sendFlags & IBV_SEND_SIGNALED) &&
      wr.opcode != IBV_WR_RDMA_WRITE_WITH_IMM) {
    return folly::makeUnexpected(Error(
        EINVAL,
        "[Ibverbx]IbvVirtualQp::postSend, unsignaled operations not supported in multi-QP mode"));
  }

  bool needsNotify =
      (wr.opcode == IBV_WR_RDMA_WRITE_WITH_IMM &&
       loadBalancingScheme_ == LoadBalancingScheme::SPRAY);
  int expectedMsgCnt = (wr.length + maxMsgSize_ - 1) / maxMsgSize_;

  sendTracker_.add(
      ActiveVirtualWr{
          .userWrId = wr.wrId,
          .remainingMsgCnt = needsNotify ? expectedMsgCnt + 1 : expectedMsgCnt,
          .aggregatedStatus = IBV_WC_SUCCESS,
          .localAddr = wr.localAddr,
          .length = wr.length,
          .remoteAddr = wr.remoteAddr,
          .opcode = wr.opcode,
          .immData = wr.immData,
          .deviceKeys = wr.deviceKeys,
          .offset = 0,
          .needsNotify = needsNotify,
          .notifyPosted = false});

  auto result = dispatchPendingSends();
  if (result.hasError()) {
    return folly::makeUnexpected(result.error());
  }

  return folly::unit;
}

// ============================================================
// Recv Path
// ============================================================

// Helper: Single-QP recv fast path (pure passthrough, no tracking)
inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postRecvSingleQp(
    const IbvVirtualRecvWr& wr) {
  ibv_recv_wr recvWr{};
  ibv_sge sge{};

  recvWr.wr_id = wr.wrId;
  recvWr.next = nullptr;

  if (wr.length > 0) {
    sge.addr = reinterpret_cast<uint64_t>(wr.localAddr);
    sge.length = wr.length;
    sge.lkey = wr.deviceKeys.at(physicalQps_[0].getDeviceId()).lkey;
    recvWr.sg_list = &sge;
    recvWr.num_sge = 1;
  } else {
    recvWr.sg_list = nullptr;
    recvWr.num_sge = 0;
  }

  ibv_recv_wr badWr{};
  auto maybePost = physicalQps_[0].postRecv(&recvWr, &badWr);
  if (maybePost.hasError()) {
    return folly::makeUnexpected(maybePost.error());
  }

  return folly::unit;
}

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postRecv(
    const IbvVirtualRecvWr& wr) {
  // Fast path: Single physical QP (pure passthrough)
  if (!isMultiQp_) {
    return postRecvSingleQp(wr);
  }

  // ============================================================
  // MULTI-QP PATH
  // ============================================================

  if (wr.length > 0) {
    return postRecvSingleQp(wr);
  }

  // ============================================================
  // ZERO-LENGTH RECV PATH (SPRAY/DQPLB notification)
  // ============================================================

  uint64_t internalId = recvTracker_.add(
      ActiveVirtualWr{
          .userWrId = wr.wrId,
          .remainingMsgCnt = 1,
          .aggregatedStatus = IBV_WC_SUCCESS,
          .localAddr = nullptr,
          .length = 0,
          .remoteAddr = 0,
          .opcode = IBV_WR_RDMA_WRITE_WITH_IMM,
          .immData = 0,
          .deviceKeys = {},
          .offset = 0,
          .needsNotify = false,
          .notifyPosted = false});

  // DQPLB mode: Initialize receiver with pre-posted recvs on first call
  if (loadBalancingScheme_ == LoadBalancingScheme::DQPLB) {
    if (!dqplbReceiverInitialized_) {
      if (initializeDqplbReceiver().hasError()) {
        return folly::makeUnexpected(Error(
            errno,
            "[Ibverbx]IbvVirtualQp::postRecv, DQPLB receiver initialization failed"));
      }
    }
    return folly::unit;
  }

  // SPRAY mode: Post zero-length recv to notifyQp
  CHECK(hasNotifyQp());

  if (notifyQp_->physicalRecvWrStatus_.size() >=
      static_cast<size_t>(maxMsgCntPerQp_)) {
    pendingRecvNotifyQue_.push_back(internalId);
    return folly::unit;
  }

  if (postRecvToNotifyQp(internalId).hasError()) {
    return folly::makeUnexpected(Error(
        errno,
        "[Ibverbx]IbvVirtualQp::postRecv, failed to post recv to notifyQp"));
  }

  return folly::unit;
}

// Replenish a single DQPLB recv on the specified QP after consuming one
inline folly::Expected<folly::Unit, Error> IbvVirtualQp::replenishDqplbRecv(
    int qpIdx) {
  CHECK(qpIdx >= 0 && qpIdx < static_cast<int>(physicalQps_.size()))
      << fmt::format(
             "[Ibverbx]IbvVirtualQp::replenishDqplbRecv, invalid qpIdx={}",
             qpIdx);

  ibv_recv_wr recvWr{};
  ibv_recv_wr badWr{};

  recvWr.wr_id = nextPhysicalWrId_++;
  recvWr.next = nullptr;
  recvWr.sg_list = nullptr;
  recvWr.num_sge = 0;

  auto maybeRecv = physicalQps_[qpIdx].postRecv(&recvWr, &badWr);
  if (maybeRecv.hasError()) {
    return folly::makeUnexpected(maybeRecv.error());
  }

  physicalQps_[qpIdx].physicalRecvWrStatus_.emplace_back(recvWr.wr_id, -1);

  return folly::unit;
}

// Post zero-length recv to notifyQp (SPRAY mode)
inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postRecvToNotifyQp(
    uint64_t internalWrId) {
  CHECK(hasNotifyQp());

  ibv_recv_wr recvWr{};
  ibv_recv_wr badWr{};

  recvWr.wr_id = nextPhysicalWrId_++;
  recvWr.next = nullptr;
  recvWr.sg_list = nullptr;
  recvWr.num_sge = 0;

  auto maybeRecv = notifyQp_->postRecv(&recvWr, &badWr);
  if (maybeRecv.hasError()) {
    return folly::makeUnexpected(maybeRecv.error());
  }

  notifyQp_->physicalRecvWrStatus_.emplace_back(recvWr.wr_id, internalWrId);

  return folly::unit;
}

// Flush pending recv notifications when notifyQp backpressure clears
inline folly::Expected<folly::Unit, Error>
IbvVirtualQp::flushPendingRecvNotifies() {
  CHECK(hasNotifyQp());

  while (!pendingRecvNotifyQue_.empty()) {
    if (notifyQp_->physicalRecvWrStatus_.size() >=
        static_cast<size_t>(maxMsgCntPerQp_)) {
      break;
    }

    uint64_t frontId = pendingRecvNotifyQue_.front();

    if (postRecvToNotifyQp(frontId).hasError()) {
      break;
    }

    pendingRecvNotifyQue_.pop_front();
  }

  return folly::unit;
}

// ============================================================
// Fragmentation Logic (dispatchPendingSends)
// ============================================================

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::dispatchPendingSends(
    int freedQpIdx) {
  while (sendTracker_.hasPendingPost()) {
    uint64_t internalId = sendTracker_.frontPendingPost();

    auto* pending = sendTracker_.find(internalId);
    CHECK(pending) << fmt::format(
        "[Ibverbx]IbvVirtualQp::dispatchPendingSends, WR {} in pendingPostQue_ but not found in activeVirtualWrs_",
        internalId);

    while (pending->offset < pending->length) {
      int qpIdx;
      if (freedQpIdx >= 0 && hasQpCapacity(freedQpIdx)) {
        qpIdx = freedQpIdx;
        freedQpIdx = -1;
      } else {
        qpIdx = findAvailableSendQp();
      }

      if (qpIdx == -1) {
        return folly::unit;
      }

      int32_t deviceId = physicalQps_.at(qpIdx).getDeviceId();

      uint32_t fragLen = std::min(
          maxMsgSize_, static_cast<int>(pending->length - pending->offset));

      auto [sendWr, sendSge] = buildPhysicalSendWr(*pending, deviceId, fragLen);
      sendWr.sg_list = &sendSge;

      ibv_send_wr badWr{};
      auto maybePost = physicalQps_.at(qpIdx).postSend(&sendWr, &badWr);
      if (maybePost.hasError()) {
        return folly::makeUnexpected(maybePost.error());
      }

      physicalQps_.at(qpIdx).physicalSendWrStatus_.emplace_back(
          sendWr.wr_id, internalId);

      pending->offset += fragLen;
    }

    sendTracker_.popPendingPost();
  }

  return folly::unit;
}

inline std::pair<ibv_send_wr, ibv_sge> IbvVirtualQp::buildPhysicalSendWr(
    const ActiveVirtualWr& pending,
    int32_t deviceId,
    uint32_t fragLen) {
  ibv_sge sendSge{};
  sendSge.addr = reinterpret_cast<uint64_t>(
      static_cast<char*>(pending.localAddr) + pending.offset);
  sendSge.length = fragLen;
  sendSge.lkey = pending.deviceKeys.at(deviceId).lkey;

  ibv_send_wr sendWr{};
  sendWr.wr_id = nextPhysicalWrId_++;
  sendWr.sg_list = &sendSge;
  sendWr.num_sge = 1;
  sendWr.send_flags = IBV_SEND_SIGNALED;

  sendWr.wr.rdma.remote_addr = pending.remoteAddr + pending.offset;
  sendWr.wr.rdma.rkey = pending.deviceKeys.at(deviceId).rkey;

  if (pending.opcode == IBV_WR_RDMA_WRITE_WITH_IMM &&
      loadBalancingScheme_ == LoadBalancingScheme::SPRAY) {
    sendWr.opcode = IBV_WR_RDMA_WRITE;
  } else {
    sendWr.opcode = pending.opcode;
    if (loadBalancingScheme_ == LoadBalancingScheme::DQPLB) {
      bool isLastFragment = (pending.offset + fragLen >= pending.length);
      sendWr.imm_data = dqplbSeqTracker.getSendImm(isLastFragment);
    }
  }

  return {sendWr, sendSge};
}

inline bool IbvVirtualQp::hasQpCapacity(int qpIdx) const {
  if (maxMsgCntPerQp_ == -1) {
    return true;
  }
  return physicalQps_.at(qpIdx).physicalSendWrStatus_.size() <
      static_cast<size_t>(maxMsgCntPerQp_);
}

// ============================================================
// Send Completion Reporting
// ============================================================

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::reportSendCompletions(
    std::vector<IbvVirtualWc>& results) {
  while (sendTracker_.hasPendingCompletion()) {
    uint64_t frontId = sendTracker_.frontPendingCompletion();
    auto* frontWr = sendTracker_.find(frontId);
    CHECK(frontWr) << fmt::format(
        "[Ibverbx]IbvVirtualQp::reportSendCompletions, WR {} in pendingCompletionQue_ but not found in activeVirtualWrs_",
        frontId);

    if (frontWr->needsNotify && !frontWr->notifyPosted) {
      if (frontWr->remainingMsgCnt > 1) {
        break;
      }

      CHECK(hasNotifyQp());
      if (notifyQp_->physicalSendWrStatus_.size() >=
          static_cast<size_t>(maxMsgCntPerQp_)) {
        pendingSendNotifyQue_.push_back(frontId);
        frontWr->notifyPosted = true;
        break;
      }

      auto maybePost = postSendToNotifyQp(frontId);
      if (maybePost.hasError()) {
        return folly::makeUnexpected(maybePost.error());
      }
      frontWr->notifyPosted = true;
      break;
    }

    if (frontWr->remainingMsgCnt > 0) {
      break;
    }

    results.push_back(buildVirtualWc(*frontWr));
    sendTracker_.remove(frontId);
    sendTracker_.popPendingCompletion();
  }

  return folly::unit;
}

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postSendToNotifyQp(
    uint64_t internalWrId) {
  auto* pending = sendTracker_.find(internalWrId);
  CHECK(pending) << fmt::format(
      "[Ibverbx]IbvVirtualQp::postSendToNotifyQp, WR {} not found",
      internalWrId);

  CHECK(hasNotifyQp());

  ibv_send_wr sendWr{};
  sendWr.wr_id = nextPhysicalWrId_++;
  sendWr.sg_list = nullptr;
  sendWr.num_sge = 0;
  sendWr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  sendWr.send_flags = IBV_SEND_SIGNALED;
  sendWr.wr.rdma.remote_addr = pending->remoteAddr;
  int32_t notifyDeviceId = notifyQp_->getDeviceId();
  sendWr.wr.rdma.rkey = pending->deviceKeys.at(notifyDeviceId).rkey;
  sendWr.imm_data = pending->immData;

  ibv_send_wr badWr{};
  auto maybePost = notifyQp_->postSend(&sendWr, &badWr);
  if (maybePost.hasError()) {
    return folly::makeUnexpected(maybePost.error());
  }

  notifyQp_->physicalSendWrStatus_.emplace_back(sendWr.wr_id, internalWrId);

  return folly::unit;
}

inline folly::Expected<folly::Unit, Error>
IbvVirtualQp::flushPendingSendNotifies() {
  CHECK(hasNotifyQp());

  while (!pendingSendNotifyQue_.empty()) {
    if (notifyQp_->physicalSendWrStatus_.size() >=
        static_cast<size_t>(maxMsgCntPerQp_)) {
      break;
    }

    uint64_t frontId = pendingSendNotifyQue_.front();
    auto* frontWr = sendTracker_.find(frontId);
    CHECK(frontWr) << fmt::format(
        "[Ibverbx]IbvVirtualQp::flushPendingSendNotifies, WR {} in pendingSendNotifyQue_ but not found in activeVirtualWrs_",
        frontId);

    if (postSendToNotifyQp(frontId).hasError()) {
      break;
    }
    pendingSendNotifyQue_.pop_front();
  }

  return folly::unit;
}

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::reportRecvCompletions(
    std::vector<IbvVirtualWc>& results) {
  while (recvTracker_.hasPendingCompletion()) {
    uint64_t frontId = recvTracker_.frontPendingCompletion();
    auto* frontWr = recvTracker_.find(frontId);
    CHECK(frontWr) << fmt::format(
        "[Ibverbx]IbvVirtualQp::reportRecvCompletions, WR {} in pendingCompletionQue_ but not found in activeVirtualWrs_",
        frontId);

    if (frontWr->remainingMsgCnt > 0) {
      break;
    }

    results.push_back(buildVirtualWc(*frontWr));
    recvTracker_.remove(frontId);
    recvTracker_.popPendingCompletion();
  }

  return folly::unit;
}

// ============================================================
// Common Helpers
// ============================================================

inline bool IbvVirtualQp::isSendOpcode(ibv_wc_opcode opcode) const {
  return opcode == IBV_WC_SEND || opcode == IBV_WC_RDMA_WRITE ||
      opcode == IBV_WC_RDMA_READ || opcode == IBV_WC_FETCH_ADD ||
      opcode == IBV_WC_COMP_SWAP;
}

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::updateWrState(
    WrTracker<ActiveVirtualWr>& tracker,
    uint64_t internalWrId,
    ibv_wc_status status,
    ibv_wc_opcode wcOpcode) {
  auto* wr = tracker.find(internalWrId);
  CHECK(wr) << fmt::format(
      "[Ibverbx] WR {} not found in tracker during updateWrState",
      internalWrId);

  wr->remainingMsgCnt--;
  wr->wcOpcode = wcOpcode;

  // First error wins
  if (wr->aggregatedStatus == IBV_WC_SUCCESS && status != IBV_WC_SUCCESS) {
    wr->aggregatedStatus = status;
    XLOGF(
        ERR,
        "[Ibverbx] Physical WC error: status={}, WR internalId={}",
        static_cast<int>(status),
        internalWrId);
  }

  return folly::unit;
}

inline IbvVirtualWc IbvVirtualQp::buildVirtualWc(
    const ActiveVirtualWr& wr) const {
  IbvVirtualWc wc;
  wc.wrId = wr.userWrId;
  wc.status = wr.aggregatedStatus;
  wc.byteLen = wr.length;
  wc.qpNum = virtualQpNum_;
  wc.immData = wr.immData;
  wc.opcode = wr.wcOpcode;
  return wc;
}

// ============================================================
// Physical Queue Status Helper (templated)
// ============================================================

template <typename PhysicalWrStatusT>
inline folly::Expected<uint64_t, Error> IbvVirtualQp::popPhysicalQueueStatus(
    std::deque<PhysicalWrStatusT>& queStatus,
    uint64_t expectedPhysicalWrId,
    const char* queueName) {
  CHECK(!queStatus.empty()) << fmt::format(
      "[Ibverbx]IbvVirtualQp::popPhysicalQueueStatus, no pending WR in {}",
      queueName);

  auto& frontStatus = queStatus.front();
  CHECK_EQ(frontStatus.physicalWrId, expectedPhysicalWrId) << fmt::format(
      "[Ibverbx]IbvVirtualQp::popPhysicalQueueStatus, {} WR ID mismatch: expected {}, got {}",
      queueName,
      frontStatus.physicalWrId,
      expectedPhysicalWrId);

  uint64_t internalWrId = frontStatus.virtualWrId;
  queStatus.pop_front();
  return internalWrId;
}

// ============================================================
// Completion Processing (processCompletion + 2x2 dispatch)
// ============================================================

inline folly::Expected<std::vector<IbvVirtualWc>, Error>
IbvVirtualQp::processCompletion(const ibv_wc& physicalWc, int32_t deviceId) {
  std::vector<IbvVirtualWc> results;

  // Step 1: Identify QP source (must check both qp_num AND deviceId for
  // multi-NIC correctness — different devices can assign the same qp_num)
  bool isNotifyQp = hasNotifyQp() &&
      (physicalWc.qp_num == notifyQp_->getQpNum()) &&
      (deviceId == notifyQp_->getDeviceId());

  // Step 2: Dispatch based on 2x2 matrix (QP type x direction)
  if (isNotifyQp) {
    bool isSend = isSendOpcode(physicalWc.opcode);
    return isSend ? processNotifyQpSendCompletion(physicalWc, results)
                  : processNotifyQpRecvCompletion(physicalWc, results);
  } else {
    auto qpIdxIt = qpNumToIdx_.find(QpId{deviceId, physicalWc.qp_num});
    CHECK(qpIdxIt != qpNumToIdx_.end()) << fmt::format(
        "[Ibverbx] unknown physical QP: qpNum={}, deviceId={}",
        physicalWc.qp_num,
        deviceId);
    int qpIdx = qpIdxIt->second;

    bool isSend = isSendOpcode(physicalWc.opcode);
    return isSend ? processDataQpSendCompletion(physicalWc, qpIdx, results)
                  : processDataQpRecvCompletion(physicalWc, qpIdx, results);
  }
}

inline folly::Expected<std::vector<IbvVirtualWc>, Error>
IbvVirtualQp::processCompletions(
    const ibv_wc* physicalWcs,
    int count,
    int32_t deviceId) {
  std::vector<IbvVirtualWc> allResults;
  allResults.reserve(count);

  for (int i = 0; i < count; i++) {
    auto result = processCompletion(physicalWcs[i], deviceId);
    if (result.hasError()) {
      return folly::makeUnexpected(result.error());
    }
    for (auto& r : *result) {
      allResults.push_back(std::move(r));
    }
  }

  return allResults;
}

// NotifyQp Send completion (SPRAY sender's notify done)
inline folly::Expected<std::vector<IbvVirtualWc>, Error>
IbvVirtualQp::processNotifyQpSendCompletion(
    const ibv_wc& physicalWc,
    std::vector<IbvVirtualWc>& results) {
  auto popResult = popPhysicalQueueStatus(
      notifyQp_->physicalSendWrStatus_, physicalWc.wr_id, "notifyQpSend");
  if (popResult.hasError()) {
    return folly::makeUnexpected(popResult.error());
  }
  uint64_t internalWrId = popResult.value();

  auto updateResult = updateWrState(
      sendTracker_, internalWrId, physicalWc.status, physicalWc.opcode);
  if (updateResult.hasError()) {
    return folly::makeUnexpected(updateResult.error());
  }

  auto reportResult = reportSendCompletions(results);
  if (reportResult.hasError()) {
    return folly::makeUnexpected(reportResult.error());
  }

  auto flushResult = flushPendingSendNotifies();
  if (flushResult.hasError()) {
    return folly::makeUnexpected(flushResult.error());
  }

  return results;
}

// NotifyQp Recv completion (SPRAY receiver's notify arrived)
inline folly::Expected<std::vector<IbvVirtualWc>, Error>
IbvVirtualQp::processNotifyQpRecvCompletion(
    const ibv_wc& physicalWc,
    std::vector<IbvVirtualWc>& results) {
  auto popResult = popPhysicalQueueStatus(
      notifyQp_->physicalRecvWrStatus_, physicalWc.wr_id, "notifyQpRecv");
  if (popResult.hasError()) {
    return folly::makeUnexpected(popResult.error());
  }
  uint64_t internalWrId = popResult.value();

  auto updateResult = updateWrState(
      recvTracker_, internalWrId, physicalWc.status, physicalWc.opcode);
  if (updateResult.hasError()) {
    return folly::makeUnexpected(updateResult.error());
  }

  auto reportResult = reportRecvCompletions(results);
  if (reportResult.hasError()) {
    return folly::makeUnexpected(reportResult.error());
  }

  auto flushResult = flushPendingRecvNotifies();
  if (flushResult.hasError()) {
    return folly::makeUnexpected(flushResult.error());
  }

  return results;
}

// DataQp Send completion (data fragment completed)
inline folly::Expected<std::vector<IbvVirtualWc>, Error>
IbvVirtualQp::processDataQpSendCompletion(
    const ibv_wc& physicalWc,
    int qpIdx,
    std::vector<IbvVirtualWc>& results) {
  auto& physicalQp = physicalQps_.at(qpIdx);

  auto popResult = popPhysicalQueueStatus(
      physicalQp.physicalSendWrStatus_, physicalWc.wr_id, "dataQpSend");
  if (popResult.hasError()) {
    return folly::makeUnexpected(popResult.error());
  }
  uint64_t internalWrId = popResult.value();

  auto updateResult = updateWrState(
      sendTracker_, internalWrId, physicalWc.status, physicalWc.opcode);
  if (updateResult.hasError()) {
    return folly::makeUnexpected(updateResult.error());
  }

  auto reportResult = reportSendCompletions(results);
  if (reportResult.hasError()) {
    return folly::makeUnexpected(reportResult.error());
  }

  auto dispatchResult = dispatchPendingSends(qpIdx);
  if (dispatchResult.hasError()) {
    return folly::makeUnexpected(dispatchResult.error());
  }

  return results;
}

// DataQp Recv completion (DQPLB recv with sequence number)
inline folly::Expected<std::vector<IbvVirtualWc>, Error>
IbvVirtualQp::processDataQpRecvCompletion(
    const ibv_wc& physicalWc,
    int qpIdx,
    std::vector<IbvVirtualWc>& results) {
  auto& physicalQp = physicalQps_.at(qpIdx);

  auto popResult = popPhysicalQueueStatus(
      physicalQp.physicalRecvWrStatus_, physicalWc.wr_id, "dataQpRecv");
  if (popResult.hasError()) {
    return folly::makeUnexpected(popResult.error());
  }

  int notifyCount = dqplbSeqTracker.processReceivedImm(physicalWc.imm_data);

  for (int i = 0; i < notifyCount; i++) {
    CHECK(recvTracker_.hasPendingCompletion()) << fmt::format(
        "[Ibverbx] DQPLB notifyCount={} exceeds outstanding recvs",
        notifyCount);

    uint64_t frontId = recvTracker_.frontPendingCompletion();
    auto* frontWr = recvTracker_.find(frontId);
    CHECK(frontWr) << fmt::format(
        "[Ibverbx] DQPLB WR {} not found in recvTracker_", frontId);

    frontWr->remainingMsgCnt--;
    frontWr->wcOpcode = physicalWc.opcode;

    if (frontWr->aggregatedStatus == IBV_WC_SUCCESS &&
        physicalWc.status != IBV_WC_SUCCESS) {
      frontWr->aggregatedStatus = physicalWc.status;
    }

    auto reportResult = reportRecvCompletions(results);
    if (reportResult.hasError()) {
      return folly::makeUnexpected(reportResult.error());
    }
  }

  auto replenishResult = replenishDqplbRecv(qpIdx);
  if (replenishResult.hasError()) {
    return folly::makeUnexpected(replenishResult.error());
  }

  return results;
}

} // namespace ibverbx
