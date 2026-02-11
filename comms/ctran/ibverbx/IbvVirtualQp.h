// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <folly/Expected.h>
#include <folly/dynamic.h>
#include <deque>
#include <optional>

#include "comms/ctran/ibverbx/Coordinator.h"
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

  inline folly::Expected<folly::Unit, Error> postSend(
      ibv_send_wr* sendWr,
      ibv_send_wr* sendWrBad,
      const std::unordered_map<int32_t, MemoryRegionKeys>& deviceIdToKeys = {});

  inline folly::Expected<folly::Unit, Error> postRecv(
      ibv_recv_wr* ibvRecvWr,
      ibv_recv_wr* badIbvRecvWr);

  inline int findAvailableSendQp();
  inline int findAvailableRecvQp();

  inline folly::Expected<VirtualQpResponse, Error> processRequest(
      VirtualQpRequest&& request);

  // v2 completion processing: Route physical CQEs to virtual WR state.
  // Stub implementation until postSend_v2/postRecv_v2 are added.
  inline folly::Expected<std::vector<IbvVirtualWc>, Error> processCompletion(
      const ibv_wc& physicalWc,
      int32_t deviceId);

 private:
#ifdef IBVERBX_TEST_FRIENDS
  IBVERBX_TEST_FRIENDS
#endif

  // updatePhysicalSendWrFromVirtualSendWr is a helper function to update
  // physical send work request (ibv_send_wr) from virtual send work request
  inline void updatePhysicalSendWrFromVirtualSendWr(
      VirtualSendWr& virtualSendWr,
      ibv_send_wr* sendWr,
      ibv_sge* sendSg,
      int32_t deviceId = 0);

  friend class IbvPd;
  friend class IbvVirtualCq;

  // Pointer to the VirtualCq this VirtualQp is registered with (for v2 path)
  IbvVirtualCq* virtualCq_{nullptr};

  std::deque<VirtualSendWr> pendingSendVirtualWrQue_;
  std::deque<VirtualRecvWr> pendingRecvVirtualWrQue_;

  inline static std::atomic<uint32_t> nextVirtualQpNum_{
      0}; // Static counter for assigning unique virtual QP numbers
  uint32_t virtualQpNum_{0}; // The unique virtual QP number assigned to
                             // instance of IbvVirtualQp.

  std::vector<IbvQp> physicalQps_;
  std::unordered_map<int, int> qpNumToIdx_;

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
  std::deque<VirtualSendWr> pendingSendNotifyVirtualWrQue_;
  std::optional<IbvQp> notifyQp_;

  // DQPLB mode specific fields and functions
  DqplbSeqTracker dqplbSeqTracker;
  bool dqplbReceiverInitialized_{
      false}; // flag to indicate if dqplb receiver is initialized
  inline folly::Expected<folly::Unit, Error> initializeDqplbReceiver();

  // mapPendingSendQueToPhysicalQp is a helper function to iterate through
  // virtualSendWr in the pendingSendVirtualWrQue_, construct physical wrs and
  // call postSend on physical QP. If qpIdx is provided, this function will
  // postSend physicalWr on qpIdx. If qpIdx is not provided, then the function
  // will find an available Qp to postSend the physical work request on.
  inline folly::Expected<folly::Unit, Error> mapPendingSendQueToPhysicalQp(
      int qpIdx = -1);

  // postSendNotifyImm is a helper function to send IMM notification message
  // after all previous messages are sent in a large message
  inline folly::Expected<folly::Unit, Error> postSendNotifyImm();
  inline folly::Expected<folly::Unit, Error> mapPendingRecvQueToPhysicalQp(
      int qpIdx = -1);
  inline folly::Expected<folly::Unit, Error> postRecvNotifyImm(int qpIdx = -1);
};

// IbvVirtualQp inline functions
inline folly::Expected<folly::Unit, Error>
IbvVirtualQp::mapPendingSendQueToPhysicalQp(int qpIdx) {
  while (!pendingSendVirtualWrQue_.empty()) {
    // Get the front of vSendQ_ and obtain the send information
    VirtualSendWr& virtualSendWr = pendingSendVirtualWrQue_.front();

    // For Send opcodes related to RDMA_WRITE operations, use user selected load
    // balancing scheme specified in loadBalancingScheme_. For all other
    // opcodes, default to using physical QP 0.
    auto availableQpIdx = -1;
    if (virtualSendWr.wr.opcode == IBV_WR_RDMA_WRITE ||
        virtualSendWr.wr.opcode == IBV_WR_RDMA_WRITE_WITH_IMM ||
        virtualSendWr.wr.opcode == IBV_WR_RDMA_READ) {
      // Find an available Qp to send
      availableQpIdx = qpIdx == -1 ? findAvailableSendQp() : qpIdx;
      qpIdx = -1; // If qpIdx is provided, it indicates that one slot has been
                  // freed for the corresponding qpIdx. After using this slot,
                  // reset qpIdx to -1.
    } else if (
        physicalQps_.at(0).physicalSendWrStatus_.size() < maxMsgCntPerQp_) {
      availableQpIdx = 0;
    }
    if (availableQpIdx == -1) {
      break;
    }

    // Update the physical send work request with virtual one
    ibv_send_wr sendWr_{};
    ibv_sge sendSg_{};
    updatePhysicalSendWrFromVirtualSendWr(
        virtualSendWr,
        &sendWr_,
        &sendSg_,
        physicalQps_.at(availableQpIdx).getDeviceId());

    // Call ibv_post_send to send the message
    ibv_send_wr badSendWr_{};
    auto maybeSend =
        physicalQps_.at(availableQpIdx).postSend(&sendWr_, &badSendWr_);
    if (maybeSend.hasError()) {
      return folly::makeUnexpected(maybeSend.error());
    }

    // Enqueue the send information to physicalQps_
    physicalQps_.at(availableQpIdx)
        .physicalSendWrStatus_.emplace_back(
            sendWr_.wr_id, virtualSendWr.wr.wr_id);

    // Decide if need to deque the front of vSendQ_
    virtualSendWr.offset += sendWr_.sg_list->length;
    virtualSendWr.remainingMsgCnt--;
    if (virtualSendWr.remainingMsgCnt == 0) {
      pendingSendVirtualWrQue_.pop_front();
    } else if (
        virtualSendWr.remainingMsgCnt == 1 &&
        virtualSendWr.sendExtraNotifyImm) {
      // Move front entry from pendingSendVirtualWrQue_ to
      // pendingSendNotifyVirtualWrQue_
      pendingSendNotifyVirtualWrQue_.push_back(
          std::move(pendingSendVirtualWrQue_.front()));
      pendingSendVirtualWrQue_.pop_front();
    }
  }
  return folly::unit;
}

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

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postSendNotifyImm() {
  auto virtualSendWr = pendingSendNotifyVirtualWrQue_.front();
  ibv_send_wr sendWr_{};
  ibv_send_wr badSendWr_{};
  ibv_sge sendSg_{};
  sendWr_.next = nullptr;
  sendWr_.sg_list = &sendSg_;
  sendWr_.num_sge = 0;
  sendWr_.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  sendWr_.send_flags = IBV_SEND_SIGNALED;
  sendWr_.wr.rdma.remote_addr = virtualSendWr.wr.wr.rdma.remote_addr;
  sendWr_.wr.rdma.rkey = virtualSendWr.wr.wr.rdma.rkey;
  sendWr_.imm_data = virtualSendWr.wr.imm_data;
  sendWr_.wr_id = nextPhysicalWrId_++;
  auto maybeSend = notifyQp_->postSend(&sendWr_, &badSendWr_);
  if (maybeSend.hasError()) {
    return folly::makeUnexpected(maybeSend.error());
  }
  notifyQp_->physicalSendWrStatus_.emplace_back(
      sendWr_.wr_id, virtualSendWr.wr.wr_id);
  virtualSendWr.remainingMsgCnt = 0;
  pendingSendNotifyVirtualWrQue_.pop_front();
  return folly::unit;
}

inline void IbvVirtualQp::updatePhysicalSendWrFromVirtualSendWr(
    VirtualSendWr& virtualSendWr,
    ibv_send_wr* sendWr,
    ibv_sge* sendSg,
    int32_t deviceId) {
  sendWr->wr_id = nextPhysicalWrId_++;

  auto lenToSend = std::min(
      int(virtualSendWr.wr.sg_list->length - virtualSendWr.offset),
      maxMsgSize_);
  sendSg->addr = virtualSendWr.wr.sg_list->addr + virtualSendWr.offset;
  sendSg->length = lenToSend;
  sendSg->lkey = virtualSendWr.deviceIdToKeys.empty()
      ? virtualSendWr.wr.sg_list->lkey
      : virtualSendWr.deviceIdToKeys.at(deviceId)
            .lkey; // Use lkey from deviceIdToKeys if not empty, otherwise from
                   // virtualSendWr.wr
  sendWr->next = nullptr;
  sendWr->sg_list = sendSg;
  sendWr->num_sge = 1;

  // Set the opcode to the same as virtual wr, except for RDMA_WRITE_WITH_IMM,
  // we'll handle the notification message separately
  switch (virtualSendWr.wr.opcode) {
    case IBV_WR_RDMA_WRITE:
    case IBV_WR_RDMA_READ:
      sendWr->opcode = virtualSendWr.wr.opcode;
      sendWr->send_flags = virtualSendWr.wr.send_flags;
      sendWr->wr.rdma.remote_addr =
          virtualSendWr.wr.wr.rdma.remote_addr + virtualSendWr.offset;

      sendWr->wr.rdma.rkey = virtualSendWr.deviceIdToKeys.empty()
          ? virtualSendWr.wr.wr.rdma.rkey
          : virtualSendWr.deviceIdToKeys.at(deviceId)
                .rkey; // Use rkey from deviceIdToKeys if not empty, otherwise
                       // from virtualSendWr.wr
      break;
    case IBV_WR_RDMA_WRITE_WITH_IMM:
      sendWr->opcode = (loadBalancingScheme_ == LoadBalancingScheme::SPRAY)
          ? IBV_WR_RDMA_WRITE
          : IBV_WR_RDMA_WRITE_WITH_IMM;
      sendWr->send_flags = IBV_SEND_SIGNALED;
      sendWr->wr.rdma.remote_addr =
          virtualSendWr.wr.wr.rdma.remote_addr + virtualSendWr.offset;
      sendWr->wr.rdma.rkey = virtualSendWr.deviceIdToKeys.empty()
          ? virtualSendWr.wr.wr.rdma.rkey
          : virtualSendWr.deviceIdToKeys.at(deviceId)
                .rkey; // Use rkey from deviceIdToKeys if not empty, otherwise
                       // from virtualSendWr.wr
      break;
    case IBV_WR_SEND:
      sendWr->opcode = virtualSendWr.wr.opcode;
      sendWr->send_flags = virtualSendWr.wr.send_flags;
      break;

    default:
      break;
  }

  if (sendWr->opcode == IBV_WR_RDMA_WRITE_WITH_IMM &&
      loadBalancingScheme_ == LoadBalancingScheme::DQPLB) {
    sendWr->imm_data =
        dqplbSeqTracker.getSendImm(virtualSendWr.remainingMsgCnt);
  }
}

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postSend(
    ibv_send_wr* sendWr,
    ibv_send_wr* sendWrBad,
    const std::unordered_map<int32_t, MemoryRegionKeys>& deviceIdToKeys) {
  // Report error if deviceCnt_ > 1 and deviceIdToKeys is not provided
  if (deviceCnt_ > 1 && deviceIdToKeys.empty()) {
    return folly::makeUnexpected(Error(
        EINVAL,
        "In IbvVirtualQp::postSend, deviceIdToKeys must be provided when using multiple NICs"));
  }

  // Report error if num_sge is more than 1
  if (sendWr->num_sge > 1) {
    return folly::makeUnexpected(Error(
        EINVAL, "In IbvVirtualQp::postSend, num_sge > 1 is not supported"));
  }

  // Report error if opcode is not supported by Ibverbx virtualQp
  switch (sendWr->opcode) {
    case IBV_WR_SEND_WITH_IMM:
    case IBV_WR_ATOMIC_CMP_AND_SWP:
    case IBV_WR_ATOMIC_FETCH_AND_ADD:
      return folly::makeUnexpected(Error(
          EINVAL,
          "In IbvVirtualQp::postSend, opcode IBV_WR_SEND_WITH_IMM, IBV_WR_ATOMIC_CMP_AND_SWP, IBV_WR_ATOMIC_FETCH_AND_ADD are not supported"));

    default:
      break;
  }

  // Calculate the chunk number for the current message and update sendWqe
  bool sendExtraNotifyImm =
      (sendWr->opcode == IBV_WR_RDMA_WRITE_WITH_IMM &&
       loadBalancingScheme_ == LoadBalancingScheme::SPRAY);
  int expectedMsgCnt =
      (sendWr->sg_list->length + maxMsgSize_ - 1) / maxMsgSize_;
  if (sendExtraNotifyImm) {
    expectedMsgCnt += 1; // After post send all data messages, will post send
                         // 1 more notification message on QP 0
  }

  // Submit request to virtualCq to enqueue VirtualWc
  VirtualCqRequest request = {
      .type = RequestType::SEND,
      .virtualQpNum = (int)virtualQpNum_,
      .expectedMsgCnt = expectedMsgCnt,
      .sendWr = sendWr,
      .sendExtraNotifyImm = sendExtraNotifyImm};
  auto coordinator = Coordinator::getCoordinator();
  CHECK(coordinator) << "Coordinator should not be nullptr during postSend!";
  coordinator->submitRequestToVirtualCq(std::move(request));

  // Set up the send work request with the completion queue entry and enqueue
  // Note: The VirtualSendWr constructor will handle deep copying of sendWr,
  // sg_list, and deviceIdToKeys (if provided; otherwise deviceIdToKeys will
  // be empty)
  pendingSendVirtualWrQue_.emplace_back(
      *sendWr,
      expectedMsgCnt,
      expectedMsgCnt,
      sendExtraNotifyImm,
      deviceIdToKeys);

  // Map large messages from vSendQ_ to pQps_
  if (mapPendingSendQueToPhysicalQp().hasError()) {
    *sendWrBad = *sendWr;
    return folly::makeUnexpected(Error(errno));
  }

  return folly::unit;
}

inline folly::Expected<VirtualQpResponse, Error> IbvVirtualQp::processRequest(
    VirtualQpRequest&& request) {
  VirtualQpResponse response;
  // If request.physicalQpNum differs from notifyQpNum, locate the corresponding
  // physical qpIdx to process this request.
  auto qpIdx = (hasNotifyQp() && request.physicalQpNum == notifyQp_->getQpNum())
      ? -1
      : qpNumToIdx_.at(request.physicalQpNum);
  // If qpIdx is -1, physicalQp is notifyQp; otherwise, physicalQp is the qpIdx
  // entry of physicalQps_
  auto& physicalQp = qpIdx == -1 ? *notifyQp_ : physicalQps_.at(qpIdx);

  if (request.type == RequestType::RECV) {
    if (physicalQp.physicalRecvWrStatus_.empty()) {
      return folly::makeUnexpected(Error(
          EINVAL,
          fmt::format(
              "In pollCq, after calling submit command to IbvVirtualQp, \
              physicalRecvWrStatus_ at physicalQp {} is empty!",
              request.physicalQpNum)));
    }

    auto& physicalRecvWrStatus = physicalQp.physicalRecvWrStatus_.front();

    if (physicalRecvWrStatus.physicalWrId != request.wrId) {
      return folly::makeUnexpected(Error(
          EINVAL,
          fmt::format(
              "In pollCq, after calling submit command to IbvVirtualQp, \
              physicalRecvWrStatus.physicalWrId({}) != request.wrId({})",
              physicalRecvWrStatus.physicalWrId,
              request.wrId)));
    }

    response.virtualWrId = physicalRecvWrStatus.virtualWrId;
    physicalQp.physicalRecvWrStatus_.pop_front();
    if (loadBalancingScheme_ == LoadBalancingScheme::DQPLB) {
      if (postRecvNotifyImm(qpIdx).hasError()) {
        return folly::makeUnexpected(
            Error(errno, fmt::format("postRecvNotifyImm() failed!")));
      }
      response.notifyCount =
          dqplbSeqTracker.processReceivedImm(request.immData);
      response.useDqplb = true;
    } else if (qpIdx != -1) {
      if (mapPendingRecvQueToPhysicalQp(qpIdx).hasError()) {
        return folly::makeUnexpected(Error(
            errno,
            fmt::format("mapPendingRecvQueToPhysicalQp({}) failed!", qpIdx)));
      }
    }
  } else if (request.type == RequestType::SEND) {
    if (physicalQp.physicalSendWrStatus_.empty()) {
      return folly::makeUnexpected(Error(
          EINVAL,
          fmt::format(
              "In pollCq, after calling submit command to IbvVirtualQp, \
              physicalSendWrStatus_ at physicalQp {} is empty!",
              request.physicalQpNum)));
    }

    auto physicalSendWrStatus = physicalQp.physicalSendWrStatus_.front();

    if (physicalSendWrStatus.physicalWrId != request.wrId) {
      return folly::makeUnexpected(Error(
          EINVAL,
          fmt::format(
              "In pollCq, after calling submit command to IbvVirtualQp, \
              physicalSendWrStatus.physicalWrId({}) != request.wrId({})",
              physicalSendWrStatus.physicalWrId,
              request.wrId)));
    }

    response.virtualWrId = physicalSendWrStatus.virtualWrId;
    physicalQp.physicalSendWrStatus_.pop_front();
    if (qpIdx != -1) {
      if (mapPendingSendQueToPhysicalQp(qpIdx).hasError()) {
        return folly::makeUnexpected(Error(
            errno,
            fmt::format("mapPendingSendQueToPhysicalQp({}) failed!", qpIdx)));
      }
    }
  } else if (request.type == RequestType::SEND_NOTIFY) {
    if (pendingSendNotifyVirtualWrQue_.empty()) {
      return folly::makeUnexpected(Error(
          EINVAL,
          fmt::format(
              "Tried to post send notify IMM message for wrId {} when pendingSendNotifyVirtualWrQue_ is empty",
              request.wrId)));
    }

    if (pendingSendNotifyVirtualWrQue_.front().wr.wr_id == request.wrId) {
      if (postSendNotifyImm().hasError()) {
        return folly::makeUnexpected(
            Error(errno, fmt::format("postSendNotifyImm() failed!")));
      }
    }
  }
  return response;
}

// Currently, this function is only invoked to receive messages with opcode
// IBV_WR_SEND. Therefore, we restrict its usage to physical QP 0.
// Note: If Dynamic QP Load Balancing (DQPLB) or other load balancing techniques
// are required in the future, this function can be updated to support more
// advanced usage.
inline int IbvVirtualQp::findAvailableRecvQp() {
  // maxMsgCntPerQp_ with a value of -1 indicates there is no limit.
  auto availableQpIdx = -1;
  if (maxMsgCntPerQp_ == -1 ||
      physicalQps_.at(0).physicalRecvWrStatus_.size() < maxMsgCntPerQp_) {
    availableQpIdx = 0;
  }

  return availableQpIdx;
}

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postRecvNotifyImm(
    int qpIdx) {
  auto& qp = qpIdx == -1 ? *notifyQp_ : physicalQps_.at(qpIdx);
  auto virtualRecvWrId = loadBalancingScheme_ == LoadBalancingScheme::SPRAY
      ? pendingRecvVirtualWrQue_.front().wr.wr_id
      : -1;
  ibv_recv_wr recvWr_{};
  ibv_recv_wr badRecvWr_{};
  ibv_sge recvSg_{};
  recvWr_.next = nullptr;
  recvWr_.sg_list = &recvSg_;
  recvWr_.num_sge = 0;
  recvWr_.wr_id = nextPhysicalWrId_++;
  auto maybeRecv = qp.postRecv(&recvWr_, &badRecvWr_);
  if (maybeRecv.hasError()) {
    return folly::makeUnexpected(maybeRecv.error());
  }
  qp.physicalRecvWrStatus_.emplace_back(recvWr_.wr_id, virtualRecvWrId);

  if (loadBalancingScheme_ == LoadBalancingScheme::SPRAY) {
    pendingRecvVirtualWrQue_.pop_front();
  }
  return folly::unit;
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

inline folly::Expected<folly::Unit, Error>
IbvVirtualQp::mapPendingRecvQueToPhysicalQp(int qpIdx) {
  while (!pendingRecvVirtualWrQue_.empty()) {
    VirtualRecvWr& virtualRecvWr = pendingRecvVirtualWrQue_.front();

    if (virtualRecvWr.wr.num_sge == 0) {
      auto maybeRecvNotifyImm = postRecvNotifyImm();
      if (maybeRecvNotifyImm.hasError()) {
        return folly::makeUnexpected(maybeRecvNotifyImm.error());
      }
      continue;
    }

    // If num_sge is > 0, then the receive work request is used to receive
    // messages with opcode IBV_WR_SEND. In this scenario, we restrict usage to
    // physical QP 0 only. The reason behind is that, IBV_WR_SEND requires a
    // strict one-to-one correspondence between send and receive WRs. If Dynamic
    // QP Load Balancing (DQPLB) is applied, send and receive WRs may be posted
    // to different physical QPs within the QP list. This mismatch can result in
    // data being delivered to the wrong address, causing data integrity issues.
    auto availableQpIdx = qpIdx != 0 ? findAvailableRecvQp() : qpIdx;
    qpIdx = -1; // If qpIdx is provided, it indicates that one slot has been
                // freed for the corresponding qpIdx. After using this slot,
                // reset qpIdx to -1.
    if (availableQpIdx == -1) {
      break;
    }

    // Get the front of vRecvQ_ and obtain the receive information
    ibv_recv_wr recvWr_{};
    ibv_recv_wr badRecvWr_{};
    ibv_sge recvSg_{};
    int lenToRecv = 0;
    if (virtualRecvWr.wr.num_sge == 1) {
      lenToRecv = std::min(
          int(virtualRecvWr.wr.sg_list->length - virtualRecvWr.offset),
          maxMsgSize_);
      recvSg_.addr = virtualRecvWr.wr.sg_list->addr + virtualRecvWr.offset;
      recvSg_.length = lenToRecv;
      recvSg_.lkey = virtualRecvWr.wr.sg_list->lkey;

      recvWr_.sg_list = &recvSg_;
      recvWr_.num_sge = 1;
    } else {
      recvWr_.sg_list = nullptr;
      recvWr_.num_sge = 0;
    }
    recvWr_.wr_id = nextPhysicalWrId_++;
    recvWr_.next = nullptr;

    // Call ibv_post_recv to receive the message
    auto maybeRecv =
        physicalQps_.at(availableQpIdx).postRecv(&recvWr_, &badRecvWr_);
    if (maybeRecv.hasError()) {
      return folly::makeUnexpected(maybeRecv.error());
    }

    // Enqueue the receive information to physicalQps_
    physicalQps_.at(availableQpIdx)
        .physicalRecvWrStatus_.emplace_back(
            recvWr_.wr_id, virtualRecvWr.wr.wr_id);

    // Decide if need to deque the front of vRecvQ_
    if (virtualRecvWr.wr.num_sge == 1) {
      virtualRecvWr.offset += lenToRecv;
    }
    virtualRecvWr.remainingMsgCnt--;
    if (virtualRecvWr.remainingMsgCnt == 0) {
      pendingRecvVirtualWrQue_.pop_front();
    }
  }
  return folly::unit;
}

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postRecv(
    ibv_recv_wr* recvWr,
    ibv_recv_wr* recvWrBad) {
  // Report error if num_sge is more than 1
  if (recvWr->num_sge > 1) {
    return folly::makeUnexpected(Error(EINVAL));
  }

  int expectedMsgCnt = 1;

  if (recvWr->num_sge == 0) { // recvWr->num_sge == 0 mean it's receiving a
                              // IMM notification message
    expectedMsgCnt = 1;
  } else if (recvWr->num_sge == 1) { // Calculate the chunk number for the
                                     // current message and update recvWqe if
                                     // num_sge is 1
    expectedMsgCnt = (recvWr->sg_list->length + maxMsgSize_ - 1) / maxMsgSize_;
  }

  // Submit request to virtualCq to enqueue VirtualWc
  VirtualCqRequest request = {
      .type = RequestType::RECV,
      .virtualQpNum = (int)virtualQpNum_,
      .expectedMsgCnt = expectedMsgCnt,
      .recvWr = recvWr};
  auto coordinator = Coordinator::getCoordinator();
  CHECK(coordinator) << "Coordinator should not be nullptr during postRecv!";
  coordinator->submitRequestToVirtualCq(std::move(request));

  // Set up the recv work request with the completion queue entry and enqueue
  pendingRecvVirtualWrQue_.emplace_back(
      *recvWr, expectedMsgCnt, expectedMsgCnt);

  if (loadBalancingScheme_ != LoadBalancingScheme::DQPLB) {
    if (mapPendingRecvQueToPhysicalQp().hasError()) {
      // For non-DQPLB modes: map messages from pendingRecvVirtualWrQue_ to
      // physicalQps_. In DQPLB mode, this mapping is unnecessary because all
      // receive notify IMM operations are pre-posted to the QPs before postRecv
      // is called.
      *recvWrBad = *recvWr;
      return folly::makeUnexpected(Error(errno));
    }
  } else if (dqplbReceiverInitialized_ == false) {
    if (initializeDqplbReceiver().hasError()) {
      *recvWrBad = *recvWr;
      return folly::makeUnexpected(Error(errno));
    }
  }

  return folly::unit;
}

// Stub: processCompletion will be implemented when postSend_v2/postRecv_v2
// are added. For now, returns empty vector (no v2 WRs are tracked yet).
inline folly::Expected<std::vector<IbvVirtualWc>, Error>
IbvVirtualQp::processCompletion(
    const ibv_wc& /*physicalWc*/,
    int32_t /*deviceId*/) {
  return std::vector<IbvVirtualWc>{};
}

} // namespace ibverbx
