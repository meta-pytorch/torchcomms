// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/IbvVirtualQp.h"

#include <folly/json.h>
#include <unordered_set>
#include "comms/ctran/ibverbx/IbvVirtualCq.h"
#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

/*** IbvVirtualQp ***/

IbvVirtualQp::IbvVirtualQp(
    std::vector<IbvQp>&& qps,
    IbvVirtualCq* virtualCq,
    int maxMsgCntPerQp,
    int maxMsgSize,
    LoadBalancingScheme loadBalancingScheme,
    std::optional<IbvQp>&& notifyQp)
    : virtualCq_(virtualCq),
      physicalQps_(std::move(qps)),
      maxMsgCntPerQp_(maxMsgCntPerQp),
      maxMsgSize_(maxMsgSize),
      loadBalancingScheme_(loadBalancingScheme),
      notifyQp_(std::move(notifyQp)) {
  CHECK(!physicalQps_.empty()) << "At least one physical QP must be provided!";
  CHECK(physicalQps_.size() == 1 || notifyQp_.has_value())
      << "notifyQp must be provided when using multiple data QPs!";

  virtualQpNum_ =
      nextVirtualQpNum_.fetch_add(1); // Assign unique virtual QP number

  for (int i = 0; i < physicalQps_.size(); i++) {
    qpNumToIdx_[QpId{
        physicalQps_.at(i).getDeviceId(), physicalQps_.at(i).qp()->qp_num}] = i;
  }

  // Calculate the number of unique devices that the physical QPs span
  std::unordered_set<uint32_t> uniqueDevices;
  for (const auto& qp : physicalQps_) {
    uniqueDevices.insert(qp.getDeviceId());
  }
  if (hasNotifyQp()) {
    uniqueDevices.insert(notifyQp_->getDeviceId());
  }
  deviceCnt_ = uniqueDevices.size();

  isMultiQp_ = (physicalQps_.size() > 1);

  // Register the virtual QP and all its mappings with the coordinator
  auto coordinator = Coordinator::getCoordinator();
  CHECK(coordinator)
      << "Coordinator should not be nullptr during IbvVirtualQp construction!";

  // Use the consolidated registration API
  coordinator->registerVirtualQpWithVirtualCqMappings(
      this, virtualCq->getVirtualCqNum(), virtualCq->getVirtualCqNum());

  // Register with VirtualCq (v2 path, dual registration)
  registerWithVirtualCq();
}

size_t IbvVirtualQp::getTotalQps() const {
  return physicalQps_.size();
}

const std::vector<IbvQp>& IbvVirtualQp::getQpsRef() const {
  return physicalQps_;
}

std::vector<IbvQp>& IbvVirtualQp::getQpsRef() {
  return physicalQps_;
}

const IbvQp& IbvVirtualQp::getNotifyQpRef() const {
  return notifyQp_.value();
}

IbvQp& IbvVirtualQp::getNotifyQpRef() {
  return notifyQp_.value();
}

uint32_t IbvVirtualQp::getVirtualQpNum() const {
  return virtualQpNum_;
}

IbvVirtualQp::IbvVirtualQp(IbvVirtualQp&& other) noexcept
    : virtualCq_(other.virtualCq_),
      isMultiQp_(other.isMultiQp_),
      sendTracker_(std::move(other.sendTracker_)),
      recvTracker_(std::move(other.recvTracker_)),
      pendingSendNotifyQue_(std::move(other.pendingSendNotifyQue_)),
      pendingRecvNotifyQue_(std::move(other.pendingRecvNotifyQue_)),
      pendingSendVirtualWrQue_(std::move(other.pendingSendVirtualWrQue_)),
      pendingRecvVirtualWrQue_(std::move(other.pendingRecvVirtualWrQue_)),
      virtualQpNum_(std::move(other.virtualQpNum_)),
      physicalQps_(std::move(other.physicalQps_)),
      qpNumToIdx_(std::move(other.qpNumToIdx_)),
      nextSendPhysicalQpIdx_(std::move(other.nextSendPhysicalQpIdx_)),
      nextRecvPhysicalQpIdx_(std::move(other.nextRecvPhysicalQpIdx_)),
      maxMsgCntPerQp_(std::move(other.maxMsgCntPerQp_)),
      maxMsgSize_(std::move(other.maxMsgSize_)),
      nextPhysicalWrId_(std::move(other.nextPhysicalWrId_)),
      deviceCnt_(std::move(other.deviceCnt_)),
      loadBalancingScheme_(std::move(other.loadBalancingScheme_)),
      pendingSendNotifyVirtualWrQue_(
          std::move(other.pendingSendNotifyVirtualWrQue_)),
      notifyQp_(std::move(other.notifyQp_)),
      dqplbSeqTracker(std::move(other.dqplbSeqTracker)),
      dqplbReceiverInitialized_(std::move(other.dqplbReceiverInitialized_)) {
  other.virtualCq_ = nullptr; // Prevent double-unregister

  // Update coordinator pointer mapping for this virtual QP after move
  auto coordinator = Coordinator::getCoordinator();
  CHECK(coordinator)
      << "Coordinator should not be nullptr during IbvVirtualQp move construction!";
  coordinator->updateVirtualQpPointer(virtualQpNum_, this);

  // Re-register with VirtualCq (v2 path)
  registerWithVirtualCq();
}

IbvVirtualQp& IbvVirtualQp::operator=(IbvVirtualQp&& other) noexcept {
  if (this != &other) {
    // Unregister current QPs from VirtualCq before moving
    unregisterFromVirtualCq();

    physicalQps_ = std::move(other.physicalQps_);
    notifyQp_ = std::move(other.notifyQp_);
    nextSendPhysicalQpIdx_ = std::move(other.nextSendPhysicalQpIdx_);
    nextRecvPhysicalQpIdx_ = std::move(other.nextRecvPhysicalQpIdx_);
    qpNumToIdx_ = std::move(other.qpNumToIdx_);
    maxMsgCntPerQp_ = std::move(other.maxMsgCntPerQp_);
    maxMsgSize_ = std::move(other.maxMsgSize_);
    deviceCnt_ = std::move(other.deviceCnt_);
    loadBalancingScheme_ = std::move(other.loadBalancingScheme_);
    pendingSendVirtualWrQue_ = std::move(other.pendingSendVirtualWrQue_);
    pendingRecvVirtualWrQue_ = std::move(other.pendingRecvVirtualWrQue_);
    virtualQpNum_ = std::move(other.virtualQpNum_);
    nextPhysicalWrId_ = std::move(other.nextPhysicalWrId_);
    pendingSendNotifyVirtualWrQue_ =
        std::move(other.pendingSendNotifyVirtualWrQue_);
    dqplbSeqTracker = std::move(other.dqplbSeqTracker);
    dqplbReceiverInitialized_ = std::move(other.dqplbReceiverInitialized_);
    virtualCq_ = other.virtualCq_;
    isMultiQp_ = other.isMultiQp_;
    sendTracker_ = std::move(other.sendTracker_);
    recvTracker_ = std::move(other.recvTracker_);
    pendingSendNotifyQue_ = std::move(other.pendingSendNotifyQue_);
    pendingRecvNotifyQue_ = std::move(other.pendingRecvNotifyQue_);

    other.virtualCq_ = nullptr; // Prevent double-unregister

    // Update coordinator pointer mapping for this virtual QP after move
    auto coordinator = Coordinator::getCoordinator();
    CHECK(coordinator)
        << "Coordinator should not be nullptr during IbvVirtualQp move assignment!";
    coordinator->updateVirtualQpPointer(virtualQpNum_, this);

    // Re-register with VirtualCq (v2 path)
    registerWithVirtualCq();
  }
  return *this;
}

IbvVirtualQp::~IbvVirtualQp() {
  // Unregister from VirtualCq (v2 path)
  unregisterFromVirtualCq();

  // Always call unregister - the coordinator will check if the pointer matches
  // and do nothing if the object was moved
  auto coordinator = Coordinator::getCoordinator();
  CHECK(coordinator)
      << "Coordinator should not be nullptr during IbvVirtualQp destruction!";
  coordinator->unregisterVirtualQp(virtualQpNum_, this);
}

void IbvVirtualQp::registerWithVirtualCq() {
  if (virtualCq_ == nullptr) {
    return;
  }

  for (size_t i = 0; i < physicalQps_.size(); i++) {
    virtualCq_->registerPhysicalQp(
        physicalQps_.at(i).qp()->qp_num,
        physicalQps_.at(i).getDeviceId(),
        this,
        isMultiQp_,
        virtualQpNum_);
  }

  if (hasNotifyQp()) {
    virtualCq_->registerPhysicalQp(
        notifyQp_->qp()->qp_num,
        notifyQp_->getDeviceId(),
        this,
        isMultiQp_,
        virtualQpNum_);
  }
}

void IbvVirtualQp::unregisterFromVirtualCq() {
  if (virtualCq_ == nullptr) {
    return;
  }

  for (size_t i = 0; i < physicalQps_.size(); i++) {
    virtualCq_->unregisterPhysicalQp(
        physicalQps_.at(i).qp()->qp_num, physicalQps_.at(i).getDeviceId());
  }

  if (hasNotifyQp()) {
    virtualCq_->unregisterPhysicalQp(
        notifyQp_->qp()->qp_num, notifyQp_->getDeviceId());
  }
}

folly::Expected<folly::Unit, Error> IbvVirtualQp::modifyVirtualQp(
    ibv_qp_attr* attr,
    int attrMask,
    const IbvVirtualQpBusinessCard& businessCard) {
  // If businessCard is not empty, use it to modify QPs with specific
  // dest_qp_num values
  if (!businessCard.qpNums_.empty()) {
    // Make sure the businessCard has the same number of QPs as physicalQps_
    if (businessCard.qpNums_.size() != physicalQps_.size()) {
      return folly::makeUnexpected(Error(
          EINVAL, "BusinessCard QP count doesn't match physical QP count"));
    }

    // Modify each QP with its corresponding dest_qp_num from the businessCard
    for (auto i = 0; i < physicalQps_.size(); i++) {
      attr->dest_qp_num = businessCard.qpNums_.at(i);
      auto maybeModifyQp = physicalQps_.at(i).modifyQp(attr, attrMask);
      if (maybeModifyQp.hasError()) {
        return folly::makeUnexpected(maybeModifyQp.error());
      }
    }
    // Only modify notifyQp if it exists
    if (hasNotifyQp()) {
      attr->dest_qp_num = businessCard.notifyQpNum_;
      auto maybeModifyQp = notifyQp_->modifyQp(attr, attrMask);
      if (maybeModifyQp.hasError()) {
        return folly::makeUnexpected(maybeModifyQp.error());
      }
    }
  } else {
    // If no businessCard provided, modify all QPs with the same attributes
    for (auto& qp : physicalQps_) {
      auto maybeModifyQp = qp.modifyQp(attr, attrMask);
      if (maybeModifyQp.hasError()) {
        return folly::makeUnexpected(maybeModifyQp.error());
      }
    }
    if (hasNotifyQp()) {
      auto maybeModifyQp = notifyQp_->modifyQp(attr, attrMask);
      if (maybeModifyQp.hasError()) {
        return folly::makeUnexpected(maybeModifyQp.error());
      }
    }
  }
  return folly::unit;
}

IbvVirtualQpBusinessCard IbvVirtualQp::getVirtualQpBusinessCard() const {
  std::vector<uint32_t> qpNums;
  qpNums.reserve(physicalQps_.size());
  for (auto& qp : physicalQps_) {
    qpNums.push_back(qp.qp()->qp_num);
  }
  uint32_t notifyQpNum = hasNotifyQp() ? notifyQp_->qp()->qp_num : 0;
  return IbvVirtualQpBusinessCard(std::move(qpNums), notifyQpNum);
}

LoadBalancingScheme IbvVirtualQp::getLoadBalancingScheme() const {
  return loadBalancingScheme_;
}

/*** IbvVirtualQpBusinessCard ***/

IbvVirtualQpBusinessCard::IbvVirtualQpBusinessCard(
    std::vector<uint32_t> qpNums,
    uint32_t notifyQpNum)
    : qpNums_(std::move(qpNums)), notifyQpNum_(notifyQpNum) {}

folly::dynamic IbvVirtualQpBusinessCard::toDynamic() const {
  folly::dynamic obj = folly::dynamic::object;
  folly::dynamic qpNumsArray = folly::dynamic::array;

  // Use fixed-width string formatting to ensure consistent size
  // All uint32_t values will be formatted as 10-digit zero-padded strings
  for (const auto& qpNum : qpNums_) {
    std::string paddedQpNum = fmt::format("{:010d}", qpNum);
    qpNumsArray.push_back(paddedQpNum);
  }

  obj["qpNums"] = std::move(qpNumsArray);
  obj["notifyQpNum"] = fmt::format("{:010d}", notifyQpNum_);
  return obj;
}

folly::Expected<IbvVirtualQpBusinessCard, Error>
IbvVirtualQpBusinessCard::fromDynamic(const folly::dynamic& obj) {
  std::vector<uint32_t> qpNums;

  if (obj.count("qpNums") > 0 && obj["qpNums"].isArray()) {
    const auto& qpNumsArray = obj["qpNums"];
    qpNums.reserve(qpNumsArray.size());

    for (const auto& qpNum : qpNumsArray) {
      CHECK(qpNum.isString()) << "qp num is not string!";
      try {
        uint32_t qpNumValue =
            static_cast<uint32_t>(std::stoul(qpNum.asString()));
        qpNums.push_back(qpNumValue);
      } catch (const std::exception& e) {
        return folly::makeUnexpected(Error(
            EINVAL,
            fmt::format(
                "Invalid QP number string format: {}. Exception: {}",
                qpNum.asString(),
                e.what())));
      }
    }
  } else {
    return folly::makeUnexpected(
        Error(EINVAL, "Invalid qpNums array received from remote side"));
  }

  uint32_t notifyQpNum = 0; // Default value for backwards compatibility
  if (obj.count("notifyQpNum") > 0 && obj["notifyQpNum"].isString()) {
    try {
      notifyQpNum =
          static_cast<uint32_t>(std::stoul(obj["notifyQpNum"].asString()));
    } catch (const std::exception& e) {
      return folly::makeUnexpected(Error(
          EINVAL,
          fmt::format(
              "Invalid notifyQpNum string format: {}. Exception: {}",
              obj["notifyQpNum"].asString(),
              e.what())));
    }
  }

  return IbvVirtualQpBusinessCard(std::move(qpNums), notifyQpNum);
}

std::string IbvVirtualQpBusinessCard::serialize() const {
  return folly::toJson(toDynamic());
}

folly::Expected<IbvVirtualQpBusinessCard, Error>
IbvVirtualQpBusinessCard::deserialize(const std::string& jsonStr) {
  try {
    folly::dynamic obj = folly::parseJson(jsonStr);
    return fromDynamic(obj);
  } catch (const std::exception& e) {
    return folly::makeUnexpected(Error(
        EINVAL,
        fmt::format(
            "Failed to parse JSON in IbvVirtualQpBusinessCard Deserialize. Exception: {}",
            e.what())));
  }
}
} // namespace ibverbx
