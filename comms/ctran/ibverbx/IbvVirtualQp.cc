// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/IbvVirtualQp.h"

#include <fmt/format.h>
#include <limits>
#include <optional>
#include <string_view>
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
  CTRAN_LOG_IF(
      FATAL,
      physicalQps_.empty(),
      "Check failed: !physicalQps_.empty(): At least one physical QP must be provided!");
  CTRAN_LOG_IF(
      FATAL,
      physicalQps_.size() != 1 && !notifyQp_.has_value(),
      "Check failed: physicalQps_.size() == 1 || notifyQp_.has_value(): notifyQp must be provided when using multiple data QPs!");

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

  // Register with VirtualCq
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
      virtualQpNum_(std::move(other.virtualQpNum_)),
      physicalQps_(std::move(other.physicalQps_)),
      qpNumToIdx_(std::move(other.qpNumToIdx_)),
      nextSendPhysicalQpIdx_(std::move(other.nextSendPhysicalQpIdx_)),
      maxMsgCntPerQp_(std::move(other.maxMsgCntPerQp_)),
      maxMsgSize_(std::move(other.maxMsgSize_)),
      nextPhysicalWrId_(std::move(other.nextPhysicalWrId_)),
      deviceCnt_(std::move(other.deviceCnt_)),
      loadBalancingScheme_(std::move(other.loadBalancingScheme_)),
      notifyQp_(std::move(other.notifyQp_)),
      dqplbSeqTracker_(std::move(other.dqplbSeqTracker_)),
      dqplbReceiverInitialized_(std::move(other.dqplbReceiverInitialized_)) {
  other.virtualCq_ = nullptr; // Prevent double-unregister

  // Re-register with VirtualCq
  registerWithVirtualCq();
}

IbvVirtualQp& IbvVirtualQp::operator=(IbvVirtualQp&& other) noexcept {
  if (this != &other) {
    // Unregister current QPs from VirtualCq before moving
    unregisterFromVirtualCq();

    physicalQps_ = std::move(other.physicalQps_);
    notifyQp_ = std::move(other.notifyQp_);
    nextSendPhysicalQpIdx_ = std::move(other.nextSendPhysicalQpIdx_);
    qpNumToIdx_ = std::move(other.qpNumToIdx_);
    maxMsgCntPerQp_ = std::move(other.maxMsgCntPerQp_);
    maxMsgSize_ = std::move(other.maxMsgSize_);
    deviceCnt_ = std::move(other.deviceCnt_);
    loadBalancingScheme_ = std::move(other.loadBalancingScheme_);
    virtualQpNum_ = std::move(other.virtualQpNum_);
    nextPhysicalWrId_ = std::move(other.nextPhysicalWrId_);
    dqplbSeqTracker_ = std::move(other.dqplbSeqTracker_);
    dqplbReceiverInitialized_ = std::move(other.dqplbReceiverInitialized_);
    virtualCq_ = other.virtualCq_;
    isMultiQp_ = other.isMultiQp_;
    sendTracker_ = std::move(other.sendTracker_);
    recvTracker_ = std::move(other.recvTracker_);
    pendingSendNotifyQue_ = std::move(other.pendingSendNotifyQue_);
    pendingRecvNotifyQue_ = std::move(other.pendingRecvNotifyQue_);

    other.virtualCq_ = nullptr; // Prevent double-unregister

    // Re-register with VirtualCq
    registerWithVirtualCq();
  }
  return *this;
}

IbvVirtualQp::~IbvVirtualQp() {
  // Unregister from VirtualCq
  unregisterFromVirtualCq();
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

Status IbvVirtualQp::modifyVirtualQp(
    ibv_qp_attr* attr,
    int attrMask,
    const IbvVirtualQpBusinessCard& businessCard) {
  // If businessCard is not empty, use it to modify QPs with specific
  // dest_qp_num values
  if (!businessCard.qpNums_.empty()) {
    // Make sure the businessCard has the same number of QPs as physicalQps_
    if (businessCard.qpNums_.size() != physicalQps_.size()) {
      return makeUnexpected(Error(
          EINVAL, "BusinessCard QP count doesn't match physical QP count"));
    }

    // Modify each QP with its corresponding dest_qp_num from the businessCard
    for (auto i = 0; i < physicalQps_.size(); i++) {
      attr->dest_qp_num = businessCard.qpNums_.at(i);
      auto maybeModifyQp = physicalQps_.at(i).modifyQp(attr, attrMask);
      if (maybeModifyQp.hasError()) {
        return makeUnexpected(maybeModifyQp.error());
      }
    }
    // Only modify notifyQp if it exists
    if (hasNotifyQp()) {
      attr->dest_qp_num = businessCard.notifyQpNum_;
      auto maybeModifyQp = notifyQp_->modifyQp(attr, attrMask);
      if (maybeModifyQp.hasError()) {
        return makeUnexpected(maybeModifyQp.error());
      }
    }
  } else {
    // If no businessCard provided, modify all QPs with the same attributes
    for (auto& qp : physicalQps_) {
      auto maybeModifyQp = qp.modifyQp(attr, attrMask);
      if (maybeModifyQp.hasError()) {
        return makeUnexpected(maybeModifyQp.error());
      }
    }
    if (hasNotifyQp()) {
      auto maybeModifyQp = notifyQp_->modifyQp(attr, attrMask);
      if (maybeModifyQp.hasError()) {
        return makeUnexpected(maybeModifyQp.error());
      }
    }
  }
  return ok();
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

namespace {

// Wire format, little-endian, no padding:
//   [0..3]   uint32_t numQpNums
//   [4..7]   uint32_t notifyQpNum
//   [8..]    numQpNums * uint32_t qpNums, in order
// Fixed-width fields keep every card the same size for a given QP count, which
// the bootstrap exchange relies on. Byte order is pinned explicitly so the
// encoding does not depend on the host's endianness.
constexpr size_t kU32Size = 4;
constexpr size_t kHeaderSize = 2 * kU32Size;

void appendU32(std::string& out, uint32_t value) {
  for (size_t byte = 0; byte < kU32Size; ++byte) {
    out.push_back(static_cast<char>((value >> (8 * byte)) & 0xffu));
  }
}

uint32_t readU32(std::string_view in, size_t offset) {
  uint32_t value = 0;
  for (size_t byte = 0; byte < kU32Size; ++byte) {
    value |= static_cast<uint32_t>(static_cast<uint8_t>(in[offset + byte]))
        << (8 * byte);
  }
  return value;
}

} // namespace

std::string IbvVirtualQpBusinessCard::serialize() const {
  CTRAN_LOG_IF(
      FATAL,
      qpNums_.size() > std::numeric_limits<uint32_t>::max(),
      "Check failed: qpNums_.size() <= UINT32_MAX: business card holds {} QP numbers, which the wire count cannot encode",
      qpNums_.size());

  std::string out;
  out.reserve(kHeaderSize + qpNums_.size() * kU32Size);
  appendU32(out, static_cast<uint32_t>(qpNums_.size()));
  appendU32(out, notifyQpNum_);
  for (const uint32_t qpNum : qpNums_) {
    appendU32(out, qpNum);
  }
  return out;
}

Expected<IbvVirtualQpBusinessCard> IbvVirtualQpBusinessCard::deserialize(
    const std::string& card) {
  const std::string_view in{card};

  if (in.size() < kHeaderSize) {
    return makeUnexpected(
        Error(EINVAL, "Truncated business card received from remote side"));
  }

  const uint32_t numQpNums = readU32(in, 0);
  const uint32_t notifyQpNum = readU32(in, kU32Size);

  // Reject a length that disagrees with the declared count in either direction,
  // so a truncated card and one with trailing bytes both fail rather than
  // silently decoding to the wrong QP set.
  const size_t expectedSize =
      kHeaderSize + static_cast<size_t>(numQpNums) * kU32Size;
  if (in.size() != expectedSize) {
    return makeUnexpected(Error(
        EINVAL,
        fmt::format(
            "Business card size {} does not match its declared {} QP numbers (expected {})",
            in.size(),
            numQpNums,
            expectedSize)));
  }

  std::vector<uint32_t> qpNums;
  qpNums.reserve(numQpNums);
  for (uint32_t i = 0; i < numQpNums; ++i) {
    qpNums.push_back(
        readU32(in, kHeaderSize + static_cast<size_t>(i) * kU32Size));
  }

  return IbvVirtualQpBusinessCard(std::move(qpNums), notifyQpNum);
}

} // namespace ibverbx
