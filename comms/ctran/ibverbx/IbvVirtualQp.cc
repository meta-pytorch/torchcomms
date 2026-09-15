// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/IbvVirtualQp.h"

#include <fmt/format.h>
#include <algorithm>
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

// The business card wire format is a fixed two-key JSON object whose values are
// all 10-digit zero-padded decimal strings:
//   {"notifyQpNum":"0000000777","qpNums":["0000000123","0000004567"]}
// Key order and the absence of whitespace match what this exchange has always
// produced, so a peer on either side of this change reads the other correctly.
constexpr std::string_view kQpNumsKey = "qpNums";
constexpr std::string_view kNotifyQpNumKey = "notifyQpNum";

// Advances past JSON whitespace.
size_t skipWs(std::string_view s, size_t pos) {
  while (
      pos < s.size() &&
      (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r')) {
    ++pos;
  }
  return pos;
}

// Reads a double-quoted string with no escape handling: every value in this
// schema is zero-padded digits, so an escape would already be malformed.
std::optional<std::string_view> parseQuoted(std::string_view s, size_t& pos) {
  pos = skipWs(s, pos);
  if (pos >= s.size() || s[pos] != '"') {
    return std::nullopt;
  }
  const size_t begin = ++pos;
  const size_t close = s.find('"', begin);
  if (close == std::string_view::npos) {
    return std::nullopt;
  }
  pos = close + 1;
  return s.substr(begin, close - begin);
}

// Parses a zero-padded decimal into a uint32_t, rejecting anything non-numeric.
std::optional<uint32_t> parseQpNum(std::string_view text) {
  if (text.empty()) {
    return std::nullopt;
  }
  uint64_t value = 0;
  for (const char c : text) {
    if (c < '0' || c > '9') {
      return std::nullopt;
    }
    value = value * 10 + static_cast<uint64_t>(c - '0');
    if (value > std::numeric_limits<uint32_t>::max()) {
      return std::nullopt;
    }
  }
  return static_cast<uint32_t>(value);
}

} // namespace

std::string IbvVirtualQpBusinessCard::serialize() const {
  // Fixed-width values keep every serialized card the same size for a given QP
  // count, which the bootstrap exchange relies on.
  std::string out = fmt::format(
      "{{\"{}\":\"{:010d}\",\"{}\":[",
      kNotifyQpNumKey,
      notifyQpNum_,
      kQpNumsKey);
  for (size_t i = 0; i < qpNums_.size(); ++i) {
    out += fmt::format("{}\"{:010d}\"", i == 0 ? "" : ",", qpNums_[i]);
  }
  out += "]}";
  return out;
}

Expected<IbvVirtualQpBusinessCard> IbvVirtualQpBusinessCard::deserialize(
    const std::string& jsonStr) {
  const std::string_view json{jsonStr};

  const size_t objectStart = skipWs(json, 0);
  if (objectStart >= json.size() || json[objectStart] != '{') {
    return makeUnexpected(
        Error(EINVAL, "Invalid business card received from remote side"));
  }

  // Keys are located by a plain substring search rather than by walking object
  // members. That is only safe because every value in this schema is decimal
  // digits, so no value can contain a key name. Adding a free-form string field
  // would break this and require a real member scan.
  const size_t qpNumsPos = json.find(kQpNumsKey, objectStart);
  if (qpNumsPos == std::string_view::npos) {
    return makeUnexpected(
        Error(EINVAL, "Invalid qpNums array received from remote side"));
  }
  size_t pos = json.find('[', qpNumsPos);
  if (pos == std::string_view::npos) {
    return makeUnexpected(
        Error(EINVAL, "Invalid qpNums array received from remote side"));
  }
  ++pos;

  std::vector<uint32_t> qpNums;
  pos = skipWs(json, pos);
  if (pos < json.size() && json[pos] == ']') {
    ++pos;
  } else {
    while (true) {
      const auto text = parseQuoted(json, pos);
      if (!text) {
        return makeUnexpected(
            Error(EINVAL, "Invalid qpNums array received from remote side"));
      }
      const auto qpNum = parseQpNum(*text);
      if (!qpNum) {
        return makeUnexpected(Error(
            EINVAL, fmt::format("Invalid QP number string format: {}", *text)));
      }
      qpNums.push_back(*qpNum);

      pos = skipWs(json, pos);
      if (pos >= json.size()) {
        return makeUnexpected(
            Error(EINVAL, "Invalid qpNums array received from remote side"));
      }
      if (json[pos] == ']') {
        ++pos;
        break;
      }
      if (json[pos] != ',') {
        return makeUnexpected(
            Error(EINVAL, "Invalid qpNums array received from remote side"));
      }
      ++pos;
    }
  }

  // A present notifyQpNum must parse; only a fully absent key falls back to 0,
  // which is what older peers that never wrote the field send.
  uint32_t notifyQpNum = 0;
  size_t notifyEnd = objectStart;
  const size_t notifyPos = json.find(kNotifyQpNumKey, objectStart);
  if (notifyPos != std::string_view::npos) {
    size_t valuePos = json.find(':', notifyPos + kNotifyQpNumKey.size());
    if (valuePos == std::string_view::npos) {
      return makeUnexpected(
          Error(EINVAL, "Malformed notifyQpNum received from remote side"));
    }
    ++valuePos;
    const auto text = parseQuoted(json, valuePos);
    if (!text) {
      return makeUnexpected(
          Error(EINVAL, "Malformed notifyQpNum received from remote side"));
    }
    const auto parsed = parseQpNum(*text);
    if (!parsed) {
      return makeUnexpected(Error(
          EINVAL, fmt::format("Invalid notifyQpNum string format: {}", *text)));
    }
    notifyQpNum = *parsed;
    notifyEnd = valuePos;
  }

  // Because the keys are found by substring search, neither value necessarily
  // ends at the last member, so the envelope is closed against whichever of the
  // two ended later. Without this a truncated card or one with trailing bytes
  // would be accepted, both of which the previous folly::parseJson rejected.
  const size_t objectEnd = skipWs(json, std::max(pos, notifyEnd));
  if (objectEnd >= json.size() || json[objectEnd] != '}') {
    return makeUnexpected(
        Error(EINVAL, "Unterminated business card received from remote side"));
  }
  if (skipWs(json, objectEnd + 1) != json.size()) {
    return makeUnexpected(Error(
        EINVAL, "Trailing data after business card received from remote side"));
  }

  return IbvVirtualQpBusinessCard(std::move(qpNums), notifyQpNum);
}

} // namespace ibverbx
