// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

// Default HCA prefix
constexpr std::string_view kDefaultHcaPrefix = "";
// Default HCA list
const std::vector<std::string> kDefaultHcaList{};
// Default port
constexpr int kIbAnyPort = -1;
constexpr int kDefaultIbDataDirect = 1;
constexpr int kIbMaxMsgCntPerQp = 100;
constexpr int kIbMaxMsgSizeByte = 100;
constexpr int kIbMaxCqe_ = 100;
constexpr int kNotifyBit = 31;
constexpr uint32_t kSeqNumMask = 0xFFFFFF; // 24 bits
constexpr int kPollCqBatchSize = 32;

enum class LoadBalancingScheme { SPRAY = 0, DQPLB = 1 };

struct Error {
  Error();
  explicit Error(int errNum);
  Error(int errNum, std::string errStr);

  const int errNum{0};
  const std::string errStr;
};

std::ostream& operator<<(std::ostream&, Error const&);

// QpId uniquely identifies a physical QP using both the device ID and QP
// number. This is necessary because different NIC devices can have QPs with the
// same QP number, so we need both fields to uniquely identify a physical QP.
struct QpId {
  int32_t deviceId{-1};
  uint32_t qpNum{0};

  bool operator==(const QpId& other) const {
    return deviceId == other.deviceId && qpNum == other.qpNum;
  }
};

// Hash function for QpId to enable use in hash maps
struct QpIdHash {
  std::size_t operator()(const QpId& id) const {
    auto h1 = std::hash<int32_t>{}(id.deviceId);
    auto h2 = std::hash<uint32_t>{}(id.qpNum);
    return h1 ^ (h2 << 1);
  }
};

} // namespace ibverbx
