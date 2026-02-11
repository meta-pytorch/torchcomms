// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

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

// Scatter-gather constants
constexpr int kMaxScatterGatherElements =
    256; // Max total scatter-gather elements per operation
constexpr int kMaxSgBuffersPerWr =
    32; // Max scatter-gather elements per work request (can be configured)

// Command types for coordinator routing and operations
enum class RequestType { SEND = 0, RECV = 1, SEND_NOTIFY = 2 };
enum class LoadBalancingScheme { SPRAY = 0, DQPLB = 1 };

struct Error {
  Error();
  explicit Error(int errNum);
  Error(int errNum, std::string errStr);

  const int errNum{0};
  const std::string errStr;
};

std::ostream& operator<<(std::ostream&, Error const&);

struct VirtualQpRequest {
  RequestType type{RequestType::SEND};
  uint64_t wrId{0};
  uint32_t physicalQpNum{0};
  int32_t deviceId{0};
  uint32_t immData{0};
  int cqIdx{0}; // CQ index from which this request originated
};

struct VirtualQpResponse {
  uint64_t virtualWrId{0};
  bool useDqplb{false};
  int notifyCount{0};
};

struct VirtualCqRequest {
  RequestType type{RequestType::SEND};
  int virtualQpNum{-1};
  int expectedMsgCnt{-1};
  ibv_send_wr* sendWr{nullptr};
  ibv_recv_wr* recvWr{nullptr};
  bool sendExtraNotifyImm{false};
};

} // namespace ibverbx
