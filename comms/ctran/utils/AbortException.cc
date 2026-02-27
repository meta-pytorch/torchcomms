// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/utils/AbortException.h"

#include <vector>

#include <fmt/core.h>
#include <fmt/format.h>

#include "comms/utils/commSpecs.h"

namespace ctran::utils {

AbortException::AbortException(
    const std::string context,
    AbortCause cause,
    std::optional<int> rank,
    std::optional<uint64_t> commHash,
    std::optional<std::string> desc)
    : Exception(context, commRemoteError, rank, commHash, desc), cause_(cause) {
  // Override msg to include abort cause
  std::vector<std::string> vec;
  if (rank) {
    vec.emplace_back(fmt::format("rank: {}", *rank));
  }
  if (commHash) {
    vec.emplace_back(fmt::format("commHash: {:x}", *commHash));
  }
  if (desc) {
    vec.emplace_back(fmt::format("desc: {}", *desc));
  }
  vec.emplace_back(fmt::format("abortCause: {}", abortCauseToString(cause_)));

  msg = fmt::format(
      "AbortException: {}, {}, result: {} ({})",
      context,
      fmt::join(vec, ", "),
      meta::comms::commCodeToName(commRemoteError),
      commRemoteError);
}

AbortCause AbortException::cause() const {
  return cause_;
}

std::string AbortException::abortCauseToString(AbortCause cause) {
  switch (cause) {
    case AbortCause::UNKNOWN:
      return "UNKNOWN";
    case AbortCause::USER_INITIATED:
      return "USER_INITIATED";
    case AbortCause::REMOTE_PEER:
      return "REMOTE_PEER";
    case AbortCause::TIMEOUT:
      return "TIMEOUT";
    case AbortCause::SYSTEM_ERROR:
      return "SYSTEM_ERROR";
    default:
      return "INVALID";
  }
}

} // namespace ctran::utils
