// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <optional>
#include <string>

#include "comms/ctran/utils/Exception.h"

namespace ctran::utils {

// TODO: Probably don't need the enum.
enum class AbortCause {
  UNKNOWN, // Unknown/unspecified abort
  USER_INITIATED, // User called setAbort() explicitly
  REMOTE_PEER, // Remote peer aborted (network failure, crash)
  TIMEOUT, // Operation timed out
  SYSTEM_ERROR, // System resource exhaustion
};

class AbortException : public Exception {
 public:
  explicit AbortException(
      const std::string context,
      AbortCause cause,
      std::optional<int> rank = std::nullopt,
      std::optional<uint64_t> commHash = std::nullopt,
      std::optional<std::string> desc = std::nullopt);

  AbortCause cause() const;

  static std::string abortCauseToString(AbortCause cause);

 private:
  AbortCause cause_;
};

} // namespace ctran::utils
