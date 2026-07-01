// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/utils/AbortException.h"

#include <vector>

#include <fmt/core.h>
#include <fmt/format.h>

#include "comms/utils/commSpecs.h"

namespace ctran::utils {

AbortException::AbortException(
    const std::string context,
    bool retriable,
    std::optional<int> rank,
    std::optional<uint64_t> commHash,
    std::optional<std::string> desc)
    : Exception(context, commRemoteError, rank, commHash, desc),
      retriable_(retriable) {
  // Override msg to include retriable status
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
  vec.emplace_back(fmt::format("retriable: {}", retriable_));

  msg = fmt::format(
      "AbortException: {}, {}, result: {} ({})",
      context,
      fmt::join(vec, ", "),
      meta::comms::commCodeToName(commRemoteError),
      commRemoteError);
}

bool AbortException::isRetriable() const {
  return retriable_;
}

} // namespace ctran::utils
