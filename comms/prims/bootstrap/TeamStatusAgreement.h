// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"

namespace comms::prims::detail {

// Common wire status for NVLink-team exchanges. Payload records place this
// field alongside the data needed by that exchange; status-only stages use it
// directly.
struct TeamStatus {
  uint32_t success{0};
};

inline TeamStatus& getTeamStatus(TeamStatus& status) {
  return status;
}

template <typename Record>
TeamStatus& getTeamStatus(Record& record) {
  return record.status;
}

/**
 * All-gathers one record per rank and turns a rank-local failure into the same
 * team-wide result. The lowest failed rank is the deterministic primary
 * failure. Every rank with a local failure receives its local error text and
 * retains its exception as a nested cause; successful peers receive the
 * primary rank and operation context.
 *
 * Record must be TeamStatus or an allGather-safe wire type with a
 * `TeamStatus status` member. The caller owns any payload validation after
 * this function returns.
 */
template <typename Record, typename Gather>
void gatherAndAgree(
    int rank,
    int nranks,
    std::vector<Record>& records,
    const std::exception_ptr& localError,
    std::string_view operation,
    Gather&& gather) {
  static_assert(std::is_trivially_copyable_v<Record>);

  if (records.size() != static_cast<std::size_t>(nranks)) {
    throw std::invalid_argument(
        "gatherAndAgree records size must equal nranks");
  }

  getTeamStatus(records[static_cast<std::size_t>(rank)]).success =
      localError ? 0u : 1u;
  const int result = std::forward<Gather>(gather)(
      records.data(), static_cast<int>(sizeof(Record)));
  if (result != 0) {
    throw std::runtime_error(
        std::string(operation) + " status allGather failed");
  }

  for (int failedRank = 0; failedRank < nranks; ++failedRank) {
    if (getTeamStatus(records[static_cast<std::size_t>(failedRank)]).success !=
        0) {
      continue;
    }

    std::string message = std::string(operation) + " failed on rank " +
        std::to_string(failedRank);
    if (localError) {
      if (failedRank != rank) {
        message += "; local rank ";
        message += std::to_string(rank);
        message += " also failed";
      }
      try {
        std::rethrow_exception(localError);
      } catch (const std::exception& error) {
        std::throw_with_nested(
            std::runtime_error(message + ": " + error.what()));
      } catch (...) {
        std::throw_with_nested(
            std::runtime_error(message + ": non-standard exception"));
      }
    }
    throw std::runtime_error(message);
  }
}

template <typename Record>
void allGatherAndAgree(
    meta::comms::IBootstrap& bootstrap,
    int rank,
    int nranks,
    std::vector<Record>& records,
    const std::exception_ptr& localError,
    std::string_view operation) {
  gatherAndAgree(
      rank, nranks, records, localError, operation, [&](void* data, int size) {
        return bootstrap.allGather(data, size, rank, nranks).get();
      });
}

} // namespace comms::prims::detail
