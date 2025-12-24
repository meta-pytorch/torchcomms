// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/perftrace/TimestampPoint.h"

#include <fmt/format.h>
#include <folly/String.h>
#include <vector>

namespace ctran::perftrace {

namespace {
// Convert tp to dur in us to be visible in trace
const int kPointToDurUs = 5;
} // namespace

std::string TimestampPoint::toJsonEntry(
    const std::string& name,
    int id,
    const int pid,
    const int seqNum) const {
  std::vector<std::string> argStrs;
  argStrs.push_back(fmt::format("\"seqNum\": \"{}\"", seqNum));
  for (const auto& [key, value] : metaData_) {
    argStrs.push_back(fmt::format("\"{}\": \"{}\"", key, value));
  }
  std::string argStr = folly::join(", ", argStrs);
  return "{\"name\": \"" + name + "\", " + "\"cat\": \"COL\", " +
      "\"id\": " + std::to_string(id) + ", " + "\"ph\": \"X\", " +
      "\"pid\": " + std::to_string(pid) + ", " + "\"args\": {" + argStr + "}," +
      "\"tid\": " + std::to_string(peer_) + ", " + "\"ts\": " +
      std::to_string(
             std::chrono::duration_cast<std::chrono::microseconds>(
                 now_.time_since_epoch())
                 .count()) +
      ", \"dur\": " + std::to_string(kPointToDurUs) + "}";
}

} // namespace ctran::perftrace
