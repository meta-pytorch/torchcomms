// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/perftrace/TimeInterval.h"

#include <fmt/format.h>
#include <folly/String.h>
#include <vector>

namespace ctran::perftrace {

std::string TimeInterval::toJsonEntry(
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
  auto dur = durationUs();
  if (dur == 0) {
    dur = 1; // Ensure always visible in trace
  }
  return "{\"name\": \"" + name + "\", " + "\"cat\": \"COL\", " +
      "\"id\": " + std::to_string(id) + ", " + "\"ph\": \"X\", " +
      "\"pid\": " + std::to_string(pid) + ", " + "\"args\": {" + argStr + "}," +
      "\"dur\": " + std::to_string(dur) + ", " +
      "\"tid\": " + std::to_string(peer_) + ", " + "\"ts\": " +
      std::to_string(
             std::chrono::duration_cast<std::chrono::microseconds>(
                 start_.time_since_epoch())
                 .count()) +
      "}";
}

} // namespace ctran::perftrace
