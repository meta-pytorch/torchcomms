// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <sstream>

#include <folly/system/ThreadName.h>

#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/LoggingFormat.h"

inline void commNamedThreadStart(
    const char* threadName,
    std::optional<int> rank = std::nullopt,
    std::optional<uint64_t> commHash = std::nullopt,
    std::optional<std::string> commDesc = std::nullopt,
    std::optional<const char*> func = std::nullopt) {
  folly::setThreadName(threadName);
  meta::comms::logger::initThreadMetaData(threadName);
  std::stringstream ss;
  std::vector<std::string> metaVec;
  if (rank.has_value()) {
    metaVec.push_back("rank " + std::to_string(rank.value()));
  }
  if (commHash.has_value()) {
    metaVec.push_back(fmt::format(" commHash {:x}", commHash.value()));
  }
  if (commDesc.has_value()) {
    metaVec.push_back("commDesc " + commDesc.value());
  }
  if (metaVec.size()) {
    ss << "for ";
  }
  if (func.has_value()) {
    metaVec.push_back("at " + std::string(func.value()));
  }
  if (metaVec.size()) {
    ss << folly::join(" ", metaVec);
  }
  CLOGF_SUBSYS(
      INFO, INIT, "[COMM THREAD] Starting {} thread {}", threadName, ss.str());
}
