// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/EventMgrHelperTypes.h"

#include <folly/String.h>
#include <folly/json/json.h>

#include "comms/utils/logger/SpdlogLogger.h"

void EventGlobalRankFilter::initialize(
    const std::vector<std::string>& rankListCvar,
    const std::string& filterName) {
  filterName_ = filterName;
  isAllowed_ = true;

  // If the rank list is empty, then allow all ranks.
  if (rankListCvar.empty()) {
    COMMS_LOG(
        INFO,
        "Empty rank list, skip {} initialization. isAllowed: {}",
        filterName_,
        isAllowed_);
    return;
  }

  // Read globalRank
  const char* rankEnv = getenv("RANK");
  if (rankEnv && strlen(rankEnv)) {
    try {
      globalRank_ = std::stoi(std::string(rankEnv));
    } catch (const std::exception& e) {
      COMMS_LOG(WARN, "Invalid RANK env: {}, error: {}", rankEnv, e.what());
      // Do not throw exception here, just skip this invalid RANK env.
    }
  }

  if (globalRank_ < 0) {
    COMMS_LOG(
        INFO,
        "Cannot get global rank, skip {} initialization. isAllowed: {}",
        filterName_,
        isAllowed_);
    return;
  }

  // Both globalRank and allowed list are set. By default disallow all ranks,
  // and allow only  if the global rank is in the list.
  isAllowed_ = false;
  for (const auto& rankStr : rankListCvar) {
    try {
      int rank = std::stoi(rankStr);
      if (rank == globalRank_) {
        isAllowed_ = true;
        break;
      }
    } catch (const std::exception& e) {
      COMMS_LOG(WARN, "Invalid rank string: {}, error: {}", rankStr, e.what());
      // Do not throw exception here, just skip this invalid rank string.
    }
  }

  COMMS_LOG(
      INFO,
      "Initialized {}, globalRank: {}, isAllowed: {}",
      filterName_,
      globalRank_,
      isAllowed_);
}
