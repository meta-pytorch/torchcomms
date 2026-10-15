// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <vector>

#include "NicStatsFeedTypes.h"

namespace meta::comms::colltrace::nicstats {

struct NicStatsRequest {
  std::vector<NicStatsSource> sources;
  std::vector<NicCounterId> counters;
  bool keep{false};
};

class INicStatsFetcher {
 public:
  virtual ~INicStatsFetcher() = default;
  virtual bool request(uint64_t commId, const NicStatsRequest& what) = 0;
};

} // namespace meta::comms::colltrace::nicstats
