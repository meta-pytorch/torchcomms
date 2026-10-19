// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "NicStatsLatest.h"

namespace meta::comms::colltrace::nicstats {

void NicStatsLatest::put(NicStatsReading /*reading*/) {}

std::vector<NicStatsReading> NicStatsLatest::snapshot() const {
  return {};
}

} // namespace meta::comms::colltrace::nicstats
