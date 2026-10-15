// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "NicStatsAttach.h"

namespace meta::comms::colltrace::nicstats {

std::shared_ptr<INicStatsFetcher> attachNicStats(int /*cudaDev*/) {
  return nullptr;
}

void prewarmNicStats(int /*cudaDev*/) noexcept {}

std::vector<NicStatsReading> drainNicStatsReadings() {
  return {};
}

} // namespace meta::comms::colltrace::nicstats
