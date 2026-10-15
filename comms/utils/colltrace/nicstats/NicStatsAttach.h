// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <vector>

#include "INicStatsFetcher.h"
#include "NicStatsFeedTypes.h"

namespace meta::comms::colltrace::nicstats {

// Null: the open-source build reads no NIC counters.
std::shared_ptr<INicStatsFetcher> attachNicStats(int cudaDev);
void prewarmNicStats(int cudaDev) noexcept;
std::vector<NicStatsReading> drainNicStatsReadings();

} // namespace meta::comms::colltrace::nicstats
