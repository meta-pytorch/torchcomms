// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include "INicStatsReadingSink.h"

namespace meta::comms::colltrace::nicstats {

class NicStatsReaders;

// Null: the open-source build reads no NIC counters.
std::shared_ptr<NicStatsReaders> startNicStats(
    int cudaDev,
    std::shared_ptr<INicStatsReadingSink> sink) noexcept;

} // namespace meta::comms::colltrace::nicstats
