// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <vector>

#include "INicStatsReadingSink.h"

namespace meta::comms::colltrace::nicstats {

// Keeps nothing: the open-source build reads no NIC counters.
class NicStatsLatest : public INicStatsReadingSink {
 public:
  void put(NicStatsReading reading) override;
  std::vector<NicStatsReading> snapshot() const;
};

} // namespace meta::comms::colltrace::nicstats
