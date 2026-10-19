// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "NicStatsFeedTypes.h"

namespace meta::comms::colltrace::nicstats {

class INicStatsReadingSink {
 public:
  virtual ~INicStatsReadingSink() = default;

  virtual void put(NicStatsReading reading) = 0;
};

} // namespace meta::comms::colltrace::nicstats
