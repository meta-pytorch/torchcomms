// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/colltrace/nicstats/NicStatsFeedTypes.h"
#include "comms/utils/colltrace/v2/CollTraceV2.h"

namespace meta::comms::colltracev2 {

class COLLTRACEV2_API CollTraceV2TestAccessor {
 public:
  // Puts `reading` where `ct.nicStats()` copies from, as a reading thread
  // would.
  static void putNicStats(
      CollTraceV2& ct,
      colltrace::nicstats::NicStatsReading reading);
};

} // namespace meta::comms::colltracev2
