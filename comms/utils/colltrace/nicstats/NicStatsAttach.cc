// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "NicStatsAttach.h"

namespace meta::comms::colltrace::nicstats {

std::shared_ptr<NicStatsReaders> startNicStats(
    int /*cudaDev*/,
    std::shared_ptr<INicStatsReadingSink> /*sink*/) noexcept {
  return nullptr;
}

} // namespace meta::comms::colltrace::nicstats
