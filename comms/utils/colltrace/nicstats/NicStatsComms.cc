// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "NicStatsComms.h"

namespace meta::comms::colltrace::nicstats {

bool attachNicStatsComm(
    std::string_view /*commDesc*/,
    int /*cudaDev*/) noexcept {
  return false;
}

void detachNicStatsComm(std::string_view /*commDesc*/) noexcept {}

bool requestNicStats(
    std::string_view /*commDesc*/,
    const NicStatsRequest& /*what*/) noexcept {
  return false;
}

bool requestNicStatsForProcess(const NicStatsRequest& /*what*/) noexcept {
  return false;
}

std::string nicStatsCommDesc(uint64_t /*commId*/) {
  return {};
}

void setNicStatsAttacherForTest(
    const std::function<
        std::shared_ptr<INicStatsFetcher>(int cudaDev)>& /*attacher*/) {}

void resetNicStatsCommsForTest() {}

} // namespace meta::comms::colltrace::nicstats
