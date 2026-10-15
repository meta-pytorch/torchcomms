// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "INicStatsFetcher.h"

namespace meta::comms::colltrace::nicstats {

// Never attaches: the open-source build reads no NIC counters.
bool attachNicStatsComm(std::string_view commDesc, int cudaDev) noexcept;
void detachNicStatsComm(std::string_view commDesc) noexcept;
bool requestNicStats(
    std::string_view commDesc,
    const NicStatsRequest& what) noexcept;
bool requestNicStatsForProcess(const NicStatsRequest& what) noexcept;
std::string nicStatsCommDesc(uint64_t commId);

void setNicStatsAttacherForTest(
    const std::function<std::shared_ptr<INicStatsFetcher>(int cudaDev)>&
        attacher);
void resetNicStatsCommsForTest();

} // namespace meta::comms::colltrace::nicstats
