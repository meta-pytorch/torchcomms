// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>
#include <string_view>

#include "NicStatsFeedTypes.h"

namespace meta::comms::colltrace::nicstats {

using NicCounterSourceKind = NicStatsSource;

inline constexpr std::array<NicCounterSourceKind, 0> kNicSourceKinds = {};

std::string_view nicSourceKindName(NicCounterSourceKind kind);
std::string_view nicCounterName(NicCounterId id);

} // namespace meta::comms::colltrace::nicstats
