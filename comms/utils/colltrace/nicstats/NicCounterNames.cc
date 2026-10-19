// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "NicCounterNames.h"

namespace meta::comms::colltrace::nicstats {

std::string_view nicSourceKindName(NicCounterSourceKind /*kind*/) {
  return "unknown";
}

std::string_view nicCounterName(NicCounterId /*id*/) {
  return "unknown";
}

std::optional<NicCounterId> nicCounterIdFromName(std::string_view /*name*/) {
  return std::nullopt;
}

} // namespace meta::comms::colltrace::nicstats
