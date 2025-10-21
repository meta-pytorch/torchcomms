// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <optional>
#include <string>

#include "comms/ctran/commstate/CommStateX.h"

namespace ctran::commstate {

// load RankTopology from a given filepath
std::optional<ncclx::RankTopology> loadTopology(
    int rank,
    const std::string& filepath);

} // namespace ctran::commstate
