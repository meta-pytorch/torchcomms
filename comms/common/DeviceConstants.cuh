// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::device {

// Statically define the warp size (unlike warpSize, this can be used in
// constexpr expressions)
constexpr uint32_t kWarpSize = 32;

} // namespace comms::device
