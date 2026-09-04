// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::prims {

enum class ReduceScatterOutputInitialization : std::uint8_t {
  COPY_OWN_INPUT,
  ALREADY_INITIALIZED,
};

} // namespace comms::prims
